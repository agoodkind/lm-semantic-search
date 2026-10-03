package collection

import (
	"fmt"
	"strconv"
	"strings"
)

// TemplateParam is one Milvus expression template parameter. Type selects the
// slice among Strings, Bools, and Int64s that stores the values.
type TemplateParam struct {
	Name    string
	Type    ScalarType
	Strings []string
	Bools   []bool
	Int64s  []int64
}

// CompiledFilter is a Milvus boolean expression and the template parameters
// its placeholders bind.
type CompiledFilter struct {
	Expression string
	Params     []TemplateParam
}

// Compile renders a validated filter tree as a Milvus boolean expression. A
// nil tree renders the empty expression, which searches the whole collection.
// Every set membership leaf renders as a placeholder such as `role in {p0}` and
// binds its values as one typed template parameter. A membership set of any
// size therefore runs in one search. An equality leaf renders its literal
// inline with the existing Milvus string escape. An all node joins its
// children with " and " and wraps only an any child in parentheses.
func Compile(filter *Filter) (CompiledFilter, error) {
	compiler := filterCompiler{params: nil, inline: false}
	if filter == nil {
		return CompiledFilter{Expression: "", Params: nil}, nil
	}
	expression, err := compiler.node(*filter)
	if err != nil {
		return CompiledFilter{Expression: "", Params: nil}, err
	}
	return CompiledFilter{Expression: expression, Params: compiler.params}, nil
}

// CompileInline renders a validated filter tree as a Milvus boolean expression
// with every membership set written inline as `column in [v1, v2]`. A request
// that cannot bind template parameters, such as a delete, uses this form. A nil
// tree renders the empty expression.
func CompileInline(filter *Filter) (string, error) {
	compiler := filterCompiler{params: nil, inline: true}
	if filter == nil {
		return "", nil
	}
	return compiler.node(*filter)
}

// filterCompiler numbers template placeholders in tree order. An inline
// compiler writes membership sets in the expression and binds no parameter.
type filterCompiler struct {
	params []TemplateParam
	inline bool
}

func (compiler *filterCompiler) node(filter Filter) (string, error) {
	switch filter.Kind {
	case FilterAll:
		return compiler.group(filter.Children, " and ", needsParenthesesInAll)
	case FilterAny:
		return compiler.group(filter.Children, " or ", needsParenthesesInAny)
	case FilterNot:
		if len(filter.Children) != 1 {
			return "", fmt.Errorf("not filter has %d children, want 1", len(filter.Children))
		}
		child, err := compiler.node(filter.Children[0])
		if err != nil {
			return "", err
		}
		return "not (" + child + ")", nil
	case FilterEquals:
		if len(filter.Values) != 1 {
			return "", fmt.Errorf("equals filter on %s has %d values, want 1", filter.Column, len(filter.Values))
		}
		return filter.Column + " == " + literal(filter.Values[0]), nil
	case FilterIn:
		return compiler.membership(filter)
	case FilterRange:
		return compileRange(filter)
	case FilterIsNull:
		return filter.Column + " IS NULL", nil
	case FilterIsPresent:
		return filter.Column + " IS NOT NULL", nil
	default:
		return "", fmt.Errorf("unknown collection filter kind %q", filter.Kind)
	}
}

func (compiler *filterCompiler) group(children []Filter, separator string, needsParentheses func(Filter) bool) (string, error) {
	if len(children) == 0 {
		return "", fmt.Errorf("filter group joined by %q has no children", strings.TrimSpace(separator))
	}
	clauses := make([]string, 0, len(children))
	for _, child := range children {
		clause, err := compiler.node(child)
		if err != nil {
			return "", err
		}
		if needsParentheses(child) {
			clause = "(" + clause + ")"
		}
		clauses = append(clauses, clause)
	}
	return strings.Join(clauses, separator), nil
}

// membership renders a set membership leaf as a template placeholder and
// records its values as one typed parameter. Validation guarantees every
// value has the column's declared type.
func (compiler *filterCompiler) membership(filter Filter) (string, error) {
	if len(filter.Values) == 0 {
		return "", fmt.Errorf("membership filter on %s has no values", filter.Column)
	}
	if compiler.inline {
		literals := make([]string, 0, len(filter.Values))
		for _, value := range filter.Values {
			literals = append(literals, literal(value))
		}
		return filter.Column + " in [" + strings.Join(literals, ", ") + "]", nil
	}
	param := TemplateParam{
		Name:    "p" + strconv.Itoa(len(compiler.params)),
		Type:    filter.Values[0].Type,
		Strings: nil,
		Bools:   nil,
		Int64s:  nil,
	}
	for _, value := range filter.Values {
		switch param.Type {
		case ScalarTypeBool:
			param.Bools = append(param.Bools, value.Bool)
		case ScalarTypeInt64:
			param.Int64s = append(param.Int64s, value.Int64)
		case ScalarTypeString:
			param.Strings = append(param.Strings, value.String)
		default:
			return "", fmt.Errorf("membership filter on %s has unsupported type %q", filter.Column, param.Type)
		}
	}
	compiler.params = append(compiler.params, param)
	return filter.Column + " in {" + param.Name + "}", nil
}

// needsParenthesesInAll reports whether a child of an and-joined group needs
// parentheses. Milvus binds and tighter than or, and only an any child renders
// an or. A not node renders its own parentheses.
func needsParenthesesInAll(child Filter) bool {
	return child.Kind == FilterAny
}

// needsParenthesesInAny reports whether a child of an or-joined group needs
// parentheses. An all group, a nested any group, and a range with both bounds
// each render and or or, and each needs them.
func needsParenthesesInAny(child Filter) bool {
	switch child.Kind {
	case FilterAll, FilterAny:
		return true
	case FilterRange:
		return child.Lower != nil && child.Upper != nil
	case FilterNot, FilterEquals, FilterIn, FilterIsNull, FilterIsPresent:
		return false
	default:
		return false
	}
}

func compileRange(filter Filter) (string, error) {
	clauses := make([]string, 0, 2)
	if filter.Lower != nil {
		clauses = append(clauses, filter.Column+" >= "+strconv.FormatInt(*filter.Lower, 10))
	}
	if filter.Upper != nil {
		clauses = append(clauses, filter.Column+" < "+strconv.FormatInt(*filter.Upper, 10))
	}
	if len(clauses) == 0 {
		return "", fmt.Errorf("range filter on %s has no bound", filter.Column)
	}
	return strings.Join(clauses, " and "), nil
}

func literal(value ScalarValue) string {
	switch value.Type {
	case ScalarTypeBool:
		return strconv.FormatBool(value.Bool)
	case ScalarTypeInt64:
		return strconv.FormatInt(value.Int64, 10)
	case ScalarTypeString:
		return `"` + EscapeString(value.String) + `"`
	default:
		return `"` + EscapeString(value.String) + `"`
	}
}

// EscapeString escapes one value for a Milvus string literal. Cursor
// conversation ids can contain raw newlines, so control bytes must become
// parser-safe escapes before any relativePath expression is sent to Milvus.
func EscapeString(value string) string {
	value = strings.ReplaceAll(value, `\`, `\\`)
	value = strings.ReplaceAll(value, `"`, `\"`)

	var builder strings.Builder
	builder.Grow(len(value))
	for index := range len(value) {
		byteValue := value[index]
		switch byteValue {
		case '\n':
			builder.WriteString(`\n`)
		case '\r':
			builder.WriteString(`\r`)
		case '\t':
			builder.WriteString(`\t`)
		default:
			if byteValue < 0x20 {
				fmt.Fprintf(&builder, `\%03o`, byteValue)
				continue
			}
			builder.WriteByte(byteValue)
		}
	}
	return builder.String()
}
