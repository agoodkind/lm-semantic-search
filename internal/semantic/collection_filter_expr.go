package semantic

import (
	"fmt"
	"strconv"
	"strings"

	"goodkind.io/lm-semantic-search/internal/model"
)

// CollectionFilterKind is the closed set of typed filter tree nodes.
type CollectionFilterKind string

const (
	// CollectionFilterAll keeps a row when every child keeps it.
	CollectionFilterAll CollectionFilterKind = "all"
	// CollectionFilterAny keeps a row when at least one child keeps it.
	CollectionFilterAny CollectionFilterKind = "any"
	// CollectionFilterNot keeps a row when its single child rejects it with a
	// concrete value. A comparison on a null value rejects the row either way.
	CollectionFilterNot CollectionFilterKind = "not"
	// CollectionFilterEquals keeps a row when the column value equals Values[0].
	CollectionFilterEquals CollectionFilterKind = "equals"
	// CollectionFilterIn keeps a row when the column value equals one of Values.
	CollectionFilterIn CollectionFilterKind = "in"
	// CollectionFilterRange keeps a row when the int64 column value is at least
	// Lower and below Upper. A nil bound is open.
	CollectionFilterRange CollectionFilterKind = "range"
	// CollectionFilterIsNull keeps a row when the column value is null.
	CollectionFilterIsNull CollectionFilterKind = "is_null"
	// CollectionFilterIsPresent keeps a row when the column value is not null.
	CollectionFilterIsPresent CollectionFilterKind = "is_present"
)

// ScalarValue is one typed scalar value. Type selects the field among String,
// Bool, and Int64 that stores the value.
type ScalarValue struct {
	Type   model.ScalarType
	String string
	Bool   bool
	Int64  int64
}

// StringScalar returns a string scalar value.
func StringScalar(value string) ScalarValue {
	return ScalarValue{Type: model.ScalarTypeString, String: value, Bool: false, Int64: 0}
}

// BoolScalar returns a bool scalar value.
func BoolScalar(value bool) ScalarValue {
	return ScalarValue{Type: model.ScalarTypeBool, String: "", Bool: value, Int64: 0}
}

// Int64Scalar returns an int64 scalar value.
func Int64Scalar(value int64) ScalarValue {
	return ScalarValue{Type: model.ScalarTypeInt64, String: "", Bool: false, Int64: value}
}

// CollectionFilter is one node of a typed filter tree over declared scalar
// columns. All and Any use Children, Not uses its single child, and every
// other kind is a leaf that tests Column. The daemon validates a tree against
// the collection declaration before any store compiles or evaluates it.
type CollectionFilter struct {
	Kind     CollectionFilterKind
	Children []CollectionFilter
	Column   string
	Values   []ScalarValue
	Lower    *int64
	Upper    *int64
}

// AllOf returns an all node over children.
func AllOf(children ...CollectionFilter) CollectionFilter {
	return CollectionFilter{Kind: CollectionFilterAll, Children: children, Column: "", Values: nil, Lower: nil, Upper: nil}
}

// AnyOf returns an any node over children.
func AnyOf(children ...CollectionFilter) CollectionFilter {
	return CollectionFilter{Kind: CollectionFilterAny, Children: children, Column: "", Values: nil, Lower: nil, Upper: nil}
}

// Negate returns a not node over child.
func Negate(child CollectionFilter) CollectionFilter {
	return CollectionFilter{Kind: CollectionFilterNot, Children: []CollectionFilter{child}, Column: "", Values: nil, Lower: nil, Upper: nil}
}

// ColumnEquals returns an equality leaf.
func ColumnEquals(column string, value ScalarValue) CollectionFilter {
	return CollectionFilter{Kind: CollectionFilterEquals, Children: nil, Column: column, Values: []ScalarValue{value}, Lower: nil, Upper: nil}
}

// ColumnIn returns a set membership leaf.
func ColumnIn(column string, values []ScalarValue) CollectionFilter {
	return CollectionFilter{Kind: CollectionFilterIn, Children: nil, Column: column, Values: values, Lower: nil, Upper: nil}
}

// ColumnRange returns a half-open int64 range leaf. A nil bound is open.
func ColumnRange(column string, lower *int64, upper *int64) CollectionFilter {
	return CollectionFilter{Kind: CollectionFilterRange, Children: nil, Column: column, Values: nil, Lower: lower, Upper: upper}
}

// ColumnIsNull returns a null test leaf.
func ColumnIsNull(column string) CollectionFilter {
	return CollectionFilter{Kind: CollectionFilterIsNull, Children: nil, Column: column, Values: nil, Lower: nil, Upper: nil}
}

// ColumnIsPresent returns a non-null test leaf.
func ColumnIsPresent(column string) CollectionFilter {
	return CollectionFilter{Kind: CollectionFilterIsPresent, Children: nil, Column: column, Values: nil, Lower: nil, Upper: nil}
}

// StringValues converts strings to string scalar values.
func StringValues(values []string) []ScalarValue {
	converted := make([]ScalarValue, 0, len(values))
	for _, value := range values {
		converted = append(converted, StringScalar(value))
	}
	return converted
}

// filterTemplateParam is one Milvus expression template parameter. Type
// selects the slice among Strings, Bools, and Int64s that stores the values.
type filterTemplateParam struct {
	Name    string
	Type    model.ScalarType
	Strings []string
	Bools   []bool
	Int64s  []int64
}

// compiledFilter is a Milvus boolean expression and the template parameters
// its placeholders bind.
type compiledFilter struct {
	Expression string
	Params     []filterTemplateParam
}

// compileCollectionFilterExpr renders a validated filter tree as a Milvus
// boolean expression. A nil tree renders the empty expression, which searches
// the whole collection. Every set membership leaf renders as a placeholder
// such as `role in {p0}` and binds its values as one typed template parameter,
// so a membership set of any size runs in one search. An equality leaf renders
// its literal inline with the existing Milvus string escape. An all node joins
// its children with " and " and wraps only an any child in parentheses.
func compileCollectionFilterExpr(filter *CollectionFilter) (compiledFilter, error) {
	compiler := filterCompiler{params: nil}
	if filter == nil {
		return compiledFilter{Expression: "", Params: nil}, nil
	}
	expression, err := compiler.node(*filter)
	if err != nil {
		return compiledFilter{Expression: "", Params: nil}, err
	}
	return compiledFilter{Expression: expression, Params: compiler.params}, nil
}

// filterCompiler numbers template placeholders in tree order.
type filterCompiler struct {
	params []filterTemplateParam
}

func (compiler *filterCompiler) node(filter CollectionFilter) (string, error) {
	switch filter.Kind {
	case CollectionFilterAll:
		return compiler.group(filter.Children, " and ", needsParenthesesInAll)
	case CollectionFilterAny:
		return compiler.group(filter.Children, " or ", needsParenthesesInAny)
	case CollectionFilterNot:
		if len(filter.Children) != 1 {
			return "", fmt.Errorf("not filter has %d children, want 1", len(filter.Children))
		}
		child, err := compiler.node(filter.Children[0])
		if err != nil {
			return "", err
		}
		return "not (" + child + ")", nil
	case CollectionFilterEquals:
		if len(filter.Values) != 1 {
			return "", fmt.Errorf("equals filter on %s has %d values, want 1", filter.Column, len(filter.Values))
		}
		return filter.Column + " == " + milvusLiteral(filter.Values[0]), nil
	case CollectionFilterIn:
		return compiler.membership(filter)
	case CollectionFilterRange:
		return compileRange(filter)
	case CollectionFilterIsNull:
		return filter.Column + " IS NULL", nil
	case CollectionFilterIsPresent:
		return filter.Column + " IS NOT NULL", nil
	default:
		return "", fmt.Errorf("unknown collection filter kind %q", filter.Kind)
	}
}

func (compiler *filterCompiler) group(children []CollectionFilter, separator string, needsParentheses func(CollectionFilter) bool) (string, error) {
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
func (compiler *filterCompiler) membership(filter CollectionFilter) (string, error) {
	if len(filter.Values) == 0 {
		return "", fmt.Errorf("membership filter on %s has no values", filter.Column)
	}
	param := filterTemplateParam{
		Name:    "p" + strconv.Itoa(len(compiler.params)),
		Type:    filter.Values[0].Type,
		Strings: nil,
		Bools:   nil,
		Int64s:  nil,
	}
	for _, value := range filter.Values {
		switch param.Type {
		case model.ScalarTypeBool:
			param.Bools = append(param.Bools, value.Bool)
		case model.ScalarTypeInt64:
			param.Int64s = append(param.Int64s, value.Int64)
		case model.ScalarTypeString:
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
func needsParenthesesInAll(child CollectionFilter) bool {
	return child.Kind == CollectionFilterAny
}

// needsParenthesesInAny reports whether a child of an or-joined group needs
// parentheses. An all group, a nested any group, and a range with both bounds
// each render and or or, and each needs them.
func needsParenthesesInAny(child CollectionFilter) bool {
	switch child.Kind {
	case CollectionFilterAll, CollectionFilterAny:
		return true
	case CollectionFilterRange:
		return child.Lower != nil && child.Upper != nil
	case CollectionFilterNot, CollectionFilterEquals, CollectionFilterIn, CollectionFilterIsNull, CollectionFilterIsPresent:
		return false
	default:
		return false
	}
}

func compileRange(filter CollectionFilter) (string, error) {
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

func milvusLiteral(value ScalarValue) string {
	switch value.Type {
	case model.ScalarTypeBool:
		return strconv.FormatBool(value.Bool)
	case model.ScalarTypeInt64:
		return strconv.FormatInt(value.Int64, 10)
	case model.ScalarTypeString:
		return `"` + escapeMilvusString(value.String) + `"`
	default:
		return `"` + escapeMilvusString(value.String) + `"`
	}
}
