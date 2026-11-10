package milvus

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"goodkind.io/lm-semantic-search/collection"
)

// SupersetExpression renders a filter over occurrence columns as a filter over
// vector collection superset columns. Every vector with at least one occurrence
// that matches filter also matches the rendered expression. An empty expression
// matches every vector. A leaf the superset columns cannot narrow, such as a
// null test or a negation, matches every vector.
func SupersetExpression(filter *collection.Filter, declaration collection.Declaration) (string, error) {
	if filter == nil {
		return "", nil
	}
	columns := make(map[string]collection.ScalarColumn, len(declaration.Scalars))
	for _, declared := range declaration.Scalars {
		columns[declared.Name] = declared
	}
	expression, _, err := supersetNode(*filter, columns)
	return expression, err
}

// supersetNode returns the rendered expression and whether the node narrows
// the vectors at all. A node that does not narrow renders empty.
func supersetNode(filter collection.Filter, columns map[string]collection.ScalarColumn) (string, bool, error) {
	switch filter.Kind {
	case collection.FilterAll:
		clauses := make([]string, 0, len(filter.Children))
		for _, child := range filter.Children {
			clause, narrows, err := supersetNode(child, columns)
			if err != nil {
				return "", false, err
			}
			if narrows {
				clauses = append(clauses, "("+clause+")")
			}
		}
		if len(clauses) == 0 {
			return "", false, nil
		}
		return strings.Join(clauses, " and "), true, nil
	case collection.FilterAny:
		clauses := make([]string, 0, len(filter.Children))
		for _, child := range filter.Children {
			clause, narrows, err := supersetNode(child, columns)
			if err != nil {
				return "", false, err
			}
			if !narrows {
				return "", false, nil
			}
			clauses = append(clauses, "("+clause+")")
		}
		if len(clauses) == 0 {
			return "", false, nil
		}
		return strings.Join(clauses, " or "), true, nil
	case collection.FilterNot, collection.FilterIsNull, collection.FilterIsPresent:
		return "", false, nil
	case collection.FilterEquals, collection.FilterIn:
		return supersetMembership(filter, columns)
	case collection.FilterRange:
		return supersetRange(filter, columns)
	default:
		return "", false, fmt.Errorf("unknown collection filter kind %q", filter.Kind)
	}
}

func supersetMembership(filter collection.Filter, columns map[string]collection.ScalarColumn) (string, bool, error) {
	declared, found := columns[filter.Column]
	if !found {
		return "", false, nil
	}
	if len(filter.Values) == 0 {
		return "", false, fmt.Errorf("membership filter on %s has no values", filter.Column)
	}
	switch declared.Type {
	case collection.ScalarTypeString:
		literals := make([]string, 0, len(filter.Values))
		for _, value := range filter.Values {
			normalized, _ := SanitizeUTF8(value.String)
			literals = append(literals, `"`+collection.EscapeString(normalized)+`"`)
		}
		expression := fmt.Sprintf("array_contains_any(%s, [%s]) or %s == true",
			supersetValuesField(declared.Name), strings.Join(literals, ", "), supersetOverflowField(declared.Name))
		return expression, true, nil
	case collection.ScalarTypeBool:
		literals := make([]string, 0, len(filter.Values))
		for _, value := range filter.Values {
			literals = append(literals, strconv.FormatBool(value.Bool))
		}
		return fmt.Sprintf("array_contains_any(%s, [%s])", supersetValuesField(declared.Name), strings.Join(literals, ", ")), true, nil
	case collection.ScalarTypeInt64:
		values := make([]int64, 0, len(filter.Values))
		for _, value := range filter.Values {
			values = append(values, value.Int64)
		}
		lowest := slices.Min(values)
		highest := slices.Max(values)
		expression := fmt.Sprintf("%s <= %d and %s >= %d",
			supersetMinimumField(declared.Name), highest, supersetMaximumField(declared.Name), lowest)
		return expression, true, nil
	default:
		return "", false, nil
	}
}

func supersetRange(filter collection.Filter, columns map[string]collection.ScalarColumn) (string, bool, error) {
	declared, found := columns[filter.Column]
	if !found || declared.Type != collection.ScalarTypeInt64 {
		return "", false, nil
	}
	clauses := make([]string, 0, 2)
	if filter.Lower != nil {
		clauses = append(clauses, fmt.Sprintf("%s >= %d", supersetMaximumField(declared.Name), *filter.Lower))
	}
	if filter.Upper != nil {
		clauses = append(clauses, fmt.Sprintf("%s < %d", supersetMinimumField(declared.Name), *filter.Upper))
	}
	if len(clauses) == 0 {
		return "", false, fmt.Errorf("range filter on %s has no bound", filter.Column)
	}
	return strings.Join(clauses, " and "), true, nil
}
