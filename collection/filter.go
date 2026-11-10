package collection

// FilterKind is the closed set of typed filter tree nodes.
type FilterKind string

const (
	// FilterAll matches a row when every child matches it.
	FilterAll FilterKind = "all"
	// FilterAny matches a row when at least one child matches it.
	FilterAny FilterKind = "any"
	// FilterNot matches a row when its single child rejects it with a
	// concrete value. A comparison on a null value rejects the row either way.
	FilterNot FilterKind = "not"
	// FilterEquals tests that the column value equals Values[0].
	FilterEquals FilterKind = "equals"
	// FilterIn tests that the column value equals one of Values.
	FilterIn FilterKind = "in"
	// FilterRange tests that the int64 column value is at least Lower and
	// below Upper. A nil bound is open.
	FilterRange FilterKind = "range"
	// FilterIsNull tests that the column value is null.
	FilterIsNull FilterKind = "is_null"
	// FilterIsPresent tests that the column value is not null.
	FilterIsPresent FilterKind = "is_present"
)

// Filter is one node of a typed filter tree over declared scalar columns. All
// and Any use Children, Not uses its single child, and every other kind is a
// leaf that tests Column. The caller validates a tree against the collection
// declaration before any store compiles or evaluates it.
type Filter struct {
	Kind     FilterKind
	Children []Filter
	Column   string
	Values   []ScalarValue
	Lower    *int64
	Upper    *int64
}

// AllOf returns an all node over children.
func AllOf(children ...Filter) Filter {
	return Filter{Kind: FilterAll, Children: children, Column: "", Values: nil, Lower: nil, Upper: nil}
}

// AnyOf returns an any node over children.
func AnyOf(children ...Filter) Filter {
	return Filter{Kind: FilterAny, Children: children, Column: "", Values: nil, Lower: nil, Upper: nil}
}

// Negate returns a not node over child.
func Negate(child Filter) Filter {
	return Filter{Kind: FilterNot, Children: []Filter{child}, Column: "", Values: nil, Lower: nil, Upper: nil}
}

// ColumnEquals returns an equality leaf.
func ColumnEquals(column string, value ScalarValue) Filter {
	return Filter{Kind: FilterEquals, Children: nil, Column: column, Values: []ScalarValue{value}, Lower: nil, Upper: nil}
}

// ColumnIn returns a set membership leaf.
func ColumnIn(column string, values []ScalarValue) Filter {
	return Filter{Kind: FilterIn, Children: nil, Column: column, Values: values, Lower: nil, Upper: nil}
}

// ColumnRange returns a half-open int64 range leaf. A nil bound is open.
func ColumnRange(column string, lower *int64, upper *int64) Filter {
	return Filter{Kind: FilterRange, Children: nil, Column: column, Values: nil, Lower: lower, Upper: upper}
}

// ColumnIsNull returns a null test leaf.
func ColumnIsNull(column string) Filter {
	return Filter{Kind: FilterIsNull, Children: nil, Column: column, Values: nil, Lower: nil, Upper: nil}
}

// ColumnIsPresent returns a non-null test leaf.
func ColumnIsPresent(column string) Filter {
	return Filter{Kind: FilterIsPresent, Children: nil, Column: column, Values: nil, Lower: nil, Upper: nil}
}

// StringValues converts strings to string scalar values.
func StringValues(values []string) []ScalarValue {
	converted := make([]ScalarValue, 0, len(values))
	for _, value := range values {
		converted = append(converted, StringScalar(value))
	}
	return converted
}
