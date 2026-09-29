package library

import (
	"fmt"
	"math"
	"strings"
	"unicode/utf8"
)

// ValidateSearchRequest reports whether request can search the namespace that
// spec declares under the configured request limits. It checks the namespace,
// the query text, the page size, the score floor, the grouping column, and
// every node of the filter tree, and returns an error that wraps
// [ErrInvalidRequest] for the first violation. It does not decode the cursor;
// Search checks a cursor against its persisted snapshot.
//
// The query is valid UTF-8 with a non-whitespace character and no NUL byte,
// because the lexical analyzer ignores every byte after the first NUL.
// PageSize is positive, and MinScore is finite and not negative. GroupBy is
// empty with a zero PerGroupLimit, or a declared column with a positive
// PerGroupLimit.
//
// Each filter node selects one operator and sets only that operator's fields.
// All and Any take one or more children, and Not takes exactly one. Equal takes
// one value, In takes one or more values, Range takes a Lower or Upper bound on
// an [Int64] column, and Prefix takes nonempty literal text on a [String]
// column. IsNull and IsPresent take only the column. Every column is declared,
// and every value matches its column type. A comparison string or prefix
// longer than the column MaxLength is valid and matches no row. A comparison
// value cannot be null, because a comparison with null is unknown and matches
// no row; IsNull selects null values. MaxFilterDepth counts nodes on the deepest path, and
// MaxFilterValues counts Values entries plus Lower and Upper bounds across the
// tree.
func (config Config) ValidateSearchRequest(spec NamespaceSpec, request SearchRequest) error {
	if err := validateRequestLimits(config); err != nil {
		return err
	}
	if err := spec.Validate(); err != nil {
		return err
	}
	if request.Namespace != spec.ID {
		return invalidRequest(fmt.Sprintf(
			"search namespace %q does not match declaration %q",
			request.Namespace,
			spec.ID,
		))
	}
	if err := validateSearchQuery(config, request.Query); err != nil {
		return err
	}
	if err := validateSearchPaging(config, request); err != nil {
		return err
	}
	columns := declaredColumns(spec)
	if err := validateSearchGrouping(spec.ID, columns, request); err != nil {
		return err
	}
	if request.Filter == nil {
		return nil
	}
	validator := filterValidator{
		namespace: spec.ID,
		columns:   columns,
		maxDepth:  config.MaxFilterDepth,
		maxValues: config.MaxFilterValues,
	}
	return validator.node(*request.Filter, "Filter", 1)
}

func validateSearchQuery(config Config, query string) error {
	if !utf8.ValidString(query) {
		return invalidRequest("search query is not valid UTF-8")
	}
	if strings.IndexByte(query, 0) >= 0 {
		return invalidRequest("search query contains a NUL byte; replace it with a space")
	}
	if strings.TrimSpace(query) == "" {
		return invalidRequest("search query contains no non-whitespace character")
	}
	if config.MaxQueryBytes > 0 && len(query) > config.MaxQueryBytes {
		return invalidRequest(fmt.Sprintf(
			"search query is %d bytes, over MaxQueryBytes %d",
			len(query),
			config.MaxQueryBytes,
		))
	}
	return nil
}

func validateSearchPaging(config Config, request SearchRequest) error {
	if request.PageSize <= 0 {
		return invalidRequest(fmt.Sprintf("search PageSize %d must be positive", request.PageSize))
	}
	if config.MaxPageSize > 0 && request.PageSize > config.MaxPageSize {
		return invalidRequest(fmt.Sprintf(
			"search PageSize %d is over MaxPageSize %d",
			request.PageSize,
			config.MaxPageSize,
		))
	}
	if math.IsNaN(request.MinScore) || math.IsInf(request.MinScore, 0) || request.MinScore < 0 {
		return invalidRequest(fmt.Sprintf(
			"search MinScore %v must be finite and not negative; zero applies no floor",
			request.MinScore,
		))
	}
	return nil
}

func validateSearchGrouping(namespace string, columns map[string]ScalarColumn, request SearchRequest) error {
	if request.GroupBy == "" {
		if request.PerGroupLimit != 0 {
			return invalidRequest(fmt.Sprintf(
				"search PerGroupLimit %d requires a GroupBy column",
				request.PerGroupLimit,
			))
		}
		return nil
	}
	if _, declared := columns[request.GroupBy]; !declared {
		return invalidRequest(fmt.Sprintf(
			"search GroupBy column %q is not declared by namespace %q",
			request.GroupBy,
			namespace,
		))
	}
	if request.PerGroupLimit <= 0 {
		return invalidRequest(fmt.Sprintf(
			"search GroupBy %q PerGroupLimit %d must be positive",
			request.GroupBy,
			request.PerGroupLimit,
		))
	}
	return nil
}

func declaredColumns(spec NamespaceSpec) map[string]ScalarColumn {
	columns := make(map[string]ScalarColumn, len(spec.Scalars))
	for _, column := range spec.Scalars {
		columns[column.Name] = column
	}
	return columns
}

// filterValidator walks one filter tree and counts its scalar values against
// the configured limit.
type filterValidator struct {
	namespace string
	columns   map[string]ScalarColumn
	maxDepth  int
	maxValues int
	values    int
}

// filterFields records which operand fields one filter node sets.
type filterFields struct {
	column   bool
	children bool
	values   bool
	bounds   bool
	prefix   bool
}

func (validator *filterValidator) node(filter Filter, path string, depth int) error {
	if validator.maxDepth > 0 && depth > validator.maxDepth {
		return invalidRequest(fmt.Sprintf(
			"%s is at filter depth %d, over MaxFilterDepth %d",
			path,
			depth,
			validator.maxDepth,
		))
	}
	switch filter.Op {
	case All, Any, Not:
		return validator.logical(filter, path, depth)
	case Equal, In:
		return validator.membership(filter, path)
	case Range:
		return validator.rangeBounds(filter, path)
	case Prefix:
		return validator.prefix(filter, path)
	case IsNull, IsPresent:
		if err := validator.operands(filter, path, filterFields{column: true}); err != nil {
			return err
		}
		_, err := validator.column(filter, path)
		return err
	default:
		return invalidRequest(fmt.Sprintf("%s operator %d is not a FilterOp", path, filter.Op))
	}
}

func (validator *filterValidator) logical(filter Filter, path string, depth int) error {
	if err := validator.operands(filter, path, filterFields{children: true}); err != nil {
		return err
	}
	if filter.Op == Not && len(filter.Children) != 1 {
		return invalidRequest(fmt.Sprintf("%s Not has %d children, want exactly 1", path, len(filter.Children)))
	}
	if len(filter.Children) == 0 {
		return invalidRequest(fmt.Sprintf("%s %s has no children", path, filterOpName(filter.Op)))
	}
	for index, child := range filter.Children {
		childPath := fmt.Sprintf("%s.Children[%d]", path, index)
		if err := validator.node(child, childPath, depth+1); err != nil {
			return err
		}
	}
	return nil
}

func (validator *filterValidator) membership(filter Filter, path string) error {
	if err := validator.operands(filter, path, filterFields{column: true, values: true}); err != nil {
		return err
	}
	column, err := validator.column(filter, path)
	if err != nil {
		return err
	}
	if filter.Op == Equal && len(filter.Values) != 1 {
		return invalidRequest(fmt.Sprintf("%s Equal has %d values, want exactly 1", path, len(filter.Values)))
	}
	if len(filter.Values) == 0 {
		return invalidRequest(path + " In has no values")
	}
	for index, value := range filter.Values {
		valuePath := fmt.Sprintf("%s.Values[%d]", path, index)
		if err := validator.value(column, value, valuePath); err != nil {
			return err
		}
	}
	return nil
}

func (validator *filterValidator) rangeBounds(filter Filter, path string) error {
	if err := validator.operands(filter, path, filterFields{column: true, bounds: true}); err != nil {
		return err
	}
	column, err := validator.column(filter, path)
	if err != nil {
		return err
	}
	if column.Type != Int64 {
		return invalidRequest(fmt.Sprintf("%s Range column %q is not Int64", path, column.Name))
	}
	if filter.Lower == nil && filter.Upper == nil {
		return invalidRequest(path + " Range sets neither Lower nor Upper")
	}
	if filter.Lower != nil {
		if err := validator.value(column, *filter.Lower, path+".Lower"); err != nil {
			return err
		}
	}
	if filter.Upper != nil {
		if err := validator.value(column, *filter.Upper, path+".Upper"); err != nil {
			return err
		}
	}
	return nil
}

func (validator *filterValidator) prefix(filter Filter, path string) error {
	if err := validator.operands(filter, path, filterFields{column: true, prefix: true}); err != nil {
		return err
	}
	column, err := validator.column(filter, path)
	if err != nil {
		return err
	}
	if column.Type != String {
		return invalidRequest(fmt.Sprintf("%s Prefix column %q is not String", path, column.Name))
	}
	if filter.Prefix == "" {
		return invalidRequest(path + " Prefix text is empty")
	}
	if !utf8.ValidString(filter.Prefix) {
		return invalidRequest(path + " Prefix text is not valid UTF-8")
	}
	return nil
}

// operands rejects every operand field that the node's operator does not use.
func (validator *filterValidator) operands(filter Filter, path string, allowed filterFields) error {
	set := filterFields{
		column:   filter.Column != "",
		children: len(filter.Children) > 0,
		values:   len(filter.Values) > 0,
		bounds:   filter.Lower != nil || filter.Upper != nil,
		prefix:   filter.Prefix != "",
	}
	fields := []struct {
		name    string
		set     bool
		allowed bool
	}{
		{name: "Column", set: set.column, allowed: allowed.column},
		{name: "Children", set: set.children, allowed: allowed.children},
		{name: "Values", set: set.values, allowed: allowed.values},
		{name: "Lower or Upper", set: set.bounds, allowed: allowed.bounds},
		{name: "Prefix", set: set.prefix, allowed: allowed.prefix},
	}
	for _, field := range fields {
		if field.set && !field.allowed {
			return invalidRequest(fmt.Sprintf("%s %s sets %s", path, filterOpName(filter.Op), field.name))
		}
	}
	return nil
}

func (validator *filterValidator) column(filter Filter, path string) (ScalarColumn, error) {
	if filter.Column == "" {
		return ScalarColumn{}, invalidRequest(fmt.Sprintf("%s %s names no column", path, filterOpName(filter.Op)))
	}
	column, declared := validator.columns[filter.Column]
	if !declared {
		return ScalarColumn{}, invalidRequest(fmt.Sprintf(
			"%s column %q is not declared by namespace %q",
			path,
			filter.Column,
			validator.namespace,
		))
	}
	return column, nil
}

func (validator *filterValidator) value(column ScalarColumn, value ScalarValue, path string) error {
	validator.values++
	if validator.maxValues > 0 && validator.values > validator.maxValues {
		return invalidRequest(fmt.Sprintf(
			"%s is filter value %d, over MaxFilterValues %d",
			path,
			validator.values,
			validator.maxValues,
		))
	}
	if value.Null {
		return invalidRequest(fmt.Sprintf("%s compares column %q with null; use IsNull", path, column.Name))
	}
	// MaxLength limits stored values. A longer comparison string is valid and
	// matches no row.
	unbounded := column
	unbounded.MaxLength = math.MaxInt
	if message := scalarValueViolation(unbounded, value); message != "" {
		return invalidRequest(fmt.Sprintf("%s: %s", path, message))
	}
	return nil
}

func filterOpName(op FilterOp) string {
	names := map[FilterOp]string{
		All:       "All",
		Any:       "Any",
		Not:       "Not",
		Equal:     "Equal",
		In:        "In",
		Range:     "Range",
		Prefix:    "Prefix",
		IsNull:    "IsNull",
		IsPresent: "IsPresent",
	}
	if name, known := names[op]; known {
		return name
	}
	return fmt.Sprintf("FilterOp(%d)", op)
}
