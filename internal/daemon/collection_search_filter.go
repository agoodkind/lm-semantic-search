package daemon

import (
	"fmt"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

const (
	// maxCollectionFilterDepth is the deepest filter tree a search accepts. The
	// root node is depth 1.
	maxCollectionFilterDepth = 16
	// maxCollectionFilterValues is the largest set a membership leaf accepts.
	// The conversation adapter sends an explicit conversation scope as one
	// membership leaf, and the search splits a scope larger than one Milvus
	// membership clause into batches.
	maxCollectionFilterValues = 65_536
)

// validateCollectionSearch checks a typed search against the collection's
// saved declaration before any store work. Every filter column and the group
// column must be declared, every literal must have its column's declared
// type, a range must test an int64 column, and the tree depth and membership
// set sizes must stay within their limits. A column violation returns an
// [adapterr.ColumnError] with the rejected column.
func validateCollectionSearch(collectionID string, declaration model.CollectionDeclaration, filter *semantic.CollectionFilter, groupBy string, perGroupLimit int32) error {
	declared := make(map[string]model.ScalarColumn, len(declaration.Scalars))
	for _, column := range declaration.Scalars {
		declared[column.Name] = column
	}
	if perGroupLimit < 0 {
		return adapterr.NewInvalidArgument(fmt.Sprintf("per_group_limit %d is negative", perGroupLimit))
	}
	if perGroupLimit > 0 && groupBy == "" {
		return adapterr.NewInvalidArgument("per_group_limit requires group_by")
	}
	if groupBy != "" {
		if _, found := declared[groupBy]; !found {
			return undeclaredColumnError(collectionID, groupBy, "group_by")
		}
	}
	if filter == nil {
		return nil
	}
	return validateFilterNode(collectionID, declared, *filter, 1)
}

func validateFilterNode(collectionID string, declared map[string]model.ScalarColumn, filter semantic.CollectionFilter, depth int) error {
	if depth > maxCollectionFilterDepth {
		return adapterr.NewInvalidArgument(fmt.Sprintf("filter tree is deeper than %d levels", maxCollectionFilterDepth))
	}
	switch filter.Kind {
	case semantic.CollectionFilterAll, semantic.CollectionFilterAny:
		if len(filter.Children) == 0 {
			return adapterr.NewInvalidArgument(fmt.Sprintf("filter %s group has no children", filter.Kind))
		}
		for _, child := range filter.Children {
			if err := validateFilterNode(collectionID, declared, child, depth+1); err != nil {
				return err
			}
		}
		return nil
	case semantic.CollectionFilterNot:
		if len(filter.Children) != 1 {
			return adapterr.NewInvalidArgument("filter negate node needs exactly one child")
		}
		return validateFilterNode(collectionID, declared, filter.Children[0], depth+1)
	case semantic.CollectionFilterEquals, semantic.CollectionFilterIn:
		return validateComparisonLeaf(collectionID, declared, filter)
	case semantic.CollectionFilterRange:
		column, err := declaredFilterColumn(collectionID, declared, filter.Column)
		if err != nil {
			return err
		}
		if column.Type != model.ScalarTypeInt64 {
			return adapterr.NewInvalidFilterColumn(column.Name, fmt.Sprintf("range filter needs an int64 column, and column %q is %s", column.Name, column.Type))
		}
		if filter.Lower == nil && filter.Upper == nil {
			return adapterr.NewInvalidFilterColumn(column.Name, fmt.Sprintf("range filter on column %q sets no bound", column.Name))
		}
		return nil
	case semantic.CollectionFilterIsNull, semantic.CollectionFilterIsPresent:
		_, err := declaredFilterColumn(collectionID, declared, filter.Column)
		return err
	default:
		return adapterr.NewInvalidArgument(fmt.Sprintf("unknown filter node kind %q", filter.Kind))
	}
}

func validateComparisonLeaf(collectionID string, declared map[string]model.ScalarColumn, filter semantic.CollectionFilter) error {
	column, err := declaredFilterColumn(collectionID, declared, filter.Column)
	if err != nil {
		return err
	}
	if len(filter.Values) == 0 {
		return adapterr.NewInvalidFilterColumn(column.Name, fmt.Sprintf("%s filter on column %q has no value", filter.Kind, column.Name))
	}
	if filter.Kind == semantic.CollectionFilterEquals && len(filter.Values) != 1 {
		return adapterr.NewInvalidFilterColumn(column.Name, fmt.Sprintf("equals filter on column %q has %d values, want 1", column.Name, len(filter.Values)))
	}
	if len(filter.Values) > maxCollectionFilterValues {
		return adapterr.NewInvalidFilterColumn(column.Name, fmt.Sprintf("membership filter on column %q has %d values, more than the limit of %d", column.Name, len(filter.Values), maxCollectionFilterValues))
	}
	for _, value := range filter.Values {
		if value.Type != column.Type {
			return adapterr.NewInvalidFilterColumn(column.Name, fmt.Sprintf("filter value for column %q is %s, and the column is %s", column.Name, describeFilterValueType(value.Type), column.Type))
		}
	}
	return nil
}

func declaredFilterColumn(collectionID string, declared map[string]model.ScalarColumn, columnName string) (model.ScalarColumn, error) {
	column, found := declared[columnName]
	if !found {
		return model.ScalarColumn{Name: columnName, Type: "", Nullable: false, MaxLength: 0}, undeclaredColumnError(collectionID, columnName, "filter")
	}
	return column, nil
}

func undeclaredColumnError(collectionID string, columnName string, use string) error {
	if columnName == "" {
		return adapterr.NewInvalidFilterColumn(columnName, use+" column is empty")
	}
	return adapterr.NewInvalidFilterColumn(columnName, fmt.Sprintf("%s column %q is not declared in collection %q", use, columnName, collectionID))
}

func describeFilterValueType(scalarType model.ScalarType) string {
	if scalarType == "" {
		return "unset"
	}
	return string(scalarType)
}
