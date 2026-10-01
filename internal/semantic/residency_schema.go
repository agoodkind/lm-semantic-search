package semantic

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

func collectionUsesLegacyResidency(ctx context.Context, client *milvusclient.Client, collectionName string) (bool, error) {
	collection, err := client.DescribeCollection(ctx, milvusclient.NewDescribeCollectionOption(collectionName))
	if err != nil {
		return false, wrapStoreError(ctx, err, "describe residency schema for "+collectionName)
	}
	if collection.Schema == nil {
		return false, fmt.Errorf("describe residency schema for %s: missing schema", collectionName)
	}
	var primaryName string
	var vectorPresent, contentPresent, contentHashPresent bool
	for _, field := range collection.Schema.Fields {
		if field.PrimaryKey && field.DataType == entity.FieldTypeVarChar {
			primaryName = field.Name
		}
		switch field.Name {
		case denseVectorFieldName:
			vectorPresent = field.DataType == entity.FieldTypeFloatVector
		case contentFieldName:
			contentPresent = field.DataType == entity.FieldTypeVarChar
		case contentHashFieldName:
			contentHashPresent = field.DataType == entity.FieldTypeVarChar
		}
	}
	// Canonical library pools own their load state and use a different primary
	// field. Legacy residency manages document rows and the reuse catalog.
	return vectorPresent && ((primaryName == idFieldName && contentPresent) ||
		(primaryName == reuseCatalogRowKeyFieldName && contentHashPresent)), nil
}
