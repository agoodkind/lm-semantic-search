package milvus_test

import (
	"testing"

	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"
)

func TestNewRowEmbeddingModelColumnWritesNullForEmptyModel(t *testing.T) {
	t.Parallel()

	modelColumn, err := milvusstore.NewRowEmbeddingModelColumn(
		"row_models",
		[]string{"model-a", ""},
	)
	if err != nil {
		t.Fatalf("The attempt to build the column failed with error %v.", err)
	}
	if modelColumn.Len() != 2 {
		t.Fatalf("The column has length %d instead of 2.", modelColumn.Len())
	}
	knownModelIsNull, err := modelColumn.IsNull(0)
	if err != nil {
		t.Fatalf("The attempt to read the null marker of row 0 failed with error %v.", err)
	}
	knownModel, err := modelColumn.GetAsString(0)
	if err != nil {
		t.Fatalf("The attempt to read the value of row 0 failed with error %v.", err)
	}
	if knownModelIsNull || knownModel != "model-a" {
		t.Fatalf("Row 0 does not store model-a because its null marker is %t and its value is %q.", knownModelIsNull, knownModel)
	}
	emptyModelIsNull, err := modelColumn.IsNull(1)
	if err != nil {
		t.Fatalf("The attempt to read the null marker of row 1 failed with error %v.", err)
	}
	if !emptyModelIsNull {
		t.Fatal("Row 1 is not null despite its empty model.")
	}
}
