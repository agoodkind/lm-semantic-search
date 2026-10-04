// Package milvus implements [collection.Store] on a Milvus collection. A
// collection stores the built-in columns below plus the scalar columns of its
// [collection.Declaration]. A hybrid collection also stores a BM25 sparse vector
// computed from the content column.
package milvus

// The TypeScript adapter created the stored collections with these camelCase
// column names. A renamed constant no longer matches the stored column.
const (
	// DenseVectorField is the dense embedding column.
	DenseVectorField = "vector"
	// SparseVectorField is the BM25 sparse vector column of a hybrid collection.
	SparseVectorField = "sparse_vector"
	// IDField is the primary key column.
	IDField = "id"
	// ContentField is the text column the BM25 function reads.
	ContentField = "content"
	// RelativePathField is the logical row key column.
	RelativePathField = "relativePath"
	// StartLineField is the first source line of a row.
	StartLineField = "startLine"
	// EndLineField is the last source line of a row.
	EndLineField = "endLine"
	// FileExtensionField is the dot-prefixed source file extension of a row.
	FileExtensionField = "fileExtension"
	// MetadataField is the metadata JSON column.
	MetadataField = "metadata"
	// SplitPartField is the nullable split position of an oversized chunk.
	SplitPartField = "splitPart"
	// ContentHashField is the nullable content hash column.
	ContentHashField = "contentHash"
	// EmbeddingModelField is the nullable embedding model column.
	EmbeddingModelField = "embeddingModel"
	// CountOutputField is the Milvus row count pseudo field.
	CountOutputField = "count(*)"
)

const (
	idFieldMaxLength            = 512
	contentFieldMaxLength       = 65_535
	relativePathFieldMaxLength  = 1024
	fileExtensionFieldMaxLength = 32
	metadataFieldMaxLength      = 65_535
	contentHashFieldMaxLength   = 64
	embeddingModelMaxLength     = 65_535
	boolBytes                   = 1
	int64Bytes                  = 8
)

// BuiltinColumnNames returns the columns the built-in collection schema
// defines. A collection declaration may not declare any of them as a scalar
// column, and a stored schema lists them outside its declared scalars.
func BuiltinColumnNames() []string {
	return []string{
		IDField,
		ContentField,
		RelativePathField,
		StartLineField,
		EndLineField,
		FileExtensionField,
		MetadataField,
		ContentHashField,
		EmbeddingModelField,
		SplitPartField,
		DenseVectorField,
		SparseVectorField,
	}
}
