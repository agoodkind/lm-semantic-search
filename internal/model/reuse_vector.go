package model

// ReuseVector is one reuse map value containing a stored dense vector and a flag indicating whether the stored source row had an embedding model.
type ReuseVector struct {
	Vector []float32
	// SourceModelKnown is true when the stored source row had a non-empty embedding model.
	// With the zero value, the insert writes no embedding model on the reused row.
	SourceModelKnown bool
}
