package model

// ReuseVector stores a reused vector and whether its source model is known.
type ReuseVector struct {
	Vector []float32
	// SourceModelKnown is true when the stored source row had a non-empty embedding model.
	// With the zero value, the insert writes no embedding model on the reused row.
	SourceModelKnown bool
}
