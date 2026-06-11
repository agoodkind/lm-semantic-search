package view

import "testing"

// The view package must stay pure data: this test locks the absence of an
// internal/model dependency at the package level. The render wall depends on
// view being importable without model.
func TestViewHasNoModelDependency(t *testing.T) {
	// Compile-time property: if any view type referenced internal/model this
	// package would import it and the import-list assertion in the render
	// package tests would fail. This test exists to document the invariant.
	_ = ProgressSurface{}
	_ = JobEntryView{}
	_ = GetIndexView{}
}
