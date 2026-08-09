package daemon

import (
	"context"
	"testing"
)

type closeTrackingSemantic struct {
	semanticIndex
	closed bool
}

func (semantic *closeTrackingSemantic) Close(context.Context) error {
	semantic.closed = true
	return nil
}

func TestManagerCloseClosesSemanticBackend(t *testing.T) {
	manager, _, _ := newTestManager(t)
	semantic := &closeTrackingSemantic{semanticIndex: manager.semantic}
	manager.semantic = semantic

	if err := manager.Close(context.Background()); err != nil {
		t.Fatalf("Manager.Close returned error: %v", err)
	}
	if !semantic.closed {
		t.Fatal("Manager.Close did not close the semantic backend")
	}
}
