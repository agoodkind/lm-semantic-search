package semantic

import (
	"errors"
	"slices"
	"testing"
)

func TestCopyChunkMutationsInsertDestinationBeforeDeletingSource(t *testing.T) {
	t.Parallel()

	operations := make([]string, 0, 4)
	err := runCopyChunkMutations(copyChunkMutations{
		insertDestination: func() error {
			operations = append(operations, "insert destination")
			return nil
		},
		persistDestination: func() error {
			operations = append(operations, "persist destination")
			return nil
		},
		deleteSource: func() error {
			operations = append(operations, "delete source")
			return nil
		},
		persistSourceDelete: func() error {
			operations = append(operations, "persist source delete")
			return nil
		},
	})
	if err != nil {
		t.Fatalf("runCopyChunkMutations returned error: %v", err)
	}
	want := []string{
		"insert destination",
		"persist destination",
		"delete source",
		"persist source delete",
	}
	if !slices.Equal(operations, want) {
		t.Fatalf("mutation order = %v, want %v", operations, want)
	}
}

func TestCopyChunkMutationsDeleteFailureKeepsBothCopies(t *testing.T) {
	t.Parallel()

	deleteErr := errors.New("delete source failed")
	sourceExists := true
	destinationExists := false
	err := runCopyChunkMutations(copyChunkMutations{
		insertDestination: func() error {
			destinationExists = true
			return nil
		},
		persistDestination: func() error {
			return nil
		},
		deleteSource: func() error {
			return deleteErr
		},
		persistSourceDelete: func() error {
			return nil
		},
	})
	if !errors.Is(err, deleteErr) {
		t.Fatalf("runCopyChunkMutations error = %v, want %v", err, deleteErr)
	}
	if !sourceExists || !destinationExists {
		t.Fatalf(
			"source exists = %t, destination exists = %t, want both true",
			sourceExists,
			destinationExists,
		)
	}
}

func TestCopyChunkMutationsInterruptionAfterFirstMutationPreservesContent(t *testing.T) {
	t.Parallel()

	interruptedErr := errors.New("copy interrupted")
	sourceExists := true
	destinationExists := false
	err := runCopyChunkMutations(copyChunkMutations{
		insertDestination: func() error {
			destinationExists = true
			return nil
		},
		persistDestination: func() error {
			return interruptedErr
		},
		deleteSource: func() error {
			sourceExists = false
			return nil
		},
		persistSourceDelete: func() error {
			return nil
		},
	})
	if !errors.Is(err, interruptedErr) {
		t.Fatalf("runCopyChunkMutations error = %v, want %v", err, interruptedErr)
	}
	if !sourceExists && !destinationExists {
		t.Fatal("copy interruption removed the only stored content")
	}
	if !sourceExists || !destinationExists {
		t.Fatalf(
			"source exists = %t, destination exists = %t, want both true",
			sourceExists,
			destinationExists,
		)
	}
}
