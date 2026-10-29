package milvus

import (
	"context"

	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
)

// CopyPhase identifies one pass of a shared copy or a copy verification.
type CopyPhase string

const (
	// CopyPhaseOccurrences writes the occurrence rows of the copied source rows.
	CopyPhaseOccurrences CopyPhase = "copy_occurrences"
	// CopyPhaseVectors reads every source vector and writes each distinct vector once.
	CopyPhaseVectors CopyPhase = "copy_vectors"
	// VerifyPhaseRows compares every source row with its target row and vector.
	VerifyPhaseRows CopyPhase = "verify_rows"
	// VerifyPhaseExtraRows checks every target row ID against the source.
	VerifyPhaseExtraRows CopyPhase = "verify_extra_rows"
	// VerifyPhaseReferences reads the vector key of every target row.
	VerifyPhaseReferences CopyPhase = "verify_vector_references"
	// VerifyPhaseVectors checks every target vector key against those references.
	VerifyPhaseVectors CopyPhase = "verify_vectors"
)

// CopyProgress reports the rows one phase has processed.
// Done is the number of rows the phase has processed.
// Total is the number of rows the phase reads, counted when the phase starts.
type CopyProgress struct {
	Phase CopyPhase
	Done  int64
	Total int64
}

// ProgressFunc receives a [CopyProgress] after each page of a phase.
// A nil ProgressFunc does not receive progress reports.
type ProgressFunc func(CopyProgress)

func (progress ProgressFunc) report(phase CopyPhase, done int64, total int64) {
	if progress != nil {
		progress(CopyProgress{Phase: phase, Done: done, Total: total})
	}
}

// countRows returns the number of rows an expression matches with strong
// consistency.
func (store *SharedStore) countRows(ctx context.Context, collectionName string, expression string) (int64, error) {
	resultSet, err := store.inner.client.Query(ctx, milvusclient.NewQueryOption(collectionName).
		WithFilter(expression).
		WithOutputFields("count(*)").
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		return 0, WrapError(ctx, err, "count rows of "+collectionName)
	}
	count, err := requiredInt64(resultSet, "count(*)", 0)
	if err != nil {
		return 0, err
	}
	return count, nil
}
