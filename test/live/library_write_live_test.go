//go:build live

package live

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"os"
	"os/exec"
	"slices"
	"syscall"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/library"
)

// TestMain runs one library child operation instead of the tests when the
// parent test sets libraryLiveChildEnv.
func TestMain(m *testing.M) {
	if encoded := os.Getenv(libraryLiveChildEnv); encoded != "" {
		os.Exit(runLibraryChild(encoded))
	}
	os.Exit(m.Run())
}

// conversationSpec is an AppendOnly namespace with one mutable column.
func conversationSpec() library.NamespaceSpec {
	return library.NamespaceSpec{
		ID:     "conversations",
		Policy: library.AppendOnly,
		Scalars: []library.ScalarColumn{
			{Name: "workspace", Type: library.String, Nullable: true, Mutable: true, MaxLength: 256},
			{Name: "archived", Type: library.Bool, Mutable: true},
			{Name: "message_index", Type: library.Int64},
		},
	}
}

// codeSpec is a ReplaceAllowed namespace for file owners.
func codeSpec() library.NamespaceSpec {
	return library.NamespaceSpec{
		ID:     "code",
		Policy: library.ReplaceAllowed,
		Scalars: []library.ScalarColumn{
			{Name: "path", Type: library.String, MaxLength: 1024},
			{Name: "part", Type: library.Int64},
		},
	}
}

func messageRow(key string, text string, index int64) library.Occurrence {
	return library.Occurrence{
		RowKey:         key,
		SortKey:        fmt.Sprintf("%08d", index),
		SourceText:     text,
		SearchText:     text,
		EmbeddingInput: text,
		Scalars: map[string]library.ScalarValue{
			"workspace":     {Type: library.String, String: "/workspace/alpha"},
			"archived":      {Type: library.Bool, Bool: false},
			"message_index": {Type: library.Int64, Int64: index},
		},
	}
}

func codeRow(path string, part int64, text string) library.Occurrence {
	return library.Occurrence{
		RowKey:         fmt.Sprintf("%s#%d", path, part),
		SortKey:        fmt.Sprintf("%s#%04d", path, part),
		SourceText:     text,
		SearchText:     text,
		EmbeddingInput: "passage: " + text,
		Scalars: map[string]library.ScalarValue{
			"path": {Type: library.String, String: path},
			"part": {Type: library.Int64, Int64: part},
		},
	}
}

func appendBatch(owner string, order uint64, token string, rows ...library.Occurrence) library.Batch {
	return library.Batch{Namespace: "conversations", OwnerID: owner, GenerationOrder: order, IdempotencyToken: token, Mode: library.Append, Rows: rows}
}

func mustApply(t *testing.T, ctx context.Context, opened *library.Library, batch library.Batch) library.ApplyReceipt {
	t.Helper()
	receipt, err := opened.Apply(ctx, batch)
	if err != nil {
		t.Fatalf("apply owner %s order %d: %v", batch.OwnerID, batch.GenerationOrder, err)
	}
	return receipt
}

func requireError(t *testing.T, err error, want error) {
	t.Helper()
	if !errors.Is(err, want) {
		t.Fatalf("error = %v, want %v", err, want)
	}
}

// assertCatalogConsistent requires every occurrence to reference a verified
// vector that exists in the backend, and an empty outbox.
func assertCatalogConsistent(t *testing.T, harness *libraryHarness, descriptor library.StoreDescriptor, collection string) catalogRows {
	t.Helper()
	rows := harness.readCatalog(descriptor)
	if rows.unverifiedLinked != 0 {
		t.Fatalf("%d occurrences reference an unverified vector", rows.unverifiedLinked)
	}
	if rows.outboxEntries != 0 {
		t.Fatalf("outbox has %d entries after a finished write", rows.outboxEntries)
	}
	backend := harness.backendVectorIDs(collection)
	for key, vectorID := range rows.occurrences {
		if _, found := slices.BinarySearch(backend, vectorID); !found {
			t.Fatalf("occurrence %s references vector %s, which the backend does not return", key, vectorID)
		}
	}
	return rows
}

func TestLibraryWriteSharesOneVectorAcrossOccurrences(t *testing.T) {
	harness := newLibraryHarness(t)
	t.Logf("embedding window start %s", time.Now().UTC().Format(time.RFC3339))
	defer func() { t.Logf("embedding window end %s", time.Now().UTC().Format(time.RFC3339)) }()
	descriptor := harness.descriptor("shared")
	opened := harness.open(descriptor, harness.vectorStore("shared_pool"))
	for _, spec := range []library.NamespaceSpec{conversationSpec(), codeSpec()} {
		if err := opened.RegisterNamespace(harness.context(), spec); err != nil {
			t.Fatalf("register %s: %v", spec.ID, err)
		}
	}

	shared := "passage: func parseConfig(path string) (Config, error)"
	first := messageRow("m1", shared, 1)
	second := messageRow("m2", shared, 2)
	second.SourceText = "a different displayed excerpt"
	second.Scalars["workspace"] = library.ScalarValue{Type: library.String, String: "/workspace/beta"}
	mustApply(t, harness.context(), opened, appendBatch("conversation-a", 1, "token-1", first, second))
	code := codeRow("config.go", 0, "func parseConfig(path string) (Config, error)")
	mustApply(t, harness.context(), opened, library.Batch{Namespace: "code", OwnerID: "config.go", GenerationOrder: 1, IdempotencyToken: "code-1", Mode: library.Replace, Rows: []library.Occurrence{code}})

	rows := assertCatalogConsistent(t, harness, descriptor, "shared_pool")
	if len(rows.occurrences) != 3 {
		t.Fatalf("catalog has %d occurrences, want 3", len(rows.occurrences))
	}
	if backend := harness.backendVectorIDs("shared_pool"); len(backend) != 1 {
		t.Fatalf("backend has %d vectors for one distinct embedding input, want 1: %v", len(backend), backend)
	}

	otherModel := descriptor
	otherModel.CatalogPath = harness.descriptor("other-model").CatalogPath
	otherModel.LockPath = harness.descriptor("other-model").LockPath
	otherModel.EmbeddingRevision = "another-revision"
	otherOpened := harness.open(otherModel, harness.vectorStore("other_model_pool"))
	if err := otherOpened.RegisterNamespace(harness.context(), conversationSpec()); err != nil {
		t.Fatalf("register under the other revision: %v", err)
	}
	mustApply(t, harness.context(), otherOpened, appendBatch("conversation-a", 1, "token-1", first))
	otherRows := harness.readCatalog(otherModel)
	if otherRows.occurrences["conversations/conversation-a/m1"] == rows.occurrences["conversations/conversation-a/m1"] {
		t.Fatal("the same input under another model revision reused the first revision's vector identity")
	}
}

func TestLibraryWriteAppendReplayAndConflicts(t *testing.T) {
	harness := newLibraryHarness(t)
	t.Logf("embedding window start %s", time.Now().UTC().Format(time.RFC3339))
	defer func() { t.Logf("embedding window end %s", time.Now().UTC().Format(time.RFC3339)) }()
	descriptor := harness.descriptor("append")
	opened := harness.open(descriptor, harness.vectorStore("append_pool"))
	if err := opened.RegisterNamespace(harness.context(), conversationSpec()); err != nil {
		t.Fatalf("register: %v", err)
	}

	firstBatch := appendBatch("conversation-a", 1, "token-1", messageRow("m1", "open the config file", 1))
	firstReceipt := mustApply(t, harness.context(), opened, firstBatch)
	if replay := mustApply(t, harness.context(), opened, firstBatch); replay != firstReceipt {
		t.Fatalf("replay receipt %+v differs from %+v", replay, firstReceipt)
	}

	secondBatch := appendBatch("conversation-a", 2, "token-2", messageRow("m1", "open the config file", 1), messageRow("m2", "read the parser tests", 2))
	secondReceipt := mustApply(t, harness.context(), opened, secondBatch)
	if replay := mustApply(t, harness.context(), opened, firstBatch); replay != firstReceipt {
		t.Fatalf("replay of an older token after a later generation returned %+v, want %+v", replay, firstReceipt)
	}
	state, err := opened.GetOwnerState(harness.context(), "conversations", "conversation-a")
	if err != nil {
		t.Fatalf("owner state: %v", err)
	}
	if state.GenerationOrder != 2 || state.IdempotencyToken != "token-2" || state.Fingerprint != secondReceipt.Fingerprint {
		t.Fatalf("owner state = %+v, want order 2, token-2, fingerprint %s", state, secondReceipt.Fingerprint)
	}

	_, err = opened.Apply(harness.context(), appendBatch("conversation-a", 2, "other-token", messageRow("m3", "new text", 3)))
	requireError(t, err, library.ErrAppendConflict)
	_, err = opened.Apply(harness.context(), appendBatch("conversation-a", 1, "unknown-token", messageRow("m3", "new text", 3)))
	requireError(t, err, library.ErrStaleGeneration)
	_, err = opened.Apply(harness.context(), appendBatch("conversation-a", 3, "token-3", messageRow("m1", "rewritten text", 1)))
	requireError(t, err, library.ErrAppendConflict)

	// A committed token replayed with other content conflicts through every
	// entry point: Apply, Stage, and CommitGeneration with a different seal.
	_, err = opened.Apply(harness.context(), appendBatch("conversation-a", 1, "token-1", messageRow("m1", "changed after commit", 1)))
	requireError(t, err, library.ErrAppendConflict)
	_, err = opened.Apply(harness.context(), appendBatch("conversation-a", 1, "token-1", messageRow("m1", "open the config file", 1), messageRow("m9", "added after commit", 9)))
	requireError(t, err, library.ErrAppendConflict)
	firstKey := library.GenerationKey{Namespace: "conversations", OwnerID: "conversation-a", GenerationOrder: 1, IdempotencyToken: "token-1"}
	requireError(t, opened.Stage(harness.context(), library.StageBatch{Key: firstKey, Mode: library.Append, Rows: []library.Occurrence{messageRow("m1", "changed after commit", 1)}}), library.ErrAppendConflict)
	otherSeal, err := library.SealRows([]library.Occurrence{messageRow("m1", "changed after commit", 1)})
	if err != nil {
		t.Fatalf("seal changed rows: %v", err)
	}
	_, err = opened.CommitGeneration(harness.context(), firstKey, otherSeal)
	requireError(t, err, library.ErrAppendConflict)
	if replay := mustApply(t, harness.context(), opened, firstBatch); replay != firstReceipt {
		t.Fatalf("identical replay after rejected rewrites returned %+v, want %+v", replay, firstReceipt)
	}

	replace := appendBatch("conversation-a", 4, "token-4", messageRow("m4", "replacement", 4))
	replace.Mode = library.Replace
	_, err = opened.Apply(harness.context(), replace)
	requireError(t, err, library.ErrInvalidRequest)
	requireError(t, opened.Delete(harness.context(), []library.OccurrenceID{{Namespace: "conversations", OwnerID: "conversation-a", RowKey: "m1"}}), library.ErrInvalidRequest)

	rows := assertCatalogConsistent(t, harness, descriptor, "append_pool")
	if len(rows.occurrences) != 2 {
		t.Fatalf("catalog has %d occurrences after rejected writes, want 2", len(rows.occurrences))
	}
}

func TestLibraryWriteReplaceStagesAndSealsWholeGenerations(t *testing.T) {
	harness := newLibraryHarness(t)
	t.Logf("embedding window start %s", time.Now().UTC().Format(time.RFC3339))
	defer func() { t.Logf("embedding window end %s", time.Now().UTC().Format(time.RFC3339)) }()
	descriptor := harness.descriptor("replace")
	opened := harness.open(descriptor, harness.vectorStore("replace_pool"))
	if err := opened.RegisterNamespace(harness.context(), codeSpec()); err != nil {
		t.Fatalf("register: %v", err)
	}
	firstRows := []library.Occurrence{
		codeRow("main.go", 0, "package main"),
		codeRow("main.go", 1, "func main() { run() }"),
		codeRow("main.go", 2, "func run() { serve() }"),
	}
	stageAndCommit(t, harness.context(), opened, library.GenerationKey{Namespace: "code", OwnerID: "main.go", GenerationOrder: 1, IdempotencyToken: "g1"}, firstRows)

	secondRows := []library.Occurrence{codeRow("main.go", 0, "package main"), codeRow("main.go", 1, "func main() { serve() }")}
	secondKey := library.GenerationKey{Namespace: "code", OwnerID: "main.go", GenerationOrder: 2, IdempotencyToken: "g2"}
	if err := opened.Stage(harness.context(), library.StageBatch{Key: secondKey, Mode: library.Replace, Rows: secondRows[:1]}); err != nil {
		t.Fatalf("stage first part: %v", err)
	}
	fullSeal, err := library.SealRows(secondRows)
	if err != nil {
		t.Fatalf("seal: %v", err)
	}
	_, err = opened.CommitGeneration(harness.context(), secondKey, fullSeal)
	requireError(t, err, library.ErrInvalidRequest)
	if rows := harness.readCatalog(descriptor); len(rows.occurrences) != 3 {
		t.Fatalf("a truncated staged generation replaced the owner: %d occurrences, want 3", len(rows.occurrences))
	}
	if err := opened.Stage(harness.context(), library.StageBatch{Key: secondKey, Mode: library.Replace, Rows: secondRows[1:]}); err != nil {
		t.Fatalf("stage second part: %v", err)
	}
	if _, err := opened.CommitGeneration(harness.context(), secondKey, fullSeal); err != nil {
		t.Fatalf("commit complete generation: %v", err)
	}
	rows := assertCatalogConsistent(t, harness, descriptor, "replace_pool")
	if len(rows.occurrences) != 2 {
		t.Fatalf("owner has %d occurrences after shrinking to 2 parts", len(rows.occurrences))
	}
	if _, found := rows.occurrences["code/main.go/main.go#2"]; found {
		t.Fatal("the removed third part is still published")
	}
	if backend := harness.backendVectorIDs("replace_pool"); len(backend) != 4 {
		t.Fatalf("backend has %d vectors, want 4 distinct inputs retained", len(backend))
	}

	abortKey := library.GenerationKey{Namespace: "code", OwnerID: "main.go", GenerationOrder: 3, IdempotencyToken: "g3"}
	if err := opened.Stage(harness.context(), library.StageBatch{Key: abortKey, Mode: library.Replace, Rows: []library.Occurrence{codeRow("main.go", 0, "package main")}}); err != nil {
		t.Fatalf("stage aborted generation: %v", err)
	}
	if err := opened.AbortGeneration(harness.context(), abortKey); err != nil {
		t.Fatalf("abort: %v", err)
	}
	_, err = opened.CommitGeneration(harness.context(), abortKey, fullSeal)
	requireError(t, err, library.ErrInvalidRequest)

	if err := opened.Delete(harness.context(), []library.OccurrenceID{{Namespace: "code", OwnerID: "main.go", RowKey: "main.go#1"}}); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if rows := harness.readCatalog(descriptor); len(rows.occurrences) != 1 {
		t.Fatalf("owner has %d occurrences after one delete, want 1", len(rows.occurrences))
	}
}

func stageAndCommit(t *testing.T, ctx context.Context, opened *library.Library, key library.GenerationKey, rows []library.Occurrence) library.ApplyReceipt {
	t.Helper()
	for _, row := range rows {
		if err := opened.Stage(ctx, library.StageBatch{Key: key, Mode: library.Replace, Rows: []library.Occurrence{row}}); err != nil {
			t.Fatalf("stage %s: %v", row.RowKey, err)
		}
	}
	seal, err := library.SealRows(rows)
	if err != nil {
		t.Fatalf("seal: %v", err)
	}
	receipt, err := opened.CommitGeneration(ctx, key, seal)
	if err != nil {
		t.Fatalf("commit: %v", err)
	}
	return receipt
}

func TestLibraryWriteReprojectsScalarsWithoutVectorWrites(t *testing.T) {
	harness := newLibraryHarness(t)
	t.Logf("embedding window start %s", time.Now().UTC().Format(time.RFC3339))
	defer func() { t.Logf("embedding window end %s", time.Now().UTC().Format(time.RFC3339)) }()
	descriptor := harness.descriptor("reproject")
	opened := harness.open(descriptor, harness.vectorStore("reproject_pool"))
	if err := opened.RegisterNamespace(harness.context(), conversationSpec()); err != nil {
		t.Fatalf("register: %v", err)
	}
	mustApply(t, harness.context(), opened, appendBatch("conversation-a", 1, "token-1", messageRow("m1", "archive this conversation later", 1)))
	before := harness.backendVectorIDs("reproject_pool")

	projection := library.ScalarProjection{
		Namespace:        "conversations",
		OwnerID:          "conversation-a",
		ProjectionOrder:  1,
		IdempotencyToken: "projection-1",
		Rows: map[string]map[string]library.ScalarValue{
			"m1": {"archived": {Type: library.Bool, Bool: true}, "workspace": {Type: library.String, Null: true}},
		},
	}
	receipt, err := opened.ReprojectScalars(harness.context(), projection)
	if err != nil {
		t.Fatalf("reproject: %v", err)
	}
	replay, err := opened.ReprojectScalars(harness.context(), projection)
	if err != nil || replay != receipt {
		t.Fatalf("replayed projection = %+v, %v, want %+v", replay, err, receipt)
	}
	if after := harness.backendVectorIDs("reproject_pool"); !slices.Equal(before, after) {
		t.Fatalf("reprojection changed backend vectors from %v to %v", before, after)
	}
	effective := harness.readScalars(descriptor, "effective_scalars", "m1")
	wantEffective := map[string]string{"archived": "bool=1", "workspace": "null"}
	if !maps.Equal(effective, wantEffective) {
		t.Fatalf("effective scalars of m1 = %v, want %v", effective, wantEffective)
	}
	original := harness.readScalars(descriptor, "occurrence_scalars", "m1")
	wantOriginal := map[string]string{"archived": "bool=0", "workspace": "string=/workspace/alpha", "message_index": "int64=1"}
	if !maps.Equal(original, wantOriginal) {
		t.Fatalf("occurrence scalars of m1 = %v, want %v", original, wantOriginal)
	}
	immutable := projection
	immutable.ProjectionOrder = 2
	immutable.IdempotencyToken = "projection-2"
	immutable.Rows = map[string]map[string]library.ScalarValue{"m1": {"message_index": {Type: library.Int64, Int64: 9}}}
	_, err = opened.ReprojectScalars(harness.context(), immutable)
	requireError(t, err, library.ErrInvalidRequest)
	conflicting := projection
	conflicting.IdempotencyToken = "projection-other"
	_, err = opened.ReprojectScalars(harness.context(), conflicting)
	requireError(t, err, library.ErrAppendConflict)
}

func TestLibraryWriteBindingRaceBetweenProcesses(t *testing.T) {
	harness := newLibraryHarness(t)
	startAt := time.Now().Add(3 * time.Second)
	children := make([]*exec.Cmd, 0, 2)
	descriptors := []library.StoreDescriptor{harness.descriptor("race-a"), harness.descriptor("race-b")}
	for _, descriptor := range descriptors {
		children = append(children, startLibraryChild(t, libraryChildRequest{
			Action:      childActionOpen,
			Environment: harness.environment,
			Database:    harness.database,
			Collection:  "race_pool",
			Descriptor:  descriptor,
			StartAt:     startAt,
		}))
	}
	statuses := make([]int, 0, len(children))
	for _, child := range children {
		status, signal := waitLibraryChild(t, child)
		if signal != 0 {
			t.Fatalf("child ended by signal %v", signal)
		}
		statuses = append(statuses, status)
	}
	slices.Sort(statuses)
	if !slices.Equal(statuses, []int{0, childExitStoreMismatch}) {
		t.Fatalf("child exit statuses = %v, want one success and one store mismatch", statuses)
	}
	winner, loser := descriptors[0], descriptors[1]
	probe, probeErr := library.Open(harness.context(), library.Config{Store: winner, Vectors: harness.vectorStore("race_pool"), Embedder: harness.embedder})
	switch {
	case errors.Is(probeErr, library.ErrStoreMismatch):
		winner, loser = loser, winner
	case probeErr != nil:
		t.Fatalf("open the first racing catalog: %v", probeErr)
	default:
		if err := probe.Close(); err != nil {
			t.Fatalf("close probe: %v", err)
		}
	}
	opened := harness.open(winner, harness.vectorStore("race_pool"))
	if err := opened.RegisterNamespace(harness.context(), conversationSpec()); err != nil {
		t.Fatalf("register on the winning catalog: %v", err)
	}
	_, err := library.Open(harness.context(), library.Config{Store: loser, Vectors: harness.vectorStore("race_pool"), Embedder: harness.embedder})
	requireError(t, err, library.ErrStoreMismatch)
	if backend := harness.backendVectorIDs("race_pool"); len(backend) != 0 {
		t.Fatalf("race wrote %d vectors, want 0", len(backend))
	}
}

func TestLibraryWriteConcurrentWriterProcesses(t *testing.T) {
	harness := newLibraryHarness(t)
	t.Logf("embedding window start %s", time.Now().UTC().Format(time.RFC3339))
	defer func() { t.Logf("embedding window end %s", time.Now().UTC().Format(time.RFC3339)) }()
	descriptor := harness.descriptor("concurrent")
	startAt := time.Now().Add(3 * time.Second)
	children := make([]*exec.Cmd, 0, 3)
	for writer := range 3 {
		batches := make([]library.Batch, 0, 3)
		for owner := range 3 {
			ownerID := fmt.Sprintf("writer-%d-owner-%d", writer, owner)
			batches = append(batches, appendBatch(ownerID, 1, "token-"+ownerID,
				messageRow("shared", "shared content repeated by every writer", 1),
				messageRow("own", "content unique to "+ownerID, 2),
			))
		}
		children = append(children, startLibraryChild(t, libraryChildRequest{
			Action:      childActionApply,
			Environment: harness.environment,
			Database:    harness.database,
			Collection:  "concurrent_pool",
			Descriptor:  descriptor,
			Namespace:   conversationSpec(),
			Batches:     batches,
			StartAt:     startAt,
		}))
	}
	for _, child := range children {
		if status, signal := waitLibraryChild(t, child); status != 0 || signal != 0 {
			t.Fatalf("writer process exited with status %d signal %v", status, signal)
		}
	}
	opened := harness.open(descriptor, harness.vectorStore("concurrent_pool"))
	rows := assertCatalogConsistent(t, harness, descriptor, "concurrent_pool")
	if len(rows.occurrences) != 18 {
		t.Fatalf("catalog has %d occurrences, want 18", len(rows.occurrences))
	}
	if backend := harness.backendVectorIDs("concurrent_pool"); len(backend) != 10 {
		t.Fatalf("backend has %d vectors, want 10 distinct inputs", len(backend))
	}
	for writer := range 3 {
		for owner := range 3 {
			ownerID := fmt.Sprintf("writer-%d-owner-%d", writer, owner)
			state, err := opened.GetOwnerState(harness.context(), "conversations", ownerID)
			if err != nil || state.GenerationOrder != 1 {
				t.Fatalf("owner %s state = %+v, %v, want order 1", ownerID, state, err)
			}
		}
	}
}

func TestLibraryWriteRecoversAfterProcessDeath(t *testing.T) {
	harness := newLibraryHarness(t)
	t.Logf("embedding window start %s", time.Now().UTC().Format(time.RFC3339))
	defer func() { t.Logf("embedding window end %s", time.Now().UTC().Format(time.RFC3339)) }()
	for _, crashPoint := range []string{crashBeforePut, crashAfterPut, crashAfterVerify, crashAfterPublish} {
		t.Run(crashPoint, func(t *testing.T) {
			descriptor := harness.descriptor("crash-" + crashPoint)
			collection := "crash_" + crashPoint
			batch := appendBatch("conversation-crash", 1, "token-1",
				messageRow("m1", "crash recovery input for "+crashPoint, 1),
				messageRow("m2", "second crash recovery input for "+crashPoint, 2),
			)
			child := startLibraryChild(t, libraryChildRequest{
				Action:      childActionApply,
				Environment: harness.environment,
				Database:    harness.database,
				Collection:  collection,
				Descriptor:  descriptor,
				Namespace:   conversationSpec(),
				Batches:     []library.Batch{batch},
				CrashPoint:  crashPoint,
			})
			if _, signal := waitLibraryChild(t, child); signal != syscall.SIGKILL {
				t.Fatalf("child at %s ended without SIGKILL", crashPoint)
			}
			crashed := harness.readCatalog(descriptor)
			if crashed.unverifiedLinked != 0 {
				t.Fatalf("after the crash %d occurrences reference an unverified vector", crashed.unverifiedLinked)
			}

			opened := harness.open(descriptor, harness.vectorStore(collection))
			if err := opened.RegisterNamespace(harness.context(), conversationSpec()); err != nil {
				t.Fatalf("register after restart: %v", err)
			}
			receipt := mustApply(t, harness.context(), opened, batch)
			if receipt.GenerationOrder != 1 {
				t.Fatalf("receipt after restart = %+v", receipt)
			}
			rows := assertCatalogConsistent(t, harness, descriptor, collection)
			if len(rows.occurrences) != 2 {
				t.Fatalf("catalog has %d occurrences after recovery, want 2", len(rows.occurrences))
			}
			if backend := harness.backendVectorIDs(collection); len(backend) != 2 {
				t.Fatalf("backend has %d live vectors after recovery, want 2", len(backend))
			}
		})
	}
}

// ambiguousVectorStore writes each vector and then reports a timeout, the
// result of an RPC that succeeded after the caller stopped waiting.
type ambiguousVectorStore struct {
	library.VectorStore
}

func (store ambiguousVectorStore) PutCanonical(ctx context.Context, record library.VectorRecord) error {
	if err := store.VectorStore.PutCanonical(ctx, record); err != nil {
		return err
	}
	return context.DeadlineExceeded
}

func TestLibraryWriteReplaysAmbiguousUpsert(t *testing.T) {
	harness := newLibraryHarness(t)
	t.Logf("embedding window start %s", time.Now().UTC().Format(time.RFC3339))
	defer func() { t.Logf("embedding window end %s", time.Now().UTC().Format(time.RFC3339)) }()
	descriptor := harness.descriptor("ambiguous")
	batch := appendBatch("conversation-ambiguous", 1, "token-1", messageRow("m1", "ambiguous upsert input", 1))

	failing, err := library.Open(harness.context(), library.Config{Store: descriptor, Vectors: ambiguousVectorStore{VectorStore: harness.vectorStore("ambiguous_pool")}, Embedder: harness.embedder})
	if err != nil {
		t.Fatalf("open with the ambiguous adapter: %v", err)
	}
	if err := failing.RegisterNamespace(harness.context(), conversationSpec()); err != nil {
		t.Fatalf("register: %v", err)
	}
	_, err = failing.Apply(harness.context(), batch)
	requireError(t, err, context.DeadlineExceeded)
	if err := failing.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	pending := harness.readCatalog(descriptor)
	if pending.outboxEntries != 1 || len(pending.occurrences) != 0 {
		t.Fatalf("after the ambiguous upsert: outbox %d, occurrences %d, want 1 and 0", pending.outboxEntries, len(pending.occurrences))
	}

	opened := harness.open(descriptor, harness.vectorStore("ambiguous_pool"))
	mustApply(t, harness.context(), opened, batch)
	rows := assertCatalogConsistent(t, harness, descriptor, "ambiguous_pool")
	if len(rows.occurrences) != 1 {
		t.Fatalf("catalog has %d occurrences, want 1", len(rows.occurrences))
	}
	if backend := harness.backendVectorIDs("ambiguous_pool"); len(backend) != 1 {
		t.Fatalf("backend has %d live vectors, want 1", len(backend))
	}
}

func TestLibraryWriteRejectsAnotherPoolOrLockPath(t *testing.T) {
	harness := newLibraryHarness(t)
	descriptor := harness.descriptor("identity")
	first, err := library.Open(harness.context(), library.Config{Store: descriptor, Vectors: harness.vectorStore("identity_pool"), Embedder: harness.embedder})
	if err != nil {
		t.Fatalf("open the catalog: %v", err)
	}
	if err := first.Close(); err != nil {
		t.Fatalf("close the catalog: %v", err)
	}

	_, err = library.Open(harness.context(), library.Config{Store: descriptor, Vectors: harness.vectorStore("other_pool"), Embedder: harness.embedder})
	requireError(t, err, library.ErrStoreMismatch)
	collections, err := harness.milvus.ListCollections(harness.context(), milvusclient.NewListCollectionOption())
	if err != nil {
		t.Fatalf("list collections: %v", err)
	}
	if slices.Contains(collections, "other_pool") {
		t.Fatal("a rejected open created the other pool collection")
	}

	movedLock := descriptor
	movedLock.LockPath = descriptor.LockPath + ".moved"
	_, err = library.Open(harness.context(), library.Config{Store: movedLock, Vectors: harness.vectorStore("identity_pool"), Embedder: harness.embedder})
	requireError(t, err, library.ErrStoreMismatch)

	reopened := harness.open(descriptor, harness.vectorStore("identity_pool"))
	if err := reopened.RegisterNamespace(harness.context(), conversationSpec()); err != nil {
		t.Fatalf("register after the rejected opens: %v", err)
	}
}
