//go:build live

package live

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/model"
)

// The three conversations every backfill scenario seeds. Each carries a tool call
// and thinking text so derived convtool/ and convthink/ rows exist alongside the
// base conv/ rows.
var seedConversationIDs = []string{"live-a", "live-b", "live-c"}

// TestScenario1FullBackfillThenReexamineIsBounded proves the core bounded-
// examination guarantee: after a full stamped backfill, an identical reexamine
// embeds nothing because every marker is current.
func TestScenario1FullBackfillThenReexamineIsBounded(t *testing.T) {
	h := newHarness(t)

	convs := seedConversations()
	backfill := h.upsert(convs, pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN, true)
	requireCompleted(t, backfill, "backfill")
	if backfill.Progress.ChunksEmbedded <= 0 {
		t.Fatalf("backfill ChunksEmbedded = %d, want > 0 (nothing embedded on first pass)\n%s", backfill.Progress.ChunksEmbedded, progressString(backfill))
	}

	second := h.upsert(convs, pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN, true)
	requireCompleted(t, second, "second reexamine")
	if second.Progress.ChunksEmbedded != 0 {
		t.Fatalf("second reexamine ChunksEmbedded = %d, want 0 (bounded examination)\n%s", second.Progress.ChunksEmbedded, progressString(second))
	}
	if second.Progress.FilesEmbedded != 0 {
		t.Fatalf("second reexamine FilesEmbedded = %d, want 0 (no conversation re-embedded)\n%s", second.Progress.FilesEmbedded, progressString(second))
	}
	if second.Progress.FilesModified != 0 {
		t.Fatalf("second reexamine FilesModified = %d, want 0 (no forced items, empty diff)\n%s", second.Progress.FilesModified, progressString(second))
	}
}

// TestScenario2AppendReexaminesOnlyThatConversation proves an appended message
// re-examines and re-embeds only its own conversation, leaving the rest skipped.
func TestScenario2AppendReexaminesOnlyThatConversation(t *testing.T) {
	h := newHarness(t)

	convs := seedConversations()
	backfill := h.upsert(convs, pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN, true)
	requireCompleted(t, backfill, "backfill")

	// Append one message to a single conversation. Its content and derived
	// fingerprint both change, so the merkle diff classifies exactly it as
	// modified.
	convs["live-b"] = appendMessage(convs["live-b"], "live-b")
	appended := h.upsert(convs, pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN, true)
	requireCompleted(t, appended, "appended reexamine")

	if appended.Progress.FilesModified != 1 {
		t.Fatalf("appended FilesModified = %d, want 1 (only the changed conversation)\n%s", appended.Progress.FilesModified, progressString(appended))
	}
	if appended.Progress.FilesEmbedded != 1 {
		t.Fatalf("appended FilesEmbedded = %d, want 1 (only the changed conversation re-embedded)\n%s", appended.Progress.FilesEmbedded, progressString(appended))
	}
	if appended.Progress.ChunksEmbedded <= 0 {
		t.Fatalf("appended ChunksEmbedded = %d, want > 0 (the appended message embedded)\n%s", appended.Progress.ChunksEmbedded, progressString(appended))
	}
}

// TestScenario3VersionBumpDoesNotReforceBackfill proves the presence-based
// backfill contract: after a full backfill, bumping the derived pipeline version
// and running another plain BACKFILL does NOT re-force or re-embed conversations
// whose derived rows are already present. A backfill fills MISSING rows and
// no-ops present ones; it is presence-based and intentionally does not detect
// stale or older-version content. Auto-version-forcing (the old per-conversation
// derivedPipelineVersion marker behavior) is retired and replaced by an
// operator-initiated force. Force-based rebuild of present-but-stale rows is
// validated separately when the force path (R6) lands.
func TestScenario3VersionBumpDoesNotReforceBackfill(t *testing.T) {
	h := newHarness(t)

	convs := seedConversations()
	backfill := h.upsert(convs, pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN, true)
	requireCompleted(t, backfill, "backfill")

	// Bump the derived pipeline version to simulate a chunking change. A plain
	// backfill is presence-based, so the bump must not pull the already-present
	// conversations back into the changed set.
	restore := daemon.SetDerivedPipelineVersionForLiveTest("2")
	defer restore()

	bumped := h.upsert(convs, pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN, true)
	requireCompleted(t, bumped, "version-bumped backfill")

	// Every derived row is present, so the presence classifier prunes all three
	// conversations. The version bump changes nothing a backfill acts on.
	if bumped.Progress.FilesModified != 0 {
		t.Fatalf("version-bumped FilesModified = %d, want 0 (backfill is presence-based; a version bump does not re-force present conversations)\n%s", bumped.Progress.FilesModified, progressString(bumped))
	}
	if bumped.Progress.FilesEmbedded != 0 {
		t.Fatalf("version-bumped FilesEmbedded = %d, want 0 (no conversation re-embedded)\n%s", bumped.Progress.FilesEmbedded, progressString(bumped))
	}
	if bumped.Progress.ChunksEmbedded != 0 {
		t.Fatalf("version-bumped ChunksEmbedded = %d, want 0 (present rows are not rebuilt by a backfill)\n%s", bumped.Progress.ChunksEmbedded, progressString(bumped))
	}
}

// TestScenario4AuthoritativeDeletePurgesRows proves an AUTHORITATIVE upsert whose
// manifest omits a conversation drops that conversation's base, tool, and
// thinking rows from the live collection while the others survive.
func TestScenario4AuthoritativeDeletePurgesRows(t *testing.T) {
	h := newHarness(t)

	convs := seedConversations()
	backfill := h.upsert(convs, pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN, true)
	requireCompleted(t, backfill, "backfill")

	dropped := "live-c"
	kept := "live-a"
	// Assert all three row families exist before the delete so the post-delete
	// == 0 checks below cannot pass vacuously against rows that were never
	// produced. The seeded conversation carries a tool call and thinking text, so
	// its base, tool, and thinking prefixes must all be present here.
	for _, prefix := range []string{convBasePrefix(dropped), convToolPrefix(dropped), convThinkPrefix(dropped)} {
		if h.countRowsWithPrefix(prefix) <= 0 {
			t.Fatalf("expected %q rows present before delete\n%s", prefix, progressString(backfill))
		}
	}

	// Deliver every conversation except the dropped one under AUTHORITATIVE, so the
	// manifest omits it and the engine reconciles it away.
	remaining := map[string][]*pb.ConversationDocument{}
	for _, id := range seedConversationIDs {
		if id == dropped {
			continue
		}
		remaining[id] = convs[id]
	}
	authoritative := h.upsert(remaining, pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_AUTHORITATIVE, false)
	requireCompleted(t, authoritative, "authoritative delete")

	for _, prefix := range []string{convBasePrefix(dropped), convToolPrefix(dropped), convThinkPrefix(dropped)} {
		if count := h.countRowsWithPrefix(prefix); count != 0 {
			t.Fatalf("after authoritative delete, %d rows remain under %q, want 0", count, prefix)
		}
	}
	if count := h.countRowsWithPrefix(convBasePrefix(kept)); count <= 0 {
		t.Fatalf("kept conversation %s lost its rows (%d) after an unrelated delete", kept, count)
	}
}

// TestScenario5PostBootstrapEmbedsZero proves the migration path: after deleting
// the on-disk derived marker, a reexamine auto-stamps the already fully embedded
// conversations and the subsequent examination embeds nothing.
func TestScenario5PostBootstrapEmbedsZero(t *testing.T) {
	h := newHarness(t)

	convs := seedConversations()
	backfill := h.upsert(convs, pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN, true)
	requireCompleted(t, backfill, "backfill")

	// Simulate a pre-marker install: remove the derived marker sidecar so the next
	// reexamine sees an empty marker store and runs the one-time bootstrap stamp.
	markerPath := h.derivedMarkerPath()
	if err := os.Remove(markerPath); err != nil {
		t.Fatalf("remove derived marker %s returned error: %v", markerPath, err)
	}

	migrated := h.upsert(convs, pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN, true)
	requireCompleted(t, migrated, "post-bootstrap reexamine")

	if migrated.Progress.ChunksEmbedded != 0 {
		t.Fatalf("post-bootstrap ChunksEmbedded = %d, want 0 (bootstrap stamped, nothing to embed)\n%s", migrated.Progress.ChunksEmbedded, progressString(migrated))
	}
	if migrated.Progress.FilesEmbedded != 0 {
		t.Fatalf("post-bootstrap FilesEmbedded = %d, want 0\n%s", migrated.Progress.FilesEmbedded, progressString(migrated))
	}

	// The auto-bootstrap must have re-stamped every fully embedded conversation at
	// the current pipeline version, recreating the marker sidecar it deleted.
	versions := h.readDerivedMarkers(markerPath)
	for _, id := range seedConversationIDs {
		if versions[id] != "1" {
			t.Fatalf("bootstrap did not stamp %s at version 1; markers = %v", id, versions)
		}
	}
}

// upsert drives one client-streaming conversation ingest over gRPC (header,
// documents, manifest, CloseAndRecv), then polls the job to a terminal state and
// returns the full model.Job so a test can read the per-run progress the wire
// Progress does not expose (FilesEmbedded and FilesModified).
func (h *harness) upsert(convs map[string][]*pb.ConversationDocument, reconcile pb.ConversationReconcileMode, reexamine bool) model.Job {
	h.t.Helper()

	stream, err := h.client.UpsertConversationDocumentsStream(correlatedContext())
	if err != nil {
		h.t.Fatalf("open UpsertConversationDocumentsStream returned error: %v", err)
	}

	header := &pb.UpsertConversationDocumentsChunk{
		Chunk: &pb.UpsertConversationDocumentsChunk_Header{Header: &pb.UpsertConversationDocumentsHeader{
			CollectionId:       h.collectionID,
			Client:             &pb.ClientInfo{Name: "live-harness"},
			ReconcileMode:      reconcile,
			ReexamineDelivered: reexamine,
		}},
	}
	if err := stream.Send(header); err != nil {
		h.t.Fatalf("send header returned error: %v", err)
	}

	documents := make([]*pb.ConversationDocument, 0)
	manifest := make([]*pb.ConversationFingerprint, 0, len(convs))
	for _, id := range sortedKeys(convs) {
		documents = append(documents, convs[id]...)
		manifest = append(manifest, &pb.ConversationFingerprint{ConversationId: id, Fingerprint: fingerprint(convs[id])})
	}
	documentsChunk := &pb.UpsertConversationDocumentsChunk{
		Chunk: &pb.UpsertConversationDocumentsChunk_Documents{Documents: &pb.UpsertConversationDocumentsDocuments{Documents: documents}},
	}
	if err := stream.Send(documentsChunk); err != nil {
		h.t.Fatalf("send documents returned error: %v", err)
	}
	manifestChunk := &pb.UpsertConversationDocumentsChunk{
		Chunk: &pb.UpsertConversationDocumentsChunk_Manifest{Manifest: &pb.UpsertConversationDocumentsManifest{Manifest: manifest}},
	}
	if err := stream.Send(manifestChunk); err != nil {
		h.t.Fatalf("send manifest returned error: %v", err)
	}

	response, err := stream.CloseAndRecv()
	if err != nil {
		h.t.Fatalf("CloseAndRecv returned error: %v", err)
	}
	jobID := response.GetJobId()
	if jobID == "" {
		h.t.Fatal("CloseAndRecv returned an empty job id")
	}
	return h.waitJob(jobID)
}

// waitJob polls the in-process manager for the job until it reaches a terminal
// state, so the test reads the full model.Job progress rather than the reduced
// wire Progress.
func (h *harness) waitJob(jobID string) model.Job {
	h.t.Helper()
	deadline := time.Now().Add(jobPollTimeout)
	for time.Now().Before(deadline) {
		job, found := h.manager.GetJob(jobID)
		if found {
			switch job.State {
			case model.JobStateCompleted, model.JobStateFailed, model.JobStateCancelled:
				return job
			}
		}
		time.Sleep(jobPollInterval)
	}
	h.t.Fatalf("job %s did not reach a terminal state within %s", jobID, jobPollTimeout)
	return model.Job{}
}

// countRowsWithPrefix counts rows in the throwaway collection whose relativePath
// begins with prefix, using a direct Milvus count(*) query at strong consistency.
func (h *harness) countRowsWithPrefix(prefix string) int64 {
	h.t.Helper()
	expression := fmt.Sprintf(`%s like "%s%%"`, relativePathField, prefix)
	resultSet, err := h.milvus.Query(correlatedContext(), milvusclient.NewQueryOption(h.collectionName).
		WithFilter(expression).
		WithOutputFields(countOutputField).
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		h.t.Fatalf("Milvus count query for %q returned error: %v", prefix, err)
	}
	column := resultSet.GetColumn(countOutputField)
	if column == nil {
		h.t.Fatalf("Milvus count query for %q returned no count column", prefix)
	}
	total, err := column.GetAsInt64(0)
	if err != nil {
		h.t.Fatalf("read count column for %q returned error: %v", prefix, err)
	}
	return total
}

// derivedMarkerPath is the on-disk derived-marker sidecar for this collection,
// beside the merkle snapshot at <merkleDir>/<codebaseID>.json.derived.
func (h *harness) derivedMarkerPath() string {
	return fmt.Sprintf("%s/%s.json.derived", h.merkleDir, h.codebaseID)
}

// readDerivedMarkers reads the marker sidecar and returns its conversation ->
// version map.
func (h *harness) readDerivedMarkers(path string) map[string]string {
	h.t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		h.t.Fatalf("read derived markers %s returned error: %v", path, err)
	}
	var file struct {
		Versions map[string]string `json:"versions"`
	}
	if err := json.Unmarshal(data, &file); err != nil {
		h.t.Fatalf("unmarshal derived markers %s returned error: %v", path, err)
	}
	return file.Versions
}

func requireCompleted(t *testing.T, job model.Job, label string) {
	t.Helper()
	if job.State != model.JobStateCompleted {
		message := ""
		if job.Error != nil {
			message = job.Error.Message
		}
		t.Fatalf("%s job state = %q, want completed (error: %s)", label, job.State, message)
	}
}

func progressString(job model.Job) string {
	return fmt.Sprintf(
		"progress: state=%s filesTotal=%d filesModified=%d filesEmbedded=%d chunksProcessed=%d chunksReused=%d chunksEmbedded=%d",
		job.State, job.Progress.FilesTotal, job.Progress.FilesModified, job.Progress.FilesEmbedded,
		job.Progress.ChunksProcessed, job.Progress.ChunksReused, job.Progress.ChunksEmbedded,
	)
}

// seedConversations builds the three-conversation fixture fresh each call, so a
// test can mutate its own copy without disturbing another.
func seedConversations() map[string][]*pb.ConversationDocument {
	convs := make(map[string][]*pb.ConversationDocument, len(seedConversationIDs))
	for _, id := range seedConversationIDs {
		convs[id] = baseConversation(id)
	}
	return convs
}

// baseConversation is a two-message conversation whose assistant turn carries a
// bash tool call and thinking text, so it produces conv/, convtool/, and
// convthink/ rows.
func baseConversation(id string) []*pb.ConversationDocument {
	return []*pb.ConversationDocument{
		{
			ConversationId: id,
			MessageIndex:   0,
			Role:           "user",
			TimestampUnix:  1712345000,
			Text:           "please list the files in " + id,
		},
		{
			ConversationId: id,
			MessageIndex:   1,
			Role:           "assistant",
			TimestampUnix:  1712345001,
			Text:           "listing the files now for " + id,
			Thinking:       "the user for " + id + " wants a directory listing, so I will run ls",
			Tools: []*pb.ConversationToolCall{
				{
					Name:      "run_shell",
					InputJson: `{"cmd":"ls -la /work/` + id + `"}`,
					Command:   "ls -la /work/" + id,
					LangHint:  "bash",
					Output:    "total 0\ndrwxr-xr-x  2 user  staff   64 " + id,
					IsError:   false,
				},
			},
		},
	}
}

// appendMessage returns a copy of docs with one additional assistant message, so
// both the content and the derived fingerprint change.
func appendMessage(docs []*pb.ConversationDocument, id string) []*pb.ConversationDocument {
	extended := make([]*pb.ConversationDocument, len(docs), len(docs)+1)
	copy(extended, docs)
	nextIndex := int32(len(docs))
	extended = append(extended, &pb.ConversationDocument{
		ConversationId: id,
		MessageIndex:   nextIndex,
		Role:           "assistant",
		TimestampUnix:  1712345002,
		Text:           "here is a follow-up answer for " + id,
		Thinking:       "adding one more turn to " + id,
	})
	return extended
}

// fingerprint hashes a conversation's ordered message content, so identical
// content yields an identical fingerprint and any change yields a new one. The
// engine compares these for equality to detect which conversations changed.
func fingerprint(docs []*pb.ConversationDocument) string {
	hasher := sha256.New()
	for _, document := range docs {
		fmt.Fprintf(hasher, "%d\x00%s\x00%s\x00%s\x00", document.MessageIndex, document.Role, document.Text, document.Thinking)
		for _, tool := range document.Tools {
			fmt.Fprintf(hasher, "%s\x00%s\x00%s\x00%s\x00", tool.Name, tool.Command, tool.InputJson, tool.Output)
		}
	}
	return hex.EncodeToString(hasher.Sum(nil))
}

func convBasePrefix(id string) string  { return "conv/" + id + "/" }
func convToolPrefix(id string) string  { return "convtool/" + id + "/" }
func convThinkPrefix(id string) string { return "convthink/" + id + "/" }

func sortedKeys(convs map[string][]*pb.ConversationDocument) []string {
	keys := make([]string, 0, len(convs))
	for key := range convs {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
