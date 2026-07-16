//go:build live

package live

import (
	"fmt"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

// TestConversationProgressAdvancesPerBatch proves the job's chunk counters and
// heartbeat advance while a single large conversation is still embedding, not
// only when it finishes. It seeds one conversation large enough to embed in
// many batches, gates the fake embedder so the test releases one batch at a
// time, and reads the job between batches. Because the gate blocks each embed
// request until the test releases it, when batch N+1 arrives the daemon has
// already folded in batch N, so the sampled sequence is a race-free record of
// progress during the run.
//
// On per-conversation progress (the current behavior) chunksProcessed and
// heartbeatAt stay flat until the whole conversation completes, so every sample
// is the initial value and this test fails. With per-batch progress they climb
// across batches and the test passes.
func TestConversationProgressAdvancesPerBatch(t *testing.T) {
	gate := &embedGate{arrived: make(chan int), release: make(chan struct{})}
	h := newHarnessWithGate(t, gate)

	convs := map[string][]*pb.ConversationDocument{"big": bigConversation("big", 60)}
	jobID := h.startUpsert(convs, pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN, true, false)

	processedSeen := make([]int32, 0)
	heartbeats := make([]time.Time, 0)
	batches := 0
	for done := false; !done; {
		select {
		case <-gate.arrived:
			batches++
			if job, ok := h.manager.GetJob(jobID); ok {
				processedSeen = append(processedSeen, job.Progress.ChunksProcessed)
				heartbeats = append(heartbeats, job.Progress.HeartbeatAt)
			}
			gate.release <- struct{}{}
		case <-time.After(15 * time.Second):
			// No embed request has arrived for a while, so the job has drained.
			done = true
		}
	}

	job := h.waitJob(jobID)
	requireCompleted(t, job, "gated big conversation")

	if batches < 3 {
		t.Fatalf("expected many embed batches for a large conversation, saw %d; the fixture is too small to test granularity", batches)
	}

	distinct := make(map[int32]bool)
	var maxProcessed int32
	for _, processed := range processedSeen {
		distinct[processed] = true
		if processed > maxProcessed {
			maxProcessed = processed
		}
	}
	if len(distinct) < 2 || maxProcessed == 0 {
		t.Fatalf("chunksProcessed did not advance during the conversation (per-batch progress missing): saw %v across %d batches", processedSeen, batches)
	}

	if len(heartbeats) >= 2 && !heartbeats[len(heartbeats)-1].After(heartbeats[0]) {
		t.Fatalf("heartbeat did not advance during the conversation: first=%s last=%s", heartbeats[0], heartbeats[len(heartbeats)-1])
	}
}

// bigConversation builds one conversation with messages assistant turns, each
// carrying text, thinking, and a tool call, so it produces many base, tool, and
// thinking chunks and therefore many embed batches under the harness's small
// batch budget.
func bigConversation(id string, messages int) []*pb.ConversationDocument {
	docs := make([]*pb.ConversationDocument, 0, messages)
	for index := 0; index < messages; index++ {
		docs = append(docs, &pb.ConversationDocument{
			ConversationId: id,
			MessageIndex:   int32(index),
			Role:           "assistant",
			TimestampUnix:  1712345000 + int64(index),
			Text:           fmt.Sprintf("message %d for %s: ran a command and summarized the result in a sentence or two", index, id),
			Thinking:       fmt.Sprintf("thinking about step %d of %s: decide the next command and why it is the right one", index, id),
			Tools: []*pb.ConversationToolCall{
				{
					Name:      "run_shell",
					InputJson: fmt.Sprintf(`{"cmd":"ls -la /work/%s/%d"}`, id, index),
					Command:   fmt.Sprintf("ls -la /work/%s/%d", id, index),
					LangHint:  "bash",
					Output:    fmt.Sprintf("total %d\nfile-%d-alpha\nfile-%d-bravo\nfile-%d-charlie", index, index, index, index),
					IsError:   false,
				},
			},
		})
	}
	return docs
}

// startUpsert streams one conversation ingest and returns the job id without
// waiting for the job to finish, so the caller can observe progress while the
// gated embedder is still running. It mirrors the streaming in upsert.
func (h *harness) startUpsert(convs map[string][]*pb.ConversationDocument, reconcile pb.ConversationReconcileMode, backfill bool, force bool) string {
	h.t.Helper()

	stream, err := h.client.UpsertConversationDocumentsStream(correlatedContext())
	if err != nil {
		h.t.Fatalf("open UpsertConversationDocumentsStream returned error: %v", err)
	}

	header := &pb.UpsertConversationDocumentsChunk{
		Chunk: &pb.UpsertConversationDocumentsChunk_Header{Header: &pb.UpsertConversationDocumentsHeader{
			CollectionId:      h.collectionID,
			Client:            &pb.ClientInfo{Name: "progress-harness"},
			ReconcileMode:     reconcile,
			BackfillDelivered: backfill,
			ForceReexamine:    force,
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
	if err := stream.Send(&pb.UpsertConversationDocumentsChunk{
		Chunk: &pb.UpsertConversationDocumentsChunk_Documents{Documents: &pb.UpsertConversationDocumentsDocuments{Documents: documents}},
	}); err != nil {
		h.t.Fatalf("send documents returned error: %v", err)
	}
	if err := stream.Send(&pb.UpsertConversationDocumentsChunk{
		Chunk: &pb.UpsertConversationDocumentsChunk_Manifest{Manifest: &pb.UpsertConversationDocumentsManifest{Manifest: manifest}},
	}); err != nil {
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
	return jobID
}
