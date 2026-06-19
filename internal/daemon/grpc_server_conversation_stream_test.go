package daemon

import (
	"context"
	"io"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/model"
	"google.golang.org/grpc"
)

// fakeUpsertStreamServer replays a fixed list of request chunks through Recv and
// captures the single SendAndClose response, so a test can drive the
// client-streaming upsert handler without a live gRPC connection. It embeds the
// generated server interface so the unused stream methods satisfy the type; only
// Recv, SendAndClose, and Context are exercised.
type fakeUpsertStreamServer struct {
	grpc.ClientStreamingServer[pb.UpsertConversationDocumentsChunk, pb.UpsertConversationDocumentsResponse]
	chunks   []*pb.UpsertConversationDocumentsChunk
	cursor   int
	response *pb.UpsertConversationDocumentsResponse
}

func (f *fakeUpsertStreamServer) Recv() (*pb.UpsertConversationDocumentsChunk, error) {
	if f.cursor >= len(f.chunks) {
		return nil, io.EOF
	}
	chunk := f.chunks[f.cursor]
	f.cursor++
	return chunk, nil
}

func (f *fakeUpsertStreamServer) SendAndClose(response *pb.UpsertConversationDocumentsResponse) error {
	f.response = response
	return nil
}

func (f *fakeUpsertStreamServer) Context() context.Context { return context.Background() }

func TestUpsertConversationDocumentsStreamQueuesJob(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	upsertedChunks := make(chan []model.StoredChunk, 1)
	manager.semantic = &fakeSemantic{
		reindex: func(ctx context.Context, codebasePath string, chunks []model.StoredChunk, removed []string) error {
			_ = ctx
			_ = codebasePath
			_ = removed
			if len(chunks) > 0 {
				select {
				case upsertedChunks <- append([]model.StoredChunk{}, chunks...):
				default:
				}
			}
			return nil
		},
	}
	server := NewGRPCServer(manager, nil)

	// The documents arrive across two documents chunks to prove the handler
	// concatenates batches in receive order; the manifest arrives as its own
	// authoritative chunk after the documents.
	stream := &fakeUpsertStreamServer{
		ClientStreamingServer: nil,
		chunks: []*pb.UpsertConversationDocumentsChunk{
			{Chunk: &pb.UpsertConversationDocumentsChunk_Header{Header: &pb.UpsertConversationDocumentsHeader{
				CollectionId: "thread-stream-jobs",
				Client:       &pb.ClientInfo{Name: "test", Pid: 0, CallerCwd: ""},
			}}},
			{Chunk: &pb.UpsertConversationDocumentsChunk_Documents{Documents: &pb.UpsertConversationDocumentsDocuments{
				Documents: []*pb.ConversationDocument{{
					ConversationId: "conv-stream",
					MessageIndex:   0,
					Role:           "user",
					TimestampUnix:  1712345678,
					Text:           "hello",
				}},
			}}},
			{Chunk: &pb.UpsertConversationDocumentsChunk_Documents{Documents: &pb.UpsertConversationDocumentsDocuments{
				Documents: []*pb.ConversationDocument{{
					ConversationId: "conv-stream",
					MessageIndex:   1,
					Role:           "assistant",
					TimestampUnix:  1712345679,
					Text:           "world",
				}},
			}}},
			{Chunk: &pb.UpsertConversationDocumentsChunk_Manifest{Manifest: &pb.UpsertConversationDocumentsManifest{
				Manifest: []*pb.ConversationFingerprint{{
					ConversationId: "conv-stream",
					Fingerprint:    "fp-1",
				}},
			}}},
		},
		cursor:   0,
		response: nil,
	}

	if err := server.UpsertConversationDocumentsStream(stream); err != nil {
		t.Fatalf("UpsertConversationDocumentsStream returned error: %v", err)
	}
	if stream.response == nil {
		t.Fatal("UpsertConversationDocumentsStream sent no response")
	}
	if stream.response.GetJobId() == "" {
		t.Fatal("UpsertConversationDocumentsStream returned an empty job id")
	}
	if !strings.Contains(stream.response.GetDisplayText(), "Started conversation ingest job") {
		t.Fatalf("stream DisplayText = %q, want ingest start text", stream.response.GetDisplayText())
	}

	select {
	case chunks := <-upsertedChunks:
		if len(chunks) != 2 {
			t.Fatalf("stream upsert passed %d chunks, want 2", len(chunks))
		}
	case <-time.After(5 * time.Second):
		t.Fatal("UpsertConversationDocumentsStream did not call semantic upsert")
	}
	waitForCondition(t, func() bool {
		job, found := manager.GetJob(stream.response.GetJobId())
		return found && job.State == model.JobStateCompleted
	})
}

func TestUpsertConversationDocumentsStreamRejectsMissingHeader(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	server := NewGRPCServer(manager, nil)
	stream := &fakeUpsertStreamServer{
		ClientStreamingServer: nil,
		chunks: []*pb.UpsertConversationDocumentsChunk{
			{Chunk: &pb.UpsertConversationDocumentsChunk_Documents{Documents: &pb.UpsertConversationDocumentsDocuments{
				Documents: []*pb.ConversationDocument{{
					ConversationId: "conv-stream",
					MessageIndex:   0,
					Role:           "user",
					TimestampUnix:  1712345678,
					Text:           "hello",
				}},
			}}},
		},
		cursor:   0,
		response: nil,
	}
	if err := server.UpsertConversationDocumentsStream(stream); err == nil {
		t.Fatal("UpsertConversationDocumentsStream accepted a stream with no header")
	}
	if stream.response != nil {
		t.Fatal("UpsertConversationDocumentsStream sent a response despite the missing header")
	}
}
