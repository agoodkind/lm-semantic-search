package daemon

import (
	"errors"
	"io"
	"log/slog"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	render "goodkind.io/lm-semantic-search/internal/render"
	"goodkind.io/lm-semantic-search/internal/view"
	"google.golang.org/grpc/status"
)

// UpsertConversationDocumentsStream is the client-streaming form of
// UpsertConversationDocuments. clyde sends one header chunk, then document
// chunks, then one manifest chunk, so the document set and the manifest are not
// bounded by the gRPC max message size. The handler accumulates the chunks and
// queues the same async job the unary RPC queues, then replies once with the job
// id through SendAndClose.
func (server *GRPCServer) UpsertConversationDocumentsStream(stream pb.SemanticSearchDaemonService_UpsertConversationDocumentsStreamServer) (err error) {
	ctx, done := beginRPC(stream.Context(), "UpsertConversationDocumentsStream")
	defer done(&err)

	collectionID := ""
	var client model.ClientInfo
	headerSeen := false
	documents := make([]model.ConversationDocument, 0)
	var manifest map[string]string
	for {
		chunk, recvErr := stream.Recv()
		if errors.Is(recvErr, io.EOF) {
			break
		}
		if recvErr != nil {
			slog.ErrorContext(ctx, "receive upsert conversation documents chunk failed", "err", recvErr)
			return status.Error(adapterr.Respond(ctx, adapterr.NewInternal("receive upsert conversation documents chunk", recvErr)))
		}
		switch payload := chunk.GetChunk().(type) {
		case *pb.UpsertConversationDocumentsChunk_Header:
			collectionID = payload.Header.GetCollectionId()
			client = pbClient(payload.Header.GetClient())
			headerSeen = true
		case *pb.UpsertConversationDocumentsChunk_Documents:
			documents = append(documents, pbConversationDocuments(payload.Documents.GetDocuments())...)
		case *pb.UpsertConversationDocumentsChunk_Manifest:
			manifest = pbConversationManifest(payload.Manifest.GetManifest())
		default:
			// An unset oneof carries no payload. Ignore it rather than fail, since a
			// future chunk variant must not break an older engine mid-stream.
		}
	}
	if !headerSeen {
		return status.Error(adapterr.Respond(ctx, adapterr.NewMissingArgument("header")))
	}
	if argErr := requireNonEmpty(ctx, collectionID, "collection_id", false); argErr != nil {
		return argErr
	}

	job, callErr := server.manager.upsertConversationDocuments(ctx, collectionID, documents, manifest, client)
	if callErr != nil {
		return status.Error(adapterr.Respond(ctx, classifyManagerError(collectionID, callErr)))
	}
	ack := view.MutationAckView{
		Kind:            view.AckUpsertConversation,
		Path:            "",
		JobID:           job.ID,
		StateLabel:      "",
		AlreadyTerminal: false,
		Deduplicated:    false,
		CollectionID:    collectionID,
		CollectionName:  "",
		CodebaseID:      job.CodebaseID,
		ConversationID:  "",
		DocumentCount:   len(documents),
		NeededCount:     0,
		TotalCount:      0,
	}
	health := server.manager.DependencyHealth()
	response := &pb.UpsertConversationDocumentsResponse{
		JobId: job.ID,
		DisplayText: server.envelopeText(
			ctx,
			health,
			render.MutationAck(ack),
			"codebase_id",
			job.CodebaseID,
			"job_id",
			job.ID,
		),
	}
	if sendErr := stream.SendAndClose(response); sendErr != nil {
		slog.ErrorContext(ctx, "send upsert conversation documents response failed", "err", sendErr)
		return status.Error(adapterr.Respond(ctx, adapterr.NewInternal("send upsert conversation documents response", sendErr)))
	}
	return nil
}
