package daemon

import (
	"errors"
	"io"
	"log/slog"
	"strings"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/grpc/status"
)

// BackfillConversationScalars is the client-streaming form of the conversation
// scalar backfill. clyde sends one header chunk, then entry chunks carrying the
// conversation id to workspace root map. The handler accumulates the entries as
// workspaceRoot and archived values per conversation, runs the generic
// vector-preserving scalar backfill, and replies once through SendAndClose.
func (server *GRPCServer) BackfillConversationScalars(stream pb.SemanticSearchDaemonService_BackfillConversationScalarsServer) (err error) {
	ctx, done := beginRPC(stream.Context(), "BackfillConversationScalars")
	defer done(&err)

	collectionID := ""
	dryRun := false
	client := model.ClientInfo{Name: "", PID: 0}
	headerSeen := false
	values := make(map[string]map[string]model.ScalarValue)
	for {
		chunk, recvErr := stream.Recv()
		if errors.Is(recvErr, io.EOF) {
			break
		}
		if recvErr != nil {
			slog.ErrorContext(ctx, "receive backfill conversation scalars chunk failed", "err", recvErr)
			return status.Error(adapterr.Respond(ctx, adapterr.NewInternal("receive backfill conversation scalars chunk", recvErr)))
		}
		switch payload := chunk.GetChunk().(type) {
		case *pb.BackfillConversationScalarsChunk_Header:
			if headerSeen {
				return status.Error(adapterr.Respond(ctx, adapterr.NewInvalidArgument("duplicate header in conversation backfill stream")))
			}
			collectionID = payload.Header.GetCollectionId()
			dryRun = payload.Header.GetDryRun()
			client = pbClient(payload.Header.GetClient())
			headerSeen = true
			if argErr := requireNonEmpty(ctx, collectionID, "collection_id", false); argErr != nil {
				return argErr
			}
			if refusal := server.refuseDuringMaintenance(ctx); refusal != nil {
				return refusal
			}
		case *pb.BackfillConversationScalarsChunk_Entries:
			if !headerSeen {
				return status.Error(adapterr.Respond(ctx, adapterr.NewMissingArgument("header")))
			}
			addConversationScalarEntries(values, payload.Entries.GetEntries())
		default:
		}
	}
	if !headerSeen {
		return status.Error(adapterr.Respond(ctx, adapterr.NewMissingArgument("header")))
	}

	changed, orphan, callErr := server.manager.backfillConversationScalars(ctx, collectionID, values, dryRun)
	if callErr != nil {
		return status.Error(adapterr.Respond(ctx, classifyManagerError(collectionID, callErr)))
	}
	slog.InfoContext(ctx, "daemon.conversation_scalar_backfill_complete", "collection_id", collectionID, "client", client.Name, "changed", changed, "orphan", orphan, "dry_run", dryRun)
	health := server.manager.DependencyHealth()
	response := &pb.BackfillConversationScalarsResponse{
		Changed: int64(changed),
		Orphan:  int64(orphan),
		DisplayText: server.envelopeText(
			ctx,
			health,
			backfillScalarsDisplayText("conversation", collectionID, changed, orphan, dryRun),
			"codebase_id",
			collectionID,
		),
	}
	if sendErr := stream.SendAndClose(response); sendErr != nil {
		slog.ErrorContext(ctx, "send backfill conversation scalars response failed", "err", sendErr)
		return status.Error(adapterr.Respond(ctx, adapterr.NewInternal("send backfill conversation scalars response", sendErr)))
	}
	return nil
}

// addConversationScalarEntries records each entry's workspaceRoot and archived
// values under its trimmed conversation id. An entry without a conversation id
// is skipped, and a later entry for the same id replaces an earlier one.
func addConversationScalarEntries(values map[string]map[string]model.ScalarValue, entries []*pb.BackfillConversationScalarEntry) {
	for _, entry := range entries {
		if entry == nil {
			continue
		}
		conversationID := strings.TrimSpace(entry.GetConversationId())
		if conversationID == "" {
			continue
		}
		values[conversationID] = map[string]model.ScalarValue{
			semantic.ConversationWorkspaceRootColumn: {Type: model.ScalarTypeString, Null: false, String: entry.GetWorkspaceRoot(), Bool: false, Int64: 0},
			semantic.ConversationArchivedColumn:      {Type: model.ScalarTypeBool, Null: false, String: "", Bool: entry.GetArchived(), Int64: 0},
		}
	}
}
