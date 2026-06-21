package daemon

import (
	"errors"
	"fmt"
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
// conversation id to workspace root map. The handler accumulates the enrichment
// map, runs the synchronous vector-preserving semantic backfill, and replies once
// through SendAndClose.
func (server *GRPCServer) BackfillConversationScalars(stream pb.SemanticSearchDaemonService_BackfillConversationScalarsServer) (err error) {
	ctx, done := beginRPC(stream.Context(), "BackfillConversationScalars")
	defer done(&err)

	collectionID := ""
	dryRun := false
	client := model.ClientInfo{Name: "", PID: 0}
	headerSeen := false
	enrichment := make(semantic.ConversationEnrichment)
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
		case *pb.BackfillConversationScalarsChunk_Entries:
			if !headerSeen {
				return status.Error(adapterr.Respond(ctx, adapterr.NewMissingArgument("header")))
			}
			addConversationScalarEntries(enrichment, payload.Entries.GetEntries())
		default:
		}
	}
	if !headerSeen {
		return status.Error(adapterr.Respond(ctx, adapterr.NewMissingArgument("header")))
	}

	changed, orphan, callErr := server.manager.backfillConversationScalars(ctx, collectionID, enrichment, dryRun)
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
			backfillConversationScalarsDisplayText(collectionID, changed, orphan, dryRun),
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

func addConversationScalarEntries(enrichment semantic.ConversationEnrichment, entries []*pb.BackfillConversationScalarEntry) {
	for _, entry := range entries {
		if entry == nil {
			continue
		}
		conversationID := strings.TrimSpace(entry.GetConversationId())
		if conversationID == "" {
			continue
		}
		enrichment[conversationID] = semantic.ConversationEnrichmentValue{
			WorkspaceRoot: entry.GetWorkspaceRoot(),
			Archived:      entry.GetArchived(),
		}
	}
}

func backfillConversationScalarsDisplayText(collectionID string, changed int, orphan int, dryRun bool) string {
	prefix := "Backfilled"
	if dryRun {
		prefix = "Dry run counted"
	}
	return fmt.Sprintf(
		"%s conversation scalars for collection '%s': %d %s changed, %d orphan %s.",
		prefix,
		collectionID,
		changed,
		plural("row", changed),
		orphan,
		plural("row", orphan),
	)
}
