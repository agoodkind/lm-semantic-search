package daemon

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	render "goodkind.io/lm-semantic-search/internal/render"
	"goodkind.io/lm-semantic-search/internal/view"
	"google.golang.org/grpc/status"
)

// collectionBackfillStreamState accumulates one scalar backfill stream. It
// enforces the frame order: one header first, then items chunks.
type collectionBackfillStreamState struct {
	request    collectionBackfillRequest
	headerSeen bool
}

// BackfillCollectionScalars is the client-streaming scalar backfill of a
// registered document collection. The handler accumulates the header and item
// frames, validates every item against the saved declaration, runs the
// backfill, and replies once with the row counts.
func (server *GRPCServer) BackfillCollectionScalars(stream pb.SemanticSearchDaemonService_BackfillCollectionScalarsServer) (err error) {
	ctx, done := beginRPC(stream.Context(), "BackfillCollectionScalars")
	defer done(&err)

	state := collectionBackfillStreamState{
		request: collectionBackfillRequest{
			CollectionID: "",
			Client:       model.ClientInfo{Name: "", PID: 0},
			Columns:      nil,
			Items:        make([]collectionBackfillItemInput, 0),
			DryRun:       false,
		},
		headerSeen: false,
	}
	for {
		chunk, recvErr := stream.Recv()
		if errors.Is(recvErr, io.EOF) {
			break
		}
		if recvErr != nil {
			slog.ErrorContext(ctx, "receive backfill collection scalars chunk failed", "err", recvErr)
			return status.Error(adapterr.Respond(ctx, adapterr.NewInternal("receive backfill collection scalars chunk", recvErr)))
		}
		if frameErr := server.acceptCollectionBackfillFrame(ctx, &state, chunk); frameErr != nil {
			return frameErr
		}
	}
	if !state.headerSeen {
		return adapterr.RespondGRPC(ctx, adapterr.NewMissingArgument("header"))
	}

	collectionID := state.request.CollectionID
	changed, orphan, callErr := server.manager.backfillCollectionItems(ctx, state.request)
	if callErr != nil {
		return adapterr.RespondGRPC(ctx, classifyManagerError(collectionID, callErr))
	}
	slog.InfoContext(ctx, "daemon.collection_scalar_backfill_complete", "collection_id", collectionID, "client", state.request.Client.Name, "changed", changed, "orphan", orphan, "dry_run", state.request.DryRun)
	health := server.manager.DependencyHealth()
	response := &pb.BackfillCollectionScalarsResponse{
		Changed: int64(changed),
		Orphan:  int64(orphan),
		DisplayText: server.envelopeText(
			ctx,
			health,
			backfillScalarsDisplayText("document", collectionID, changed, orphan, state.request.DryRun),
			"codebase_id",
			collectionID,
		),
	}
	if sendErr := stream.SendAndClose(response); sendErr != nil {
		slog.ErrorContext(ctx, "send backfill collection scalars response failed", "err", sendErr)
		return status.Error(adapterr.Respond(ctx, adapterr.NewInternal("send backfill collection scalars response", sendErr)))
	}
	return nil
}

// acceptCollectionBackfillFrame applies one stream frame to state and returns
// the gRPC error that ends the stream when the frame breaks the order.
func (server *GRPCServer) acceptCollectionBackfillFrame(ctx context.Context, state *collectionBackfillStreamState, chunk *pb.BackfillCollectionScalarsStreamRequest) error {
	switch payload := chunk.GetChunk().(type) {
	case *pb.BackfillCollectionScalarsStreamRequest_Header:
		return server.acceptCollectionBackfillHeader(ctx, state, payload.Header)
	case *pb.BackfillCollectionScalarsStreamRequest_Items:
		if !state.headerSeen {
			return adapterr.RespondGRPC(ctx, adapterr.NewMissingArgument("header"))
		}
		for _, item := range payload.Items.GetItems() {
			state.request.Items = append(state.request.Items, collectionBackfillItemInput{
				ItemID:  item.GetItemId(),
				Scalars: pbCollectionScalars(item.GetScalars()),
			})
		}
		return nil
	default:
		// The handler skips an unset oneof. A future frame variant then does not
		// break an older engine mid-stream.
		return nil
	}
}

// acceptCollectionBackfillHeader binds the stream to one registered
// collection. It applies the maintenance refusal before it resolves the
// collection or buffers any item.
func (server *GRPCServer) acceptCollectionBackfillHeader(ctx context.Context, state *collectionBackfillStreamState, header *pb.BackfillCollectionScalarsHeader) error {
	if state.headerSeen {
		return adapterr.RespondGRPC(ctx, adapterr.NewInvalidArgument("duplicate header in collection backfill stream"))
	}
	state.headerSeen = true
	state.request.CollectionID = header.GetCollectionId()
	state.request.Client = pbClient(header.GetClient())
	state.request.Columns = header.GetColumns()
	state.request.DryRun = header.GetDryRun()
	if argErr := requireNonEmpty(ctx, state.request.CollectionID, "collection_id", false); argErr != nil {
		return argErr
	}
	if refusal := server.refuseDuringMaintenance(ctx); refusal != nil {
		return refusal
	}
	if _, err := server.manager.registeredCollection(state.request.CollectionID); err != nil {
		return adapterr.RespondGRPC(ctx, err)
	}
	return nil
}

// DeleteCollectionItem queues a job that removes one item's rows from a
// registered document collection. It applies the maintenance refusal before it
// resolves the collection or queues the job.
func (server *GRPCServer) DeleteCollectionItem(ctx context.Context, request *pb.DeleteCollectionItemRequest) (resp *pb.DeleteCollectionItemResponse, err error) {
	ctx, done := beginRPC(ctx, "DeleteCollectionItem")
	defer done(&err)
	if argErr := requireNonEmpty(ctx, request.GetCollectionId(), "collection_id", false); argErr != nil {
		return nil, argErr
	}
	if argErr := requireNonEmpty(ctx, request.GetItemId(), "item_id", false); argErr != nil {
		return nil, argErr
	}
	if refusal := server.refuseDuringMaintenance(ctx); refusal != nil {
		return nil, refusal
	}
	job, callErr := server.manager.deleteCollectionItem(ctx, request.GetCollectionId(), request.GetItemId(), pbClient(request.GetClient()))
	if callErr != nil {
		return nil, adapterr.RespondGRPC(ctx, classifyManagerError(request.GetCollectionId(), callErr))
	}
	ack := view.MutationAckView{
		Kind:            view.AckDeleteCollectionItem,
		Path:            "",
		JobID:           job.ID,
		StateLabel:      "",
		AlreadyTerminal: false,
		Deduplicated:    false,
		CollectionID:    request.GetCollectionId(),
		CollectionName:  "",
		CodebaseID:      job.CodebaseID,
		ItemID:          request.GetItemId(),
		DocumentCount:   0,
		NeededCount:     0,
		TotalCount:      0,
	}
	health := server.manager.DependencyHealth()
	return &pb.DeleteCollectionItemResponse{
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
	}, nil
}

func backfillScalarsDisplayText(kind string, collectionID string, changed int, orphan int, dryRun bool) string {
	prefix := "Backfilled"
	if dryRun {
		prefix = "Dry run counted"
	}
	return fmt.Sprintf(
		"%s %s scalars for collection '%s': %d %s changed, %d orphan %s.",
		prefix,
		kind,
		collectionID,
		changed,
		plural("row", changed),
		orphan,
		plural("row", orphan),
	)
}
