package daemon

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	render "goodkind.io/lm-semantic-search/internal/render"
	"goodkind.io/lm-semantic-search/internal/view"
	"google.golang.org/grpc/status"
)

const (
	// maxCollectionRowsPerFrame bounds the rows one rows chunk may send.
	maxCollectionRowsPerFrame = 1024
	// maxCollectionStreamBytes bounds the row bytes one item upsert stream may
	// send: row keys, item ids, text, scalar column names, and string values.
	maxCollectionStreamBytes = 64 << 20
)

// SyncCollectionManifest diffs a registered document collection's item
// manifest against the engine checkpoint and returns the item ids the engine
// needs.
func (server *GRPCServer) SyncCollectionManifest(ctx context.Context, request *pb.SyncCollectionManifestRequest) (resp *pb.SyncCollectionManifestResponse, err error) {
	ctx, done := beginRPC(ctx, "SyncCollectionManifest")
	defer done(&err)
	if argErr := requireNonEmpty(ctx, request.GetCollectionId(), "collection_id", false); argErr != nil {
		return nil, argErr
	}
	manifest, manifestErr := pbCollectionManifest(request.GetManifest())
	if manifestErr != nil {
		return nil, adapterr.RespondGRPC(ctx, manifestErr)
	}
	needed, callErr := server.manager.SyncCollectionManifest(ctx, request.GetCollectionId(), manifest)
	if callErr != nil {
		return nil, adapterr.RespondGRPC(ctx, classifyManagerError(request.GetCollectionId(), callErr))
	}
	ack := view.MutationAckView{
		Kind:            view.AckCollectionManifest,
		Path:            "",
		JobID:           "",
		StateLabel:      "",
		AlreadyTerminal: false,
		Deduplicated:    false,
		CollectionID:    request.GetCollectionId(),
		CollectionName:  "",
		CodebaseID:      request.GetCollectionId(),
		ConversationID:  "",
		DocumentCount:   0,
		NeededCount:     len(needed),
		TotalCount:      len(request.GetManifest()),
	}
	health := server.manager.DependencyHealth()
	return &pb.SyncCollectionManifestResponse{
		NeededItemIds: needed,
		DisplayText: server.envelopeText(
			ctx,
			health,
			render.MutationAck(ack),
			"codebase_id",
			request.GetCollectionId(),
		),
	}, nil
}

// collectionStreamState accumulates one item upsert stream. It enforces the
// frame order: one header first, then rows chunks, then at most one manifest
// chunk last.
type collectionStreamState struct {
	request      collectionItemsRequest
	headerSeen   bool
	manifestSeen bool
	streamBytes  int
}

// UpsertCollectionItemsStream is the client-streaming generic item upsert. The
// handler validates the frame order and the per-frame and per-stream bounds
// while it accumulates rows. It then validates every row against the saved
// declaration, queues one async ingest job, and replies once with the job id.
func (server *GRPCServer) UpsertCollectionItemsStream(stream pb.SemanticSearchDaemonService_UpsertCollectionItemsStreamServer) (err error) {
	ctx, done := beginRPC(stream.Context(), "UpsertCollectionItemsStream")
	defer done(&err)

	state := collectionStreamState{
		request: collectionItemsRequest{
			CollectionID: "",
			Client:       model.ClientInfo{Name: "", PID: 0},
			Rows:         make([]collectionRowInput, 0),
			Manifest:     nil,
			Absence:      absenceRetain,
			Backfill:     false,
			Force:        false,
		},
		headerSeen:   false,
		manifestSeen: false,
		streamBytes:  0,
	}
	for {
		chunk, recvErr := stream.Recv()
		if errors.Is(recvErr, io.EOF) {
			break
		}
		if recvErr != nil {
			slog.ErrorContext(ctx, "receive upsert collection items chunk failed", "err", recvErr)
			return status.Error(adapterr.Respond(ctx, adapterr.NewInternal("receive upsert collection items chunk", recvErr)))
		}
		if frameErr := server.acceptCollectionFrame(ctx, &state, chunk); frameErr != nil {
			return frameErr
		}
	}
	if !state.headerSeen {
		return adapterr.RespondGRPC(ctx, adapterr.NewMissingArgument("header"))
	}

	job, callErr := server.manager.upsertCollectionItems(ctx, state.request)
	if callErr != nil {
		return adapterr.RespondGRPC(ctx, classifyManagerError(state.request.CollectionID, callErr))
	}
	ack := view.MutationAckView{
		Kind:            view.AckUpsertCollectionItems,
		Path:            "",
		JobID:           job.ID,
		StateLabel:      "",
		AlreadyTerminal: false,
		Deduplicated:    false,
		CollectionID:    state.request.CollectionID,
		CollectionName:  "",
		CodebaseID:      job.CodebaseID,
		ConversationID:  "",
		DocumentCount:   len(state.request.Rows),
		NeededCount:     0,
		TotalCount:      0,
	}
	health := server.manager.DependencyHealth()
	response := &pb.UpsertCollectionItemsStreamResponse{
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
		slog.ErrorContext(ctx, "send upsert collection items response failed", "err", sendErr)
		return status.Error(adapterr.Respond(ctx, adapterr.NewInternal("send upsert collection items response", sendErr)))
	}
	return nil
}

// acceptCollectionFrame applies one stream frame to state and returns the gRPC
// error that ends the stream when the frame breaks the order or a bound.
func (server *GRPCServer) acceptCollectionFrame(ctx context.Context, state *collectionStreamState, chunk *pb.UpsertCollectionItemsStreamRequest) error {
	switch payload := chunk.GetChunk().(type) {
	case *pb.UpsertCollectionItemsStreamRequest_Header:
		return server.acceptCollectionHeader(ctx, state, payload.Header)
	case *pb.UpsertCollectionItemsStreamRequest_Rows:
		if !state.headerSeen {
			return adapterr.RespondGRPC(ctx, adapterr.NewMissingArgument("header"))
		}
		if state.manifestSeen {
			return adapterr.RespondGRPC(ctx, adapterr.NewInvalidArgument("rows chunk after the manifest chunk in collection upsert stream"))
		}
		rows := payload.Rows.GetRows()
		if len(rows) > maxCollectionRowsPerFrame {
			return adapterr.RespondGRPC(ctx, adapterr.NewInvalidArgument(fmt.Sprintf("rows chunk sends %d rows; the limit is %d", len(rows), maxCollectionRowsPerFrame)))
		}
		for _, row := range rows {
			input := pbCollectionRow(row)
			state.streamBytes += collectionRowInputBytes(input)
			if state.streamBytes > maxCollectionStreamBytes {
				return adapterr.RespondGRPC(ctx, adapterr.NewInvalidArgument(fmt.Sprintf("collection upsert stream exceeds %d row bytes; split the upsert into several streams", maxCollectionStreamBytes)))
			}
			state.request.Rows = append(state.request.Rows, input)
		}
		return nil
	case *pb.UpsertCollectionItemsStreamRequest_Manifest:
		if !state.headerSeen {
			return adapterr.RespondGRPC(ctx, adapterr.NewMissingArgument("header"))
		}
		if state.manifestSeen {
			return adapterr.RespondGRPC(ctx, adapterr.NewInvalidArgument("duplicate manifest chunk in collection upsert stream"))
		}
		manifest, manifestErr := pbCollectionManifest(payload.Manifest.GetManifest())
		if manifestErr != nil {
			return adapterr.RespondGRPC(ctx, manifestErr)
		}
		state.request.Manifest = manifest
		state.manifestSeen = true
		return nil
	default:
		// The handler skips an unset oneof. A future frame variant then does not
		// break an older engine mid-stream.
		return nil
	}
}

// acceptCollectionHeader binds the stream to one registered collection. It
// resolves the collection and applies the maintenance refusal before any row
// is buffered.
func (server *GRPCServer) acceptCollectionHeader(ctx context.Context, state *collectionStreamState, header *pb.UpsertCollectionItemsHeader) error {
	if state.headerSeen {
		return adapterr.RespondGRPC(ctx, adapterr.NewInvalidArgument("duplicate header in collection upsert stream"))
	}
	state.headerSeen = true
	state.request.CollectionID = header.GetCollectionId()
	state.request.Client = pbClient(header.GetClient())
	state.request.Absence = collectionAbsencePolicyFromProto(header.GetReconcileMode())
	state.request.Backfill = header.GetBackfillDelivered()
	state.request.Force = header.GetForceReexamine()
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

// collectionAbsencePolicyFromProto maps the wire reconcile mode to the internal
// absence policy. AUTHORITATIVE deletes items the manifest omits. RETAIN and
// the unset default keep them.
func collectionAbsencePolicyFromProto(mode pb.CollectionReconcileMode) absencePolicy {
	if mode == pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_AUTHORITATIVE {
		return absenceDeleteGuarded
	}
	return absenceRetain
}

// pbCollectionManifest converts wire fingerprints to a manifest. It rejects an
// empty item id.
func pbCollectionManifest(fingerprints []*pb.CollectionItemFingerprint) (map[string]string, error) {
	manifest := make(map[string]string, len(fingerprints))
	for _, fingerprint := range fingerprints {
		itemID := strings.TrimSpace(fingerprint.GetItemId())
		if itemID == "" {
			return nil, adapterr.NewMissingArgument("manifest item_id")
		}
		manifest[itemID] = fingerprint.GetFingerprint()
	}
	return manifest, nil
}

// pbCollectionRow converts one wire row. An unset scalar value converts to a
// null value.
func pbCollectionRow(row *pb.CollectionRow) collectionRowInput {
	return collectionRowInput{RowKey: row.GetRowKey(), ItemID: row.GetItemId(), Text: row.GetText(), Scalars: pbCollectionScalars(row.GetScalars())}
}

// pbCollectionScalars converts wire scalar values. An unset value converts to a
// null value.
func pbCollectionScalars(wireScalars []*pb.CollectionScalarValue) []collectionScalarInput {
	scalars := make([]collectionScalarInput, 0, len(wireScalars))
	for _, scalar := range wireScalars {
		value := model.ScalarValue{Type: "", Null: false, String: "", Bool: false, Int64: 0}
		switch typed := scalar.GetValue().(type) {
		case *pb.CollectionScalarValue_StringValue:
			value.Type = model.ScalarTypeString
			value.String = typed.StringValue
		case *pb.CollectionScalarValue_BoolValue:
			value.Type = model.ScalarTypeBool
			value.Bool = typed.BoolValue
		case *pb.CollectionScalarValue_Int64Value:
			value.Type = model.ScalarTypeInt64
			value.Int64 = typed.Int64Value
		default:
			value.Null = true
		}
		scalars = append(scalars, collectionScalarInput{Column: scalar.GetColumn(), Value: value})
	}
	return scalars
}

// collectionRowInputBytes counts the bytes one row adds to the stream bound.
func collectionRowInputBytes(input collectionRowInput) int {
	total := len(input.RowKey) + len(input.ItemID) + len(input.Text)
	for _, scalar := range input.Scalars {
		total += len(scalar.Column) + len(scalar.Value.String)
	}
	return total
}
