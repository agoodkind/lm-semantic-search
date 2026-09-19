package daemon

import (
	"context"
	"log/slog"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	render "goodkind.io/lm-semantic-search/internal/render"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
)

// SetMaintenanceMode turns maintenance mode on or off in the running daemon.
// It is the one control an operator needs around a store backup or restore:
// on stops every daemon-driven store interaction and refuses new work, off
// lets the next sweep resume it. The reply names the jobs still running so
// the operator knows the store is not quiet yet.
func (server *GRPCServer) SetMaintenanceMode(ctx context.Context, request *pb.SetMaintenanceModeRequest) (resp *pb.SetMaintenanceModeResponse, err error) {
	ctx, done := beginRPC(ctx, "SetMaintenanceMode")
	defer done(&err)
	peerInfo, _ := peer.FromContext(ctx)
	slog.InfoContext(ctx, "maintenance mode change requested",
		"enabled", request.GetEnabled(),
		"reason", request.GetReason(),
		"client", request.GetClient().GetName(),
		"peer", peerInfo.String(),
	)
	state, activeJobs, callErr := server.manager.SetMaintenance(ctx, request.GetEnabled(), request.GetReason())
	if callErr != nil {
		return nil, status.Error(adapterr.Respond(ctx, adapterr.NewInternal("set maintenance mode", callErr)))
	}
	health := server.manager.DependencyHealth()
	maintenanceView := resolveMaintenanceView(state, activeJobs)
	return &pb.SetMaintenanceModeResponse{
		Maintenance: toMaintenanceStatus(state),
		ActiveJobs:  safeInt32(activeJobs),
		DisplayText: server.envelopeText(ctx, health, render.MaintenanceAck(maintenanceView)),
	}, nil
}
