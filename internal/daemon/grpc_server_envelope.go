package daemon

import (
	"context"
	"strings"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	render "goodkind.io/lm-semantic-search/internal/render"
	"goodkind.io/lm-semantic-search/internal/view"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// envelopeText builds the human-facing display text for a daemon response.
// It joins the maintenance banner, dependency-health banner, and body, in
// that order, with single newlines. The maintenance banner appears only
// while maintenance mode is on. The dependency-health banner appears only
// while a shared dependency is degraded.
//
// Only envelopeText adds these banners. Body renderers do not add them.
// The caller passes the health snapshot it already used to build the body.
func (server *GRPCServer) envelopeText(health dependencyHealth, body string) string {
	banner := joinBannerLines(
		render.MaintenanceBanner(resolveMaintenanceView(server.manager.Maintenance(), 0)),
		render.HealthBanner(resolveBannerView(health, server.manager.config)),
	)
	if banner == "" {
		return body
	}
	if strings.TrimSpace(body) == "" {
		return banner
	}
	return banner + "\n" + body
}

// joinBannerLines joins the banners that are present with single newlines, so
// the envelope carries no blank line when only one of them shows.
func joinBannerLines(banners ...string) string {
	present := make([]string, 0, len(banners))
	for _, banner := range banners {
		if banner != "" {
			present = append(present, banner)
		}
	}
	return strings.Join(present, "\n")
}

// resolveMaintenanceView reduces the persisted mode to the view the render
// layer formats, with the start time already rendered in the host's zone.
func resolveMaintenanceView(state model.MaintenanceState, activeJobs int) view.MaintenanceView {
	sinceLabel := ""
	if state.Enabled && !state.Since.IsZero() {
		sinceLabel = formatStatusTime(state.Since)
	}
	return view.MaintenanceView{
		Enabled:    state.Enabled,
		Reason:     state.Reason,
		SinceLabel: sinceLabel,
		ActiveJobs: activeJobs,
	}
}

// toMaintenanceStatus converts the persisted mode to its wire form. The start
// time is omitted while zero, so a mode that is off carries no stamp.
func toMaintenanceStatus(state model.MaintenanceState) *pb.MaintenanceStatus {
	result := &pb.MaintenanceStatus{
		Enabled: state.Enabled,
		Reason:  state.Reason,
		Since:   nil,
	}
	if !state.Since.IsZero() {
		result.Since = timestamppb.New(state.Since)
	}
	return result
}

// refuseDuringMaintenance returns the gRPC refusal for a mutation the daemon
// does not accept in maintenance mode, or nil while the mode is off.
func (server *GRPCServer) refuseDuringMaintenance(ctx context.Context) error {
	refusal := server.manager.maintenanceRefusal()
	if refusal == nil {
		return nil
	}
	return status.Error(adapterr.Respond(ctx, refusal))
}
