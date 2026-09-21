package daemon

import (
	"context"
	"strings"

	"goodkind.io/gklog/correlation"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	render "goodkind.io/lm-semantic-search/internal/render"
	"goodkind.io/lm-semantic-search/internal/view"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// appendCorrelationRef prefixes one compact diagnostics line to a display
// text so every successful response starts with a greppable correlation
// header. Extras are key/value pairs for ids the trace context does not
// already carry, such as codebase_id and job_id.
func appendCorrelationRef(displayText string, ctx context.Context, extras ...string) string {
	corr := correlation.FromContext(ctx)
	line := correlation.HeaderLine(corr, extras...)
	if line == "" {
		return displayText
	}
	if strings.TrimSpace(displayText) == "" {
		return line
	}
	return line + "\n" + displayText
}

// envelopeText composes the human-facing display text for a read surface as the
// shared envelope: the maintenance banner (only while the operator's mode is
// on), the dependency-health banner (only when a shared dependency is
// degraded), then the correlation header, then the body, joined with single
// newlines. It is the one place the banners are prepended, so every surface
// shows each at most once and the body renderers never carry them. The caller
// passes the health snapshot it already read so the banner and the body agree.
func (server *GRPCServer) envelopeText(ctx context.Context, health dependencyHealth, body string, extras ...string) string {
	withHeader := appendCorrelationRef(body, ctx, extras...)
	banner := joinBannerLines(
		render.MaintenanceBanner(resolveMaintenanceView(server.manager.Maintenance(), 0)),
		render.HealthBanner(resolveBannerView(health, server.manager.config)),
	)
	if banner == "" {
		return withHeader
	}
	if strings.TrimSpace(withHeader) == "" {
		return banner
	}
	return banner + "\n" + withHeader
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
