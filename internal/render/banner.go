package render

import (
	"fmt"
	"strings"

	"goodkind.io/lm-semantic-search/internal/view"
)

// maintenanceEffects names what the daemon stops doing while maintenance is
// on, so every surface that mentions the mode describes it the same way.
const maintenanceEffects = "Background sync, repair, automatic rebuilds, collection loads, and search are paused until it is turned off."

// MaintenanceBanner returns the maintenance-mode banner, or an empty string
// while the mode is off. It sits above the dependency banner because it is a
// choice the operator made rather than a fault.
func MaintenanceBanner(maintenance view.MaintenanceView) string {
	if !maintenance.Enabled {
		return ""
	}
	headline := "🛠 Maintenance mode is on"
	if maintenance.SinceLabel != "" {
		headline += " since " + maintenance.SinceLabel
	}
	if maintenance.Reason != "" {
		headline += ": " + maintenance.Reason
	}
	return headline + "\n   " + maintenanceEffects
}

// MaintenanceAck renders the reply to a maintenance mode change.
func MaintenanceAck(maintenance view.MaintenanceView) string {
	if !maintenance.Enabled {
		return "Maintenance mode is off. Background sync, repair, automatic rebuilds, collection loads, and search resume on the next sweep."
	}
	ack := MaintenanceBanner(maintenance)
	if maintenance.ActiveJobs > 0 {
		ack += fmt.Sprintf("\n   %d %s still running; wait for %s to finish or cancel %s before touching the store.",
			maintenance.ActiveJobs,
			plural("job", maintenance.ActiveJobs),
			pluralPronoun(maintenance.ActiveJobs),
			pluralPronoun(maintenance.ActiveJobs),
		)
	}
	return ack
}

// pluralPronoun picks the object pronoun for a count of jobs.
func pluralPronoun(count int) string {
	if count == 1 {
		return "it"
	}
	return "them"
}

// HealthBanner returns the dependency-health banner for a resolved view.
func HealthBanner(banner view.BannerView) string {
	if strings.TrimSpace(banner.Headline) == "" {
		return ""
	}
	var buf strings.Builder
	if err := statusTemplates.ExecuteTemplate(&buf, "banner.md.tmpl", banner); err != nil {
		return "🟥 " + banner.Headline
	}
	return strings.TrimRight(buf.String(), "\n")
}
