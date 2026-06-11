package render

import (
	"strings"

	"goodkind.io/lm-semantic-search/internal/view"
)

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
