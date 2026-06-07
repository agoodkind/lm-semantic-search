package daemon

import (
	"strings"

	"goodkind.io/lm-semantic-search/internal/config"
)

// bannerView is the data the dependency-health banner template renders. The
// banner is one global section the surface envelope prepends when a shared
// dependency is degraded; it is never per-codebase.
type bannerView struct {
	// Headline is the one-line statement of what is wrong and what it means.
	Headline string
	// Detail is the supporting line: the relevant environment variable and, when
	// known, the last-reachable time. Empty omits the line.
	Detail string
}

// renderHealthBanner returns the dependency-health banner for the current health
// record, or an empty string when the shared dependencies are healthy. It reads
// the cached record and config only, so it never blocks on a live probe. The
// variant matches the failure mode so the operator sees the specific cause once,
// at the top of every surface.
func renderHealthBanner(health dependencyHealth, cfg config.Config) string {
	if !health.Degraded() {
		return ""
	}
	view := bannerViewFor(health, cfg)
	var buf strings.Builder
	if err := statusTemplates.ExecuteTemplate(&buf, "banner.md.tmpl", view); err != nil {
		return "🟥 " + view.Headline
	}
	return strings.TrimRight(buf.String(), "\n")
}

func bannerViewFor(health dependencyHealth, cfg config.Config) bannerView {
	lastReachable := "last reachable " + formatStatusTime(health.LastHealthyAt)
	generic := bannerView{
		Headline: "A shared dependency is degraded. Indexing is paused until it recovers.",
		Detail:   lastReachable,
	}
	switch health.Mode {
	case dependencyEmbedderUnreachable:
		return bannerView{
			Headline: "Embedding server unreachable. Indexing is paused and resumes automatically when it returns.",
			Detail:   joinBannerDetail(embedderEndpointRef(cfg), lastReachable),
		}
	case dependencyEmbedderRejected:
		return bannerView{
			Headline: "Embedding server is rejecting requests. Indexing is paused until the embedding config is fixed.",
			Detail:   joinBannerDetail("Check the model name, dimensions, and credentials", embedderEndpointRef(cfg)),
		}
	case dependencyStoreUnavailable:
		return bannerView{
			Headline: "Vector store unavailable. Search and indexing are paused until it returns.",
			Detail:   joinBannerDetail(storeEndpointRef(cfg), lastReachable),
		}
	case dependencyHealthy:
		return generic
	default:
		return generic
	}
}

// embedderEndpointRef names the embedding endpoint by its env var so the operator
// can check the same setting the daemon reads.
func embedderEndpointRef(cfg config.Config) string {
	url := strings.TrimSpace(cfg.OpenAIBaseURL)
	if url == "" {
		return "OPENAI_BASE_URL is unset"
	}
	return "OPENAI_BASE_URL=" + url
}

func storeEndpointRef(cfg config.Config) string {
	address := strings.TrimSpace(cfg.MilvusAddress)
	if address == "" {
		return "MILVUS_ADDRESS is unset"
	}
	return "MILVUS_ADDRESS=" + address
}

func joinBannerDetail(parts ...string) string {
	kept := make([]string, 0, len(parts))
	for _, part := range parts {
		if strings.TrimSpace(part) != "" {
			kept = append(kept, part)
		}
	}
	return strings.Join(kept, " · ")
}
