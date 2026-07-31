package daemon

import (
	"strings"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/status"
	"goodkind.io/lm-semantic-search/internal/view"
)

// resolveBannerView returns the dependency-health banner view for the current
// health record, or an empty view when the shared dependencies are healthy. It
// reads the cached record and config only, so it never blocks on a live probe.
func resolveBannerView(health dependencyHealth, cfg config.Config) view.BannerView {
	if !health.Degraded() {
		return view.BannerView{Headline: "", Detail: ""}
	}
	// The time is read for the dependency this banner is about to name, so the
	// detail line never reports a store probe's round trip as the moment the
	// embedding endpoint last answered. A dependency that has not answered in this
	// daemon's lifetime reads as unknown, which is the true state of the evidence.
	lastReachable := "last reachable " + formatStatusTime(health.lastReachableAt())
	headline := status.BannerHeadlineFor(health.Mode)
	switch health.Mode {
	case dependencyEmbedderRejected:
		return view.BannerView{
			Headline: headline,
			Detail:   joinBannerDetail("Check the model name, dimensions, and credentials", embedderEndpointRef(cfg)),
		}
	case dependencyEmbedderPaused:
		return view.BannerView{
			Headline: headline,
			Detail:   joinBannerDetail("Leave low power mode or resume the embedding service", embedderEndpointRef(cfg)),
		}
	case dependencyStoreUnavailable:
		return view.BannerView{
			Headline: headline,
			Detail:   joinBannerDetail(storeEndpointRef(cfg), lastReachable),
		}
	case dependencyEmbedderUnreachable, dependencyEmbedderBusy:
		return view.BannerView{
			Headline: headline,
			Detail:   joinBannerDetail(embedderEndpointRef(cfg), lastReachable),
		}
	default:
		return view.BannerView{Headline: headline, Detail: lastReachable}
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
