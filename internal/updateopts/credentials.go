package updateopts

import (
	"context"
	"log/slog"
	"os"
	"os/exec"
	"strings"
	"time"
)

const githubCLITimeout = 5 * time.Second

// githubTokenEnvNames lists the environment variables read for a GitHub token,
// in priority order.
var githubTokenEnvNames = []string{"GH_TOKEN", "GITHUB_TOKEN"}

// resolveGitHubToken returns a GitHub token for release queries and asset
// downloads. It reads GH_TOKEN, then GITHUB_TOKEN, then the output of
// `gh auth token --hostname github.com`. It returns an empty string when none
// resolves, and the updater then sends unauthenticated requests. GitHub limits
// unauthenticated API requests to 60 per hour per address. Logs record only
// the credential source, never the token.
func resolveGitHubToken(ctx context.Context, log *slog.Logger) string {
	if log == nil {
		log = slog.Default()
	}
	for _, name := range githubTokenEnvNames {
		if token := strings.TrimSpace(os.Getenv(name)); token != "" {
			log.DebugContext(ctx, "update.credentials.resolved", "source", name)
			return token
		}
	}
	helperContext, cancel := context.WithTimeout(ctx, githubCLITimeout)
	defer cancel()
	output, err := exec.CommandContext(helperContext, "gh", "auth", "token", "--hostname", "github.com").Output()
	if err != nil {
		log.DebugContext(ctx, "update.credentials.unavailable", "source", "gh")
		return ""
	}
	token := strings.TrimSpace(string(output))
	if token == "" {
		log.DebugContext(ctx, "update.credentials.unavailable", "source", "gh")
		return ""
	}
	log.DebugContext(ctx, "update.credentials.resolved", "source", "gh")
	return token
}
