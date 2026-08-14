//go:build restartacceptance

package restartacceptance

import (
	"context"
	"net"
	"regexp"

	"goodkind.io/lm-semantic-search/test/sandboxharness"
)

type embeddingProxy = sandboxharness.EmbeddingProxy
type milvusProxy = sandboxharness.EmbeddingStoreProxy

var acceptanceInputPattern = regexp.MustCompile(`restart_acceptance_id:([0-9]+\.go)`)

func newEmbeddingProxy(listener net.Listener, backendURL string) (*embeddingProxy, error) {
	return sandboxharness.StartEmbeddingProxy(sandboxharness.EmbeddingProxyOptions{
		Listener:   listener,
		BackendURL: backendURL,
		IdentifyInput: func(input string) (string, bool) {
			match := acceptanceInputPattern.FindStringSubmatch(input)
			if len(match) != 2 {
				return "", false
			}
			return match[1], true
		},
		Start: false,
	})
}

func newMilvusProxy(listener net.Listener, backendAddress string) (*milvusProxy, error) {
	return sandboxharness.StartEmbeddingStoreProxy(sandboxharness.EmbeddingStoreProxyOptions{
		Listener:       listener,
		BackendAddress: backendAddress,
		Start:          false,
	})
}

func waitForRelay(ctx context.Context, cancel context.CancelFunc, client <-chan error, server <-chan error) error {
	return sandboxharness.WaitForRelay(ctx, cancel, client, server)
}
