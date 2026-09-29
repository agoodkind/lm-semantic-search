package providers_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/embedding/providers"
	"goodkind.io/lm-semantic-search/internal/model"
)

const (
	productionModel   = "nvidia/NV-EmbedCode-7b-v1"
	productionAPIKey  = "provider-test-key" //gitleaks:allow // not a secret: the local test endpoint checks this literal
	wantAttempts      = 4
	wantBackoffBase   = 200 * time.Millisecond
	backoffTolerance  = 150 * time.Millisecond
	requestDimensions = 4096
)

// configEnvironment lists every variable that config.Default reads for the
// embedding provider, profile, and config and state roots. The test unsets each
// one. The config file and the context env file under the temporary home then
// set the result.
var configEnvironment = []string{
	"HOME",
	"XDG_CONFIG_HOME",
	"XDG_STATE_HOME",
	"CLAUDE_CONTEXTD_CONFIG_ROOT",
	"CLAUDE_CONTEXTD_STATE_ROOT",
	"CLAUDE_CONTEXT_PROFILE",
	"EMBEDDING_PROVIDER",
	"EMBEDDING_MODEL",
	"EMBEDDING_DIMENSION",
	"OFFLINE_EMBEDDING_MODEL",
	"OPENAI_API_KEY",
	"OPENAI_BASE_URL",
	"CLAUDE_CONTEXT_EMBEDDING_REQUEST_TIMEOUT_MS",
}

// embeddingEndpoint is a local OpenAI-compatible endpoint that answers every
// embeddings request with HTTP 429 and records each request.
type embeddingEndpoint struct {
	mutex    sync.Mutex
	arrivals []time.Time
	paths    []string
	bodies   []map[string]any
	headers  []string
}

func (endpoint *embeddingEndpoint) ServeHTTP(writer http.ResponseWriter, request *http.Request) {
	body := map[string]any{}
	_ = json.NewDecoder(request.Body).Decode(&body)
	endpoint.mutex.Lock()
	endpoint.arrivals = append(endpoint.arrivals, time.Now())
	endpoint.paths = append(endpoint.paths, request.URL.Path)
	endpoint.bodies = append(endpoint.bodies, body)
	endpoint.headers = append(endpoint.headers, request.Header.Get("Authorization"))
	endpoint.mutex.Unlock()
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(http.StatusTooManyRequests)
	_, _ = writer.Write([]byte(`{"error":{"message":"busy","type":"rate_limit"}}`))
}

// productionShapedHome writes a home directory shaped like the production LMS
// configuration: ~/.context/.env sets EMBEDDING_PROVIDER=OpenAI, the base URL,
// and the model, and ~/.config/lm-semantic-search/config.json sets the
// provider, model, API key, and hybrid mode with no profile and no dimension.
// extraConfig adds keys to config.json.
func productionShapedHome(t *testing.T, baseURL string, apiKey string, extraConfig map[string]any) {
	t.Helper()
	for _, name := range configEnvironment {
		t.Setenv(name, "")
		if err := os.Unsetenv(name); err != nil {
			t.Fatalf("unset %s: %v", name, err)
		}
	}
	home := t.TempDir()
	t.Setenv("HOME", home)
	contextDir := filepath.Join(home, ".context")
	if err := os.MkdirAll(contextDir, 0o700); err != nil {
		t.Fatalf("create context dir: %v", err)
	}
	envFile := "EMBEDDING_PROVIDER=OpenAI\nOPENAI_BASE_URL=" + baseURL + "\nEMBEDDING_MODEL=" + productionModel + "\n"
	if err := os.WriteFile(filepath.Join(contextDir, ".env"), []byte(envFile), 0o600); err != nil {
		t.Fatalf("write context env file: %v", err)
	}
	configDir := filepath.Join(home, ".config", "lm-semantic-search")
	if err := os.MkdirAll(configDir, 0o700); err != nil {
		t.Fatalf("create config dir: %v", err)
	}
	fileConfig := map[string]any{
		"embeddingProvider": "OpenAI",
		"embeddingModel":    productionModel,
		"openaiBaseUrl":     baseURL,
		"hybridMode":        true,
		"milvusAddress":     "localhost:19530",
	}
	if apiKey != "" {
		fileConfig["openaiApiKey"] = apiKey
	}
	for key, value := range extraConfig {
		fileConfig[key] = value
	}
	encoded, err := json.Marshal(fileConfig)
	if err != nil {
		t.Fatalf("encode config file: %v", err)
	}
	if err := os.WriteFile(filepath.Join(configDir, "config.json"), encoded, 0o600); err != nil {
		t.Fatalf("write config file: %v", err)
	}
}

// TestNewBuildsTheHostedProviderForTheProductionConfig loads a production-shaped
// configuration through config.Default and builds the daemon embedder with
// providers.New. The hosted OpenAI-compatible adapter serves it: requests go to
// the configured endpoint with the configured model, the API key as a bearer
// credential, and the dimension only when one is configured, and a busy
// endpoint gets 4 attempts with backoffs of 200, 400, and 800 ms. These are
// the values of the daemon adapter on origin/main before providers.New. The
// local endpoint answers every request with HTTP 429. The test checks which
// adapter providers.New selects and the requests and retries that adapter
// sends; it does not test embedding results, which the live suites test
// against the real endpoint.
func TestNewBuildsTheHostedProviderForTheProductionConfig(t *testing.T) {
	for _, testCase := range []struct {
		name          string
		extraConfig   map[string]any
		wantDimension float64
	}{
		{name: "production shape without a dimension", extraConfig: nil, wantDimension: 0},
		{name: "configured dimension", extraConfig: map[string]any{"embeddingDimension": requestDimensions}, wantDimension: requestDimensions},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			endpoint := &embeddingEndpoint{}
			server := httptest.NewServer(endpoint)
			defer server.Close()
			productionShapedHome(t, server.URL+"/v1", productionAPIKey, testCase.extraConfig)

			cfg, err := config.Default()
			if err != nil {
				t.Fatalf("config.Default: %v", err)
			}
			if cfg.Profile != config.ProfileStandard || cfg.EmbeddingProvider != model.EmbeddingProviderOpenAI {
				t.Fatalf("config.Default profile %q provider %q, want %q and %q", cfg.Profile, cfg.EmbeddingProvider, config.ProfileStandard, model.EmbeddingProviderOpenAI)
			}
			provider, err := providers.New(context.Background(), cfg)
			if err != nil {
				t.Fatalf("providers.New: %v", err)
			}
			if provider.ProviderName() != model.EmbeddingProviderOpenAI {
				t.Fatalf("provider %q, want the hosted %q adapter", provider.ProviderName(), model.EmbeddingProviderOpenAI)
			}
			if _, err := provider.Embed(context.Background(), "daemon reload bind gap"); err == nil {
				t.Fatal("Embed against a busy endpoint returned no error")
			}

			endpoint.mutex.Lock()
			defer endpoint.mutex.Unlock()
			if len(endpoint.arrivals) != wantAttempts {
				t.Fatalf("endpoint received %d requests, want %d attempts", len(endpoint.arrivals), wantAttempts)
			}
			for index := range endpoint.arrivals {
				if endpoint.paths[index] != "/v1/embeddings" {
					t.Fatalf("request %d path %q, want /v1/embeddings", index, endpoint.paths[index])
				}
				if endpoint.headers[index] != "Bearer "+productionAPIKey {
					t.Fatalf("request %d Authorization %q, want the configured key", index, endpoint.headers[index])
				}
				if endpoint.bodies[index]["model"] != productionModel {
					t.Fatalf("request %d model %v, want %q", index, endpoint.bodies[index]["model"], productionModel)
				}
				dimension, sent := endpoint.bodies[index]["dimensions"]
				if testCase.wantDimension == 0 && sent {
					t.Fatalf("request %d sent dimensions %v with no configured dimension", index, dimension)
				}
				if testCase.wantDimension != 0 && dimension != testCase.wantDimension {
					t.Fatalf("request %d dimensions %v, want %v", index, dimension, testCase.wantDimension)
				}
				if index == 0 {
					continue
				}
				gap := endpoint.arrivals[index].Sub(endpoint.arrivals[index-1])
				wantGap := wantBackoffBase << (index - 1)
				if gap < wantGap-backoffTolerance || gap > wantGap+time.Second {
					t.Fatalf("gap before attempt %d is %s, want about %s", index+1, gap, wantGap)
				}
			}
		})
	}
}

// TestNewRequiresAnAPIKeyForTheHostedProvider proves the daemon embedder still
// refuses a hosted configuration with no API key.
func TestNewRequiresAnAPIKeyForTheHostedProvider(t *testing.T) {
	server := httptest.NewServer(&embeddingEndpoint{})
	defer server.Close()
	productionShapedHome(t, server.URL+"/v1", "", nil)
	cfg, err := config.Default()
	if err != nil {
		t.Fatalf("config.Default: %v", err)
	}
	if _, err := providers.New(context.Background(), cfg); err == nil || !strings.Contains(err.Error(), "requires an API key") {
		t.Fatalf("providers.New without an API key returned %v, want the API key requirement", err)
	}
}
