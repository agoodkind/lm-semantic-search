// Package config resolves daemon runtime paths and settings.
package config

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

const (
	defaultStateDirName              = "lm-semantic-search"
	defaultSocketName                = "lm-semantic-search-daemon.sock"
	defaultLogFileName               = "lm-semantic-search-daemon.log"
	defaultSyncInterval              = 300000
	defaultSyncLockAge               = 600000
	defaultDebugListenAddr           = "127.0.0.1:6480"
	defaultPerfCountersIntervalMS    = 60000
	defaultMaxConcurrentIndexJobs    = 3
	defaultEmbeddingBatchTokenBudget = 6000
	defaultEmbeddingRequestTimeoutMS = 300000
	defaultMaxJobChunks              = 200000
	defaultMaxConversationsPerIngest = 100
	defaultMaxJobBytes               = 1073741824
	defaultExpectedJobGrowthFactor   = 4
	defaultExpectedJobGrowthFloor    = 10000
	// defaultLogRotationMaxBytes caps each active log file before gklog rotates
	// it on write. 5 MB keeps several backups inside the retention budget.
	defaultLogRotationMaxBytes = 5 * 1024 * 1024
	// defaultLogRetentionBytes caps the total bytes of rotated log backups the
	// retention sweep keeps. ~20 MB bounds the logs dir that grew to 1.1 GB.
	defaultLogRetentionBytes = 20 * 1024 * 1024
	// defaultLogCleanupIntervalMS sets the cadence of the background retention
	// sweep. Five minutes keeps the sweep off the hot path while staying timely.
	defaultLogCleanupIntervalMS = 300000
	nvEmbedCodeQueryPrefix      = "Instruct: Retrieve code or text relevant to the query.\nQuery: "
	// EmbedModelInputTokenLimit is the embedding model's hard per-input token
	// limit. The server rejects a longer single input with HTTP 400
	// context_length_exceeded ("maximum context length is 4096 tokens") and drops
	// that input, so splitting targets this limit to divide and index an oversized
	// input instead of losing it. It is always enforced, even when
	// embeddingMaxTokens is unset, so a missing or partial config cannot disable
	// the split and let the embedder drop content.
	EmbedModelInputTokenLimit = 4096
	// embedTokenSafetyMargin scales the token cap down before splitting, because a
	// byte-based budget cannot see the model's real tokenizer and dense or
	// non-Latin text packs more tokens per byte than the estimate assumes.
	embedTokenSafetyMargin = 0.9
	// conservativeEmbedBytesPerToken{Num,Den} express ~2.5 bytes per token, below
	// the densest measured content (~2.77 bytes/token). Converting a token cap into
	// a byte budget at this ratio overestimates tokens, so a byte-budgeted
	// sub-chunk stays under the model limit even for dense text the byte count
	// cannot weigh directly.
	conservativeEmbedBytesPerTokenNum = 5
	conservativeEmbedBytesPerTokenDen = 2
)

type embeddingProvider string

const embeddingProviderOpenAI embeddingProvider = "OpenAI"

// Config describes daemon runtime paths on the local machine.
type Config struct {
	ConfigRoot   string
	ConfigPath   string
	StateRoot    string
	SocketPath   string
	RegistryPath string
	JobsPath     string
	EventsPath   string
	LogsDir      string
	LogPath      string
	MerkleDir    string
	LocksDir     string
	SocketsDir   string
	ChunksDir    string
	GraphDir     string
	ContextRoot  string

	// Profile is the user-facing capability selector expanded by ApplyProfile.
	// Empty or "standard" keeps the Milvus plus OpenAI-compatible default; "offline"
	// selects the embedded local store and the in-process ONNX embedder.
	Profile string

	EmbeddingProvider string
	EmbeddingModel    string
	// OfflineEmbeddingModel selects a pinned ONNX model preset for the offline
	// profile. ApplyProfile derives EmbeddingModel and EmbeddingDimension from it.
	OfflineEmbeddingModel string
	EmbeddingBatchSize    int
	// EmbeddingBatchTokenBudget caps the estimated tokens (bytes/4) packed into
	// one embedding request. EmbeddingBatchSize stays as the row-count ceiling.
	EmbeddingBatchTokenBudget int
	// EmbeddingMaxTokens caps the tokens of a single chunk sent to the embedder,
	// so a chunk is split rather than dropped when it would exceed the model's
	// input limit. An unset value (0) does not disable the cap: the model's hard
	// input-token limit is always enforced by EffectiveEmbedTokenCapForLimit, and a
	// configured value only tightens the cap below that limit. Use
	// EffectiveEmbedTokenCapForLimit and EmbedChunkByteBudget to apply the safety
	// margin.
	EmbeddingMaxTokens int
	// EmbeddingRequestTimeoutMS bounds one embedding HTTP request. A wedged or
	// unresponsive embedder makes an unbounded request hang forever, which strands
	// the indexing goroutine and the background sync (the embed call has no other
	// deadline). Past this bound the request fails as unreachable so the job fails
	// and retries later instead of hanging. Zero disables the bound.
	EmbeddingRequestTimeoutMS int
	EmbeddingDimension        int32
	OpenAIAPIKey              string
	OpenAIBaseURL             string
	// QueryInstructionPrefix is prepended to query-time embedding text only.
	// Stored document vectors are embedded bare and stay valid.
	QueryInstructionPrefix string
	CustomIgnorePatterns   []string
	IncludeSubmodules      []string
	MilvusAddress          string
	MilvusToken            string
	// MilvusMutationCallTimeoutMS bounds one Milvus row-mutating call: Insert,
	// Upsert, Delete, Flush, FlushAll, Import, ReplicateMessage, and
	// TruncateCollection. The duration of those calls scales with the number of
	// rows they match, and a filter-based Delete matches an unbounded row count,
	// so no fixed value is provably sufficient for every collection. An operator
	// whose collection is large enough for a valid mutation to be cancelled at
	// the built-in bound raises this instead of rebuilding the daemon.
	//
	// Zero, a negative count, and a count above MaxMilvusMutationCallTimeoutMS
	// all keep the transport package's own five-minute bound, so no unset or
	// out-of-range value can leave a mutation unbounded. Convert it with
	// MilvusMutationCallTimeout rather than multiplying it directly.
	MilvusMutationCallTimeoutMS int
	// IndexBackend selects the vector store implementation: "milvus" (default) or
	// "local". Derived from Profile by ApplyProfile; may also be set directly.
	IndexBackend           string
	CollectionNameOverride string
	HybridMode             bool
	BackgroundSyncEnabled  bool
	SyncIntervalMS         int
	TriggerWatcherEnabled  bool
	FileWatcherEnabled     bool
	SyncLockStaleMS        int

	// DebugListenerEnabled controls whether the daemon starts a
	// loopback-only HTTP listener exposing pprof and expvar handlers for
	// live profiling and counter inspection.
	DebugListenerEnabled bool
	// DebugListenAddr is the loopback host:port the debug listener binds to.
	// It must stay on a loopback address so the profiling surface is never
	// reachable off-host.
	DebugListenAddr string
	// PerfCountersIntervalMS sets the cadence, in milliseconds, of the
	// periodic daemon.perf_counters slog line. A value of zero or below
	// disables the line entirely.
	PerfCountersIntervalMS int
	// MaxConcurrentIndexJobs caps how many index or converge jobs may run
	// their embedding pass simultaneously, bounding peak memory and load on
	// the embedding endpoint.
	MaxConcurrentIndexJobs int
	// MaxJobChunks caps the chunks one job may write before admission halts it.
	MaxJobChunks int32
	// MaxConversationsPerIngest caps the conversation ids one manifest sync may request.
	MaxConversationsPerIngest int
	// MaxJobBytes caps the chunk content bytes one job may write.
	MaxJobBytes int64
	// ExpectedJobGrowthFactor caps growth relative to the last successful run
	// or largest matching sibling worktree.
	ExpectedJobGrowthFactor float64
	// ExpectedJobGrowthFloor gives normal growth a fixed chunk allowance above
	// the expected baseline.
	ExpectedJobGrowthFloor int32
	// ResumeIndexingOnBoot controls whether daemon startup relaunches
	// codebases that were left mid-index when the daemon last stopped.
	ResumeIndexingOnBoot bool
	// LogRotationMaxBytes caps each active log file's size before gklog rotates
	// it on write, so the per-concern files and the combined service log stay
	// bounded. Converted to gklog's whole-megabyte rotation cap at install time.
	LogRotationMaxBytes int64
	// LogRetentionBytes caps the total bytes of rotated log backups the
	// background retention sweep keeps. Backups past this budget are deleted
	// oldest first, off the log-write path. Zero or below keeps everything.
	LogRetentionBytes int64
	// LogCleanupEnabled turns the background retention sweep on. When false the
	// sweep audits eligible backups but deletes nothing.
	LogCleanupEnabled bool
	// LogCleanupIntervalMS sets how often the background retention sweep runs a
	// pass after the immediate boot pass.
	LogCleanupIntervalMS int
}

type persistedConfig struct {
	Profile                   string `json:"profile"`
	EmbeddingProvider         string `json:"embeddingProvider"`
	EmbeddingModel            string `json:"embeddingModel"`
	OfflineEmbeddingModel     string `json:"offlineEmbeddingModel"`
	EmbeddingBatchSize        int    `json:"embeddingBatchSize"`
	EmbeddingBatchTokenBudget int    `json:"embeddingBatchTokenBudget"`
	EmbeddingMaxTokens        int    `json:"embeddingMaxTokens"`
	// EmbeddingRequestTimeoutMS is a pointer so an omitted config.json field (nil)
	// is distinct from an explicit 0, which disables the bound. A plain int would
	// collapse a persisted 0 into the default and make the disable case
	// unexpressible from config.json.
	EmbeddingRequestTimeoutMS *int   `json:"embeddingRequestTimeoutMs"`
	EmbeddingDimension        int32  `json:"embeddingDimension"`
	OpenAIAPIKey              string `json:"openaiApiKey"`
	OpenAIBaseURL             string `json:"openaiBaseUrl"`
	QueryInstructionPrefix    string `json:"queryInstructionPrefix"`
	MilvusAddress             string `json:"milvusAddress"`
	MilvusToken               string `json:"milvusToken"`
	// MilvusMutationCallTimeoutMS is a plain int because zero is not a distinct
	// setting here: it means "use the transport default", the same as an omitted
	// field, since a mutation must never run unbounded.
	MilvusMutationCallTimeoutMS int    `json:"milvusMutationCallTimeoutMs"`
	CollectionNameOverride      string `json:"collectionNameOverride"`
	HybridMode                  *bool  `json:"hybridMode"`
}

type embeddingConfigDefaults struct {
	provider             string
	model                string
	offlineModel         string
	queryInstructionText string
}

func resolveEmbeddingConfigDefaults(
	fileConfig persistedConfig,
) embeddingConfigDefaults {
	provider := envOrDefault(
		"EMBEDDING_PROVIDER",
		string(embeddingProviderOpenAI),
	)
	if provider == string(embeddingProviderOpenAI) &&
		fileConfig.EmbeddingProvider != "" {
		provider = fileConfig.EmbeddingProvider
	}

	model := fileConfig.EmbeddingModel
	if model == "" {
		model = envOrDefault("EMBEDDING_MODEL", "text-embedding-3-small")
	}
	offlineModel := fileConfig.OfflineEmbeddingModel
	if offlineModel == "" {
		offlineModel = offlinemodel.DefaultName
	}
	offlineModel = envOrDefault("OFFLINE_EMBEDDING_MODEL", offlineModel)

	queryInstructionText := fileConfig.QueryInstructionPrefix
	if queryInstructionText == "" && strings.Contains(model, "NV-EmbedCode") {
		queryInstructionText = nvEmbedCodeQueryPrefix
	}
	return embeddingConfigDefaults{
		provider:             provider,
		model:                model,
		offlineModel:         offlineModel,
		queryInstructionText: queryInstructionText,
	}
}

// Default returns the daemon configuration derived from the local environment.
func Default() (Config, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		slog.Error("resolve user home directory failed", "err", err)
		return Config{}, fmt.Errorf("resolve user home directory: %w", err)
	}

	loadContextEnvFile(filepath.Join(homeDir, ".context", ".env"))

	defaultConfigRoot := filepath.Join(resolveXDGConfigHome(homeDir), defaultStateDirName)
	defaultStateRoot := filepath.Join(resolveXDGStateHome(homeDir), defaultStateDirName)

	configRoot := envOrDefault("CLAUDE_CONTEXTD_CONFIG_ROOT", defaultConfigRoot)
	configPath := filepath.Join(configRoot, "config.json")

	stateRoot := defaultStateRoot
	stateRoot = envOrDefault("CLAUDE_CONTEXTD_STATE_ROOT", stateRoot)
	socketsDir := filepath.Join(stateRoot, "sockets")
	logsDir := filepath.Join(stateRoot, "logs")
	contextRoot := filepath.Join(homeDir, ".context")

	socketPath := envOrDefault("CLAUDE_CONTEXTD_SOCKET_PATH", filepath.Join(socketsDir, defaultSocketName))
	logPath := envOrDefault("CLAUDE_CONTEXTD_LOG_PATH", filepath.Join(logsDir, defaultLogFileName))

	fileConfig := readPersistedConfig(configPath)
	embeddingDefaults := resolveEmbeddingConfigDefaults(fileConfig)

	batchTokenBudget := intOrDefault(fileConfig.EmbeddingBatchTokenBudget, defaultEmbeddingBatchTokenBudget)
	embeddingMaxTokens := resolveEmbeddingMaxTokens(fileConfig.EmbeddingMaxTokens)
	// An explicit config.json value (including 0 to disable) wins over the
	// default; a nil pointer means the field was omitted. The env var overrides
	// either.
	requestTimeoutMS := defaultEmbeddingRequestTimeoutMS
	if fileConfig.EmbeddingRequestTimeoutMS != nil {
		requestTimeoutMS = *fileConfig.EmbeddingRequestTimeoutMS
	}
	return ApplyProfile(Config{
		Profile: resolveProfile(fileConfig.Profile), IndexBackend: IndexBackendMilvus,
		ConfigRoot:                  configRoot,
		ConfigPath:                  configPath,
		StateRoot:                   stateRoot,
		SocketPath:                  socketPath,
		RegistryPath:                filepath.Join(stateRoot, "registry.json"),
		JobsPath:                    filepath.Join(stateRoot, "jobs.jsonl"),
		EventsPath:                  filepath.Join(stateRoot, "events.jsonl"),
		LogsDir:                     logsDir,
		LogPath:                     logPath,
		MerkleDir:                   filepath.Join(stateRoot, "merkle"),
		LocksDir:                    filepath.Join(stateRoot, "locks"),
		SocketsDir:                  socketsDir,
		ChunksDir:                   filepath.Join(stateRoot, "chunks"),
		GraphDir:                    filepath.Join(stateRoot, "graph"),
		ContextRoot:                 contextRoot,
		EmbeddingProvider:           envOrDefault("EMBEDDING_PROVIDER", embeddingDefaults.provider),
		EmbeddingModel:              envOrDefault("EMBEDDING_MODEL", embeddingDefaults.model),
		OfflineEmbeddingModel:       embeddingDefaults.offlineModel,
		EmbeddingBatchSize:          envIntOrDefault("EMBEDDING_BATCH_SIZE", intOrDefault(fileConfig.EmbeddingBatchSize, 32)),
		EmbeddingBatchTokenBudget:   batchTokenBudget,
		EmbeddingMaxTokens:          embeddingMaxTokens,
		EmbeddingRequestTimeoutMS:   envIntOrDefault("CLAUDE_CONTEXT_EMBEDDING_REQUEST_TIMEOUT_MS", requestTimeoutMS),
		EmbeddingDimension:          envInt32OrDefault("EMBEDDING_DIMENSION", fileConfig.EmbeddingDimension),
		OpenAIAPIKey:                envOrDefault("OPENAI_API_KEY", fileConfig.OpenAIAPIKey),
		OpenAIBaseURL:               envOrDefault("OPENAI_BASE_URL", fileConfig.OpenAIBaseURL),
		QueryInstructionPrefix:      embeddingDefaults.queryInstructionText,
		CustomIgnorePatterns:        parseCommaSeparated(os.Getenv("CUSTOM_IGNORE_PATTERNS")),
		IncludeSubmodules:           parseCommaSeparated(os.Getenv("CLAUDE_CONTEXT_INCLUDE_SUBMODULES")),
		MilvusAddress:               envOrDefault("MILVUS_ADDRESS", fileConfig.MilvusAddress),
		MilvusToken:                 envOrDefault("MILVUS_TOKEN", fileConfig.MilvusToken),
		MilvusMutationCallTimeoutMS: resolveMilvusMutationCallTimeoutMS(fileConfig.MilvusMutationCallTimeoutMS),
		CollectionNameOverride:      envOrDefault("CODE_CHUNKS_COLLECTION_NAME_OVERRIDE", fileConfig.CollectionNameOverride),
		HybridMode:                  envBoolOrDefault("HYBRID_MODE", boolOrDefault(fileConfig.HybridMode, true)),
		BackgroundSyncEnabled:       envBoolOrDefault("CLAUDE_CONTEXT_BACKGROUND_SYNC", true),
		SyncIntervalMS:              envIntOrDefault("CLAUDE_CONTEXT_SYNC_INTERVAL_MS", defaultSyncInterval),
		TriggerWatcherEnabled:       envBoolOrDefault("CLAUDE_CONTEXT_TRIGGER_WATCHER", true),
		FileWatcherEnabled:          envBoolOrDefault("CLAUDE_CONTEXT_FILE_WATCHER", true),
		SyncLockStaleMS:             envIntOrDefault("CLAUDE_CONTEXT_SYNC_LOCK_STALE_MS", defaultSyncLockAge),
		DebugListenerEnabled:        envBoolOrDefault("CLAUDE_CONTEXT_DEBUG_LISTENER", true),
		DebugListenAddr:             envOrDefault("CLAUDE_CONTEXT_DEBUG_LISTEN_ADDR", defaultDebugListenAddr),
		PerfCountersIntervalMS:      envIntOrDefault("CLAUDE_CONTEXT_PERF_COUNTERS_INTERVAL_MS", defaultPerfCountersIntervalMS),
		MaxConcurrentIndexJobs:      envIntOrDefault("CLAUDE_CONTEXT_MAX_CONCURRENT_INDEX_JOBS", defaultMaxConcurrentIndexJobs),
		MaxJobChunks:                envInt32OrDefault("CLAUDE_CONTEXT_MAX_JOB_CHUNKS", defaultMaxJobChunks),
		MaxConversationsPerIngest:   envIntOrDefault("CLAUDE_CONTEXT_MAX_CONVERSATIONS_PER_INGEST", defaultMaxConversationsPerIngest),
		MaxJobBytes:                 envInt64OrDefault("CLAUDE_CONTEXT_MAX_JOB_BYTES", defaultMaxJobBytes),
		ExpectedJobGrowthFactor:     envFloat64OrDefault("CLAUDE_CONTEXT_EXPECTED_JOB_GROWTH_FACTOR", defaultExpectedJobGrowthFactor),
		ExpectedJobGrowthFloor:      envInt32OrDefault("CLAUDE_CONTEXT_EXPECTED_JOB_GROWTH_FLOOR", defaultExpectedJobGrowthFloor),
		ResumeIndexingOnBoot:        envBoolOrDefault("CLAUDE_CONTEXT_RESUME_ON_BOOT", true),
		LogRotationMaxBytes:         envInt64OrDefault("CLAUDE_CONTEXT_LOG_ROTATION_MAX_BYTES", defaultLogRotationMaxBytes),
		LogRetentionBytes:           envInt64OrDefault("CLAUDE_CONTEXT_LOG_RETENTION_BYTES", defaultLogRetentionBytes),
		LogCleanupEnabled:           envBoolOrDefault("CLAUDE_CONTEXT_LOG_CLEANUP_ENABLED", true),
		LogCleanupIntervalMS:        envIntOrDefault("CLAUDE_CONTEXT_LOG_CLEANUP_INTERVAL_MS", defaultLogCleanupIntervalMS),
	}), nil
}

func resolveXDGConfigHome(homeDir string) string {
	return envOrDefault("XDG_CONFIG_HOME", filepath.Join(homeDir, ".config"))
}

func resolveXDGStateHome(homeDir string) string {
	return envOrDefault("XDG_STATE_HOME", filepath.Join(homeDir, ".local", "state"))
}

func envOrDefault(name string, fallback string) string {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	return value
}

func envIntOrDefault(name string, fallback int) int {
	rawValue := os.Getenv(name)
	if rawValue == "" {
		return fallback
	}

	parsedValue, err := strconv.Atoi(rawValue)
	if err != nil {
		return fallback
	}
	return parsedValue
}

func envInt32OrDefault(name string, fallback int32) int32 {
	rawValue := os.Getenv(name)
	if rawValue == "" {
		return fallback
	}

	parsedValue, err := strconv.ParseInt(rawValue, 10, 32)
	if err != nil {
		return fallback
	}
	return int32(parsedValue)
}

func envInt64OrDefault(name string, fallback int64) int64 {
	rawValue := os.Getenv(name)
	if rawValue == "" {
		return fallback
	}

	parsedValue, err := strconv.ParseInt(rawValue, 10, 64)
	if err != nil {
		return fallback
	}
	return parsedValue
}

func envFloat64OrDefault(name string, fallback float64) float64 {
	rawValue := os.Getenv(name)
	if rawValue == "" {
		return fallback
	}

	parsedValue, err := strconv.ParseFloat(rawValue, 64)
	if err != nil {
		return fallback
	}
	return parsedValue
}

func envBoolOrDefault(name string, fallback bool) bool {
	rawValue := os.Getenv(name)
	if rawValue == "" {
		return fallback
	}

	parsedValue, err := strconv.ParseBool(rawValue)
	if err != nil {
		return fallback
	}
	return parsedValue
}

func readPersistedConfig(path string) persistedConfig {
	data, err := os.ReadFile(path)
	if err != nil {
		var cfg persistedConfig
		return cfg
	}

	var cfg persistedConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		slog.Error("read persisted daemon config failed", "path", path, "err", err)
		var emptyConfig persistedConfig
		return emptyConfig
	}
	return cfg
}

// resolveEmbeddingMaxTokens applies the env override over the config value. An
// unset or non-positive value resolves to 0, which falls back to the model's
// hard input-token limit in EffectiveEmbedTokenCapForLimit rather than disabling
// the split; a warning names the knob to set for a tighter, batching-friendly cap.
func resolveEmbeddingMaxTokens(fileValue int) int {
	value := envIntOrDefault("EMBEDDING_MAX_TOKENS", fileValue)
	if value <= 0 {
		slog.Warn(
			"embeddingMaxTokens is unset; splitting falls back to the model's hard input-token limit, set the knob for a tighter per-chunk cap",
			"config_field", "embeddingMaxTokens",
			"env_var", "EMBEDDING_MAX_TOKENS",
			"model_token_limit", EmbedModelInputTokenLimit,
		)
		return 0
	}
	return value
}

// MaxMilvusMutationCallTimeoutMS is the largest millisecond count that converts
// to a [time.Duration] without overflowing its int64 nanosecond range. Anything
// larger wraps on multiplication, so it is rejected rather than converted.
const MaxMilvusMutationCallTimeoutMS = int64(math.MaxInt64) / int64(time.Millisecond)

// MilvusMutationCallTimeout converts a configured millisecond count into the
// Milvus mutation bound. Zero, a negative count, and a count above
// MaxMilvusMutationCallTimeoutMS all return zero, which the transport reads as
// "keep the built-in bound".
//
// The range check is what makes the conversion total. Multiplying an
// out-of-range count instead wraps it: -9223372036855 milliseconds becomes a
// positive duration of roughly 292 years, which removes the bound the policy
// exists to enforce, and a count just above the maximum wraps to a tiny or
// negative duration that fails every mutation immediately. Both are silent, so
// the count is validated before it is multiplied rather than after.
func MilvusMutationCallTimeout(milliseconds int) time.Duration {
	count := int64(milliseconds)
	if count <= 0 || count > MaxMilvusMutationCallTimeoutMS {
		return 0
	}
	return time.Duration(count) * time.Millisecond
}

// resolveMilvusMutationCallTimeoutMS applies the env override over the config
// value and rejects anything MilvusMutationCallTimeout cannot convert, so the
// resolved field only ever holds zero or a usable count. A rejected value warns
// and names the knob, because an operator who mistypes the count would otherwise
// see the built-in bound silently stay in place.
func resolveMilvusMutationCallTimeoutMS(fileValue int) int {
	value := envIntOrDefault("CLAUDE_CONTEXT_MILVUS_MUTATION_CALL_TIMEOUT_MS", fileValue)
	if value == 0 {
		return 0
	}
	if MilvusMutationCallTimeout(value) == 0 {
		slog.Warn(
			"milvusMutationCallTimeoutMs is not a usable millisecond count; keeping the built-in Milvus mutation bound",
			"value", value,
			"max", MaxMilvusMutationCallTimeoutMS,
			"config_field", "milvusMutationCallTimeoutMs",
			"env_var", "CLAUDE_CONTEXT_MILVUS_MUTATION_CALL_TIMEOUT_MS",
		)
		return 0
	}
	return value
}

// EffectiveEmbedTokenCapForLimit returns the per-chunk token cap after the safety
// margin against modelLimit, the active model's hard input-token limit. The model
// limit is always enforced, so an unset or larger maxTokens still caps at the
// model limit rather than disabling the split and letting the embedder drop or
// truncate an oversized input; a configured value below the model limit tightens
// the cap further. A non-positive modelLimit falls back to the OpenAI-compatible
// limit. The result is always at least one and always below modelLimit, so the
// local ONNX backend can pass its 2048/512 preset limit and the Milvus backend
// its 4096 OpenAI-compatible limit through one path.
func EffectiveEmbedTokenCapForLimit(maxTokens int, modelLimit int) int {
	if modelLimit <= 0 {
		modelLimit = EmbedModelInputTokenLimit
	}
	modelCap := max(int(float64(modelLimit)*embedTokenSafetyMargin), 1)
	if maxTokens <= 0 {
		return modelCap
	}
	configuredCap := int(float64(maxTokens) * embedTokenSafetyMargin)
	return max(min(configuredCap, modelCap), 1)
}

// EmbedChunkByteBudget returns the byte budget a byte-oriented splitter uses to
// keep a sub-chunk within EffectiveEmbedTokenCap real tokens against the
// OpenAI-compatible model limit. It is EmbedChunkByteBudgetForLimit specialized
// to EmbedModelInputTokenLimit.
func EmbedChunkByteBudget(maxTokens int) int {
	return EmbedChunkByteBudgetForLimit(maxTokens, EmbedModelInputTokenLimit)
}

// EmbedChunkByteBudgetForLimit returns the byte budget that keeps a sub-chunk
// within EffectiveEmbedTokenCapForLimit real tokens against modelLimit. It
// converts the token cap at a conservative bytes-per-token ratio below the
// densest measured content, so dense text stays under the model limit even though
// the byte count cannot see the real tokenizer. It is always positive because the
// model limit is always enforced.
func EmbedChunkByteBudgetForLimit(maxTokens int, modelLimit int) int {
	tokenCap := EffectiveEmbedTokenCapForLimit(maxTokens, modelLimit)
	return tokenCap * conservativeEmbedBytesPerTokenNum / conservativeEmbedBytesPerTokenDen
}

// ActiveEmbedTokenLimit reports the embedding model's hard per-input token limit
// for the active provider: the offline ONNX preset's maximum for the ONNX
// provider, otherwise the OpenAI-compatible model input limit. The split path
// derives its byte budget from this so the local backend splits at the preset's
// real 2048 (embeddinggemma) or 512 (bge-small) limit instead of the 4096
// OpenAI-compatible limit that would let the ONNX tokenizer truncate the input.
func ActiveEmbedTokenLimit(cfg Config) int {
	if strings.EqualFold(strings.TrimSpace(cfg.EmbeddingProvider), EmbeddingProviderONNX) {
		preset, err := offlinemodel.Resolve(cfg.OfflineEmbeddingModel)
		if err == nil && preset.MaximumTokens > 0 {
			return int(preset.MaximumTokens)
		}
	}
	return EmbedModelInputTokenLimit
}

func intOrDefault(value int, fallback int) int {
	if value == 0 {
		return fallback
	}
	return value
}

func stringOrDefault(value string, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}

func resolveProfile(persistedProfile string) string {
	resolvedProfile := envOrDefault(
		"CLAUDE_CONTEXT_PROFILE",
		stringOrDefault(persistedProfile, ProfileStandard),
	)
	normalizedProfile := strings.ToLower(strings.TrimSpace(resolvedProfile))
	return stringOrDefault(normalizedProfile, ProfileStandard)
}

func boolOrDefault(value *bool, fallback bool) bool {
	if value == nil {
		return fallback
	}
	return *value
}

// loadContextEnvFile loads KEY=VALUE pairs from ~/.context/.env (or any path
// supplied by the caller). It only sets keys that are not already present in
// the process environment so explicit env-var overrides win. Lines starting
// with '#' and blank lines are ignored.
func loadContextEnvFile(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	for line := range strings.SplitSeq(string(data), "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		equalsIndex := strings.IndexByte(trimmed, '=')
		if equalsIndex <= 0 {
			continue
		}
		key := strings.TrimSpace(trimmed[:equalsIndex])
		value := strings.TrimSpace(trimmed[equalsIndex+1:])
		if key == "" {
			continue
		}
		if (strings.HasPrefix(value, `"`) && strings.HasSuffix(value, `"`)) ||
			(strings.HasPrefix(value, `'`) && strings.HasSuffix(value, `'`)) {
			value = value[1 : len(value)-1]
		}
		if _, alreadySet := os.LookupEnv(key); alreadySet {
			continue
		}
		if err := os.Setenv(key, value); err != nil {
			slog.Error("set env from .context/.env failed", "key", key, "err", err)
		}
	}
}

// parseCommaSeparated returns a trimmed, non-empty list from a comma-separated
// string. Returns nil for empty input so the field cleanly distinguishes
// "unset" from "explicit empty list".
func parseCommaSeparated(value string) []string {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil
	}
	parts := strings.Split(value, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		trimmed := strings.TrimSpace(part)
		if trimmed == "" {
			continue
		}
		out = append(out, trimmed)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
