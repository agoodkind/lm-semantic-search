// Package local runs an ONNX embedding model in the calling process.
package local

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"runtime"
	"strings"

	"goodkind.io/lm-semantic-search/embedding"
	"goodkind.io/lm-semantic-search/internal/embedding/onnx"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
	"goodkind.io/lm-semantic-search/internal/onnxruntimedist"
)

// ErrModelUnavailable reports that a model file is absent from the cache root
// and its download failed.
var ErrModelUnavailable = onnx.ErrArtifactUnavailable

// ErrRuntimeUnavailable reports that New opened no ONNX Runtime library from
// the executable's directory or runpath. The error names each path it tried.
var ErrRuntimeUnavailable = onnx.ErrRuntimeLibraryUnavailable

const runtimeDirectoryMode = 0o755

// RuntimeVersion is the ONNX Runtime release that InstallRuntime installs.
const RuntimeVersion = onnxruntimedist.Version

// Runtime is the ONNX Runtime library file the process opened and the version
// that file reports.
type Runtime struct {
	Path    string
	Version string
}

// LoadRuntime opens the ONNX Runtime library from the same paths as New. A
// process that opened one library file keeps it until the process exits.
func LoadRuntime() (Runtime, error) {
	library, err := onnx.LoadRuntimeLibrary()
	if err != nil {
		slog.Warn("ONNX Runtime library is unavailable", "pinned_version", RuntimeVersion)
		return Runtime{Path: "", Version: ""}, fmt.Errorf("load ONNX Runtime: %w", err)
	}
	return Runtime{Path: library.Path, Version: library.Version}, nil
}

// ModelInstalled reports whether every file of a model exists under cacheRoot.
// It downloads nothing and does not verify checksums.
func ModelInstalled(cacheRoot string, name string) (bool, error) {
	present, err := onnx.ModelFilesPresent(cacheRoot, name)
	if err != nil {
		slog.Warn("check local embedding model failed", "model", name, "err", err)
		return false, fmt.Errorf("check local embedding model %q: %w", name, err)
	}
	return present, nil
}

// InstallModel downloads every missing or checksum-mismatched file of a model
// into cacheRoot through httpClient.
func InstallModel(ctx context.Context, httpClient *http.Client, cacheRoot string, name string) error {
	if err := onnx.InstallModelFiles(ctx, httpClient, cacheRoot, name); err != nil {
		slog.ErrorContext(ctx, "install local embedding model failed", "model", name, "err", err)
		return fmt.Errorf("install local embedding model %q: %w", name, err)
	}
	return nil
}

// InstallRuntime downloads the pinned ONNX Runtime release for the running
// platform, verifies its SHA-256, and writes the shared library and its SONAME
// and unversioned symlinks into directory. httpClient downloads the archive.
// New opens the library from the directory of the running executable. No other
// function in this module calls InstallRuntime.
func InstallRuntime(ctx context.Context, httpClient *http.Client, directory string) error {
	if strings.TrimSpace(directory) == "" {
		return errors.New("install ONNX Runtime: directory is required")
	}
	archive, err := onnxruntimedist.ArchiveFor(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		slog.ErrorContext(ctx, "resolve ONNX Runtime archive failed", "err", err)
		return fmt.Errorf("install ONNX Runtime: %w", err)
	}
	names, err := onnxruntimedist.LibraryNamesFor(runtime.GOOS)
	if err != nil {
		slog.ErrorContext(ctx, "resolve ONNX Runtime library names failed", "err", err)
		return fmt.Errorf("install ONNX Runtime: %w", err)
	}
	workDirectory, err := os.MkdirTemp("", "lms-onnxruntime-install.")
	if err != nil {
		slog.ErrorContext(ctx, "create ONNX Runtime download directory failed", "err", err)
		return fmt.Errorf("install ONNX Runtime: %w", err)
	}
	defer func() { _ = os.RemoveAll(workDirectory) }()
	archiveDirectory, err := onnxruntimedist.FetchArchive(ctx, httpClient, archive, workDirectory)
	if err != nil {
		return fmt.Errorf("install ONNX Runtime: %w", err)
	}
	if err := os.MkdirAll(directory, runtimeDirectoryMode); err != nil {
		slog.ErrorContext(ctx, "create ONNX Runtime directory failed", "directory", directory, "err", err)
		return fmt.Errorf("install ONNX Runtime into %s: %w", directory, err)
	}
	if err := onnxruntimedist.InstallLibrary(archiveDirectory, names, directory); err != nil {
		return fmt.Errorf("install ONNX Runtime into %s: %w", directory, err)
	}
	return nil
}

// Options requires CacheRoot and uses DefaultModel when Model is empty.
type Options struct {
	Model     string
	CacheRoot string
}

// Model is the fixed description of one supported model.
type Model struct {
	Name string
	// Dimension is the width of every vector the model returns.
	Dimension int
	// QueryPrefix is the text a caller prepends to a search query before
	// embedding it.
	QueryPrefix string
	// MaximumTokens is the longest input the model embeds. The provider refuses
	// a longer input.
	MaximumTokens int
}

// DefaultModel is the model an empty [Options.Model] selects.
const DefaultModel = offlinemodel.DefaultName

// Models returns the supported model names.
func Models() []string {
	return offlinemodel.Names()
}

// Describe returns the description of a supported model. An empty name selects
// [DefaultModel].
func Describe(name string) (Model, error) {
	preset, err := offlinemodel.Resolve(name)
	if err != nil {
		slog.Error("describe local embedding model failed", "model", name, "err", err)
		return Model{}, fmt.Errorf("describe local embedding model: %w", err)
	}
	return Model{
		Name:          preset.Name,
		Dimension:     int(preset.Dimension),
		QueryPrefix:   preset.QueryPrefix,
		MaximumTokens: int(preset.MaximumTokens),
	}, nil
}

// New returns the in-process provider for a model. Every returned vector has
// unit length. Providers for the same model and cache root share one loaded
// model for the life of the process.
func New(ctx context.Context, options Options) (embedding.Provider, error) {
	if strings.TrimSpace(options.CacheRoot) == "" {
		err := errors.New("local embedding provider requires a cache root")
		slog.ErrorContext(ctx, "create local embedding provider failed", "model", options.Model, "err", err)
		return nil, err
	}
	provider, err := onnx.NewProviderForModel(ctx, options.Model, options.CacheRoot)
	if err != nil {
		slog.ErrorContext(ctx, "create local embedding provider failed", "model", options.Model, "err", err)
		return nil, fmt.Errorf("create local embedding provider: %w", err)
	}
	return provider, nil
}
