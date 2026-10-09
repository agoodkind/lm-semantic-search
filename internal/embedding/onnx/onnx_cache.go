package onnx

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"

	"goodkind.io/lm-semantic-search/internal/modeldownload"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

const (
	offlineModelCacheDirectory = "embedding-models"
	tokenizerArtifactKind      = "tokenizer"
)

// ErrArtifactUnavailable marks a failure to fetch an offline model artifact
// from its remote host, either a transport error or a non-success HTTP status.
// The daemon must not assume network access, so callers can degrade gracefully
// on this condition, and tests that need a downloaded artifact skip on it. It is
// exported so a test outside this package can tell a missing download apart from
// a real provider failure.
var ErrArtifactUnavailable = modeldownload.ErrUnavailable

type cachedModelFiles struct {
	modelPath     string
	tokenizerPath string
}

// ensureModelFiles returns the local paths of one preset's model and tokenizer,
// downloading and checksum-verifying whatever is missing. cacheRoot is the
// machine-wide artifact cache rather than one daemon's state root, so a
// throwaway daemon reuses an already-downloaded model instead of fetching it
// again.
func ensureModelFiles(
	ctx context.Context,
	httpClient *http.Client,
	cacheRoot string,
	preset offlinemodel.Preset,
	progress modeldownload.ProgressFunc,
	sleep modeldownload.SleepFunc,
) (cachedModelFiles, error) {
	if cacheRoot == "" {
		return cachedModelFiles{}, fmt.Errorf(
			"offline embedding model %q requires a model cache root",
			preset.Name,
		)
	}
	modelDirectory := filepath.Join(
		cacheRoot,
		offlineModelCacheDirectory,
		preset.Name,
	)
	modelFilename, err := artifactFilename(preset.ModelONNXURL)
	if err != nil {
		return cachedModelFiles{}, err
	}
	modelPath := filepath.Join(modelDirectory, modelFilename)
	if err := ensureArtifact(
		ctx,
		httpClient,
		preset.Name,
		"model",
		preset.ModelONNXURL,
		preset.ModelSHA256,
		modelPath,
		progress,
		sleep,
	); err != nil {
		return cachedModelFiles{}, err
	}

	if preset.ModelDataURL != "" {
		modelDataFilename, filenameErr := artifactFilename(preset.ModelDataURL)
		if filenameErr != nil {
			return cachedModelFiles{}, filenameErr
		}
		if err := ensureArtifact(
			ctx,
			httpClient,
			preset.Name,
			"model data",
			preset.ModelDataURL,
			preset.ModelDataSHA256,
			filepath.Join(modelDirectory, modelDataFilename),
			progress,
			sleep,
		); err != nil {
			return cachedModelFiles{}, err
		}
	}

	tokenizerFilename, err := artifactFilename(preset.TokenizerURL)
	if err != nil {
		return cachedModelFiles{}, err
	}
	tokenizerPath := filepath.Join(modelDirectory, tokenizerFilename)
	if err := ensureArtifact(
		ctx,
		httpClient,
		preset.Name,
		tokenizerArtifactKind,
		preset.TokenizerURL,
		preset.TokenizerSHA256,
		tokenizerPath,
		progress,
		sleep,
	); err != nil {
		return cachedModelFiles{}, err
	}
	return cachedModelFiles{
		modelPath:     modelPath,
		tokenizerPath: tokenizerPath,
	}, nil
}

// ModelFilesPresent reports whether every artifact file of the preset exists in
// cacheRoot. It reads no file contents and downloads nothing.
// NewProviderForModel verifies each checksum.
func ModelFilesPresent(cacheRoot string, modelName string) (bool, error) {
	preset, err := offlinemodel.Resolve(modelName)
	if err != nil {
		slog.Error("resolve offline embedding model failed", "model", modelName, "err", err)
		return false, fmt.Errorf("resolve offline embedding model: %w", err)
	}
	modelDirectory := filepath.Join(cacheRoot, offlineModelCacheDirectory, preset.Name)
	for _, rawURL := range []string{preset.ModelONNXURL, preset.ModelDataURL, preset.TokenizerURL} {
		if rawURL == "" {
			continue
		}
		filename, filenameErr := artifactFilename(rawURL)
		if filenameErr != nil {
			return false, filenameErr
		}
		info, statErr := os.Stat(filepath.Join(modelDirectory, filename))
		if errors.Is(statErr, os.ErrNotExist) {
			return false, nil
		}
		if statErr != nil {
			slog.Error("inspect offline embedding artifact failed", "path", filepath.Join(modelDirectory, filename), "err", statErr)
			return false, fmt.Errorf("inspect offline embedding artifact %s: %w", filename, statErr)
		}
		if !info.Mode().IsRegular() {
			return false, nil
		}
	}
	return true, nil
}

// InstallModelFiles downloads and checksum-verifies every missing or
// mismatched artifact of the preset into cacheRoot through httpClient.
func InstallModelFiles(ctx context.Context, httpClient *http.Client, cacheRoot string, modelName string) error {
	return InstallModelFilesWithProgress(ctx, httpClient, cacheRoot, modelName, nil)
}

// InstallModelFilesWithProgress installs model files with download progress.
// A nil progress function disables reports.
func InstallModelFilesWithProgress(
	ctx context.Context,
	httpClient *http.Client,
	cacheRoot string,
	modelName string,
	progress modeldownload.ProgressFunc,
) error {
	preset, err := offlinemodel.Resolve(modelName)
	if err != nil {
		slog.ErrorContext(ctx, "resolve offline embedding model failed", "model", modelName, "err", err)
		return fmt.Errorf("resolve offline embedding model: %w", err)
	}
	_, err = ensureModelFiles(ctx, httpClient, cacheRoot, preset, progress, nil)
	return err
}

func artifactFilename(rawURL string) (string, error) {
	parsedURL, err := url.Parse(rawURL)
	if err != nil {
		slog.Error("parse offline model URL failed", "url", rawURL, "err", err)
		return "", fmt.Errorf("parse offline model URL %q: %w", rawURL, err)
	}
	filename := path.Base(parsedURL.Path)
	if filename == "." || filename == "/" || filename == "" {
		return "", fmt.Errorf("offline model URL %q has no filename", rawURL)
	}
	return filename, nil
}

func ensureArtifact(
	ctx context.Context,
	httpClient *http.Client,
	modelName string,
	artifactKind string,
	rawURL string,
	expectedSHA256 string,
	destinationPath string,
	progress modeldownload.ProgressFunc,
	sleep modeldownload.SleepFunc,
) error {
	request := modeldownload.Request{
		HTTPClient:      httpClient,
		URL:             rawURL,
		SHA256:          expectedSHA256,
		DestinationPath: destinationPath,
		Progress:        progress,
		Sleep:           sleep,
	}
	if err := modeldownload.Ensure(ctx, request); err != nil {
		slog.ErrorContext(
			ctx,
			"cache offline embedding artifact failed",
			"model",
			modelName,
			"artifact",
			artifactKind,
			"err",
			err,
		)
		return fmt.Errorf(
			"cache offline embedding %s %q: %w",
			artifactKind,
			modelName,
			err,
		)
	}
	return nil
}
