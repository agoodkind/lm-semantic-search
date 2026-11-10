package onnx

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"path/filepath"

	"goodkind.io/lm-semantic-search/embedding"
	"goodkind.io/lm-semantic-search/internal/modeldownload"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

type presetArtifact struct {
	path   string
	sha256 string
}

func presetArtifacts(
	cacheRoot string,
	preset offlinemodel.Preset,
) ([]presetArtifact, cachedModelFiles, error) {
	if cacheRoot == "" {
		return nil, cachedModelFiles{}, fmt.Errorf(
			"offline embedding model %q requires a model cache root",
			preset.Name,
		)
	}
	modelDirectory := filepath.Join(cacheRoot, offlineModelCacheDirectory, preset.Name)
	sources := []struct {
		rawURL string
		sha256 string
	}{
		{rawURL: preset.ModelONNXURL, sha256: preset.ModelSHA256},
		{rawURL: preset.ModelDataURL, sha256: preset.ModelDataSHA256},
		{rawURL: preset.TokenizerURL, sha256: preset.TokenizerSHA256},
	}
	artifacts := make([]presetArtifact, 0, len(sources))
	files := cachedModelFiles{modelPath: "", tokenizerPath: ""}
	for _, source := range sources {
		if source.rawURL == "" {
			continue
		}
		filename, err := artifactFilename(source.rawURL)
		if err != nil {
			return nil, cachedModelFiles{}, err
		}
		artifactPath := filepath.Join(modelDirectory, filename)
		artifacts = append(artifacts, presetArtifact{path: artifactPath, sha256: source.sha256})
		switch source.rawURL {
		case preset.ModelONNXURL:
			files.modelPath = artifactPath
		case preset.TokenizerURL:
			files.tokenizerPath = artifactPath
		}
	}
	return artifacts, files, nil
}

// PresetFilesInstalled checks every preset artifact against its pinned
// SHA-256 without sending a network request.
func PresetFilesInstalled(
	ctx context.Context,
	cacheRoot string,
	preset offlinemodel.Preset,
) (bool, error) {
	artifacts, _, err := presetArtifacts(cacheRoot, preset)
	if err != nil {
		return false, err
	}
	for _, artifact := range artifacts {
		installed, installedErr := modeldownload.Installed(ctx, artifact.path, artifact.sha256)
		if installedErr != nil {
			slog.ErrorContext(ctx, "onnx.preset_artifact.verify_failed", "path", artifact.path, "err", installedErr)
			return false, fmt.Errorf("read or hash preset artifact %s: %w", artifact.path, installedErr)
		}
		if !installed {
			return false, nil
		}
	}
	return true, nil
}

// InstallPresetFilesWithProgress downloads missing artifacts or artifacts
// with a different checksum. A nil progress function disables reports.
// A nil sleep function selects a timer.
func InstallPresetFilesWithProgress(
	ctx context.Context,
	httpClient *http.Client,
	cacheRoot string,
	preset offlinemodel.Preset,
	progress modeldownload.ProgressFunc,
	sleep modeldownload.SleepFunc,
) error {
	_, err := ensureModelFiles(ctx, httpClient, cacheRoot, preset, progress, sleep)
	return err
}

// NewProviderForInstalledPreset loads installed files without sending a
// network request.
func NewProviderForInstalledPreset(
	cacheRoot string,
	preset offlinemodel.Preset,
) (embedding.Provider, error) {
	_, files, err := presetArtifacts(cacheRoot, preset)
	if err != nil {
		return nil, err
	}
	return providerForFiles(files, preset)
}
