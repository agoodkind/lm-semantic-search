package embedding

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"

	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

const (
	offlineModelCacheDirectory = "embedding-models"
	offlineModelDirectoryMode  = 0o700
	offlineModelFileMode       = 0o644
)

type cachedModelFiles struct {
	modelPath     string
	tokenizerPath string
}

func ensureModelFiles(
	ctx context.Context,
	httpClient *http.Client,
	stateRoot string,
	preset offlinemodel.Preset,
) (cachedModelFiles, error) {
	if stateRoot == "" {
		return cachedModelFiles{}, fmt.Errorf(
			"offline embedding model %q requires a state root",
			preset.Name,
		)
	}
	modelDirectory := filepath.Join(
		stateRoot,
		offlineModelCacheDirectory,
		preset.Name,
	)
	modelFilename, err := artifactFilename(preset.ModelONNXURL)
	if err != nil {
		return cachedModelFiles{}, err
	}
	modelPath := filepath.Join(modelDirectory, modelFilename)
	if err := ensurePresetArtifact(
		ctx,
		httpClient,
		preset.Name,
		"model",
		preset.ModelONNXURL,
		preset.ModelSHA256,
		modelPath,
	); err != nil {
		return cachedModelFiles{}, err
	}

	if preset.ModelDataURL != "" {
		modelDataFilename, filenameErr := artifactFilename(preset.ModelDataURL)
		if filenameErr != nil {
			return cachedModelFiles{}, filenameErr
		}
		if err := ensurePresetArtifact(
			ctx,
			httpClient,
			preset.Name,
			"model data",
			preset.ModelDataURL,
			preset.ModelDataSHA256,
			filepath.Join(modelDirectory, modelDataFilename),
		); err != nil {
			return cachedModelFiles{}, err
		}
	}

	tokenizerFilename, err := artifactFilename(preset.TokenizerURL)
	if err != nil {
		return cachedModelFiles{}, err
	}
	tokenizerPath := filepath.Join(modelDirectory, tokenizerFilename)
	if err := ensurePresetArtifact(
		ctx,
		httpClient,
		preset.Name,
		"tokenizer",
		preset.TokenizerURL,
		preset.TokenizerSHA256,
		tokenizerPath,
	); err != nil {
		return cachedModelFiles{}, err
	}
	return cachedModelFiles{
		modelPath:     modelPath,
		tokenizerPath: tokenizerPath,
	}, nil
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

func ensurePresetArtifact(
	ctx context.Context,
	httpClient *http.Client,
	modelName string,
	artifactKind string,
	rawURL string,
	expectedSHA256 string,
	destinationPath string,
) error {
	if err := ensureArtifact(
		ctx,
		httpClient,
		rawURL,
		expectedSHA256,
		destinationPath,
	); err != nil {
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

func ensureArtifact(
	ctx context.Context,
	httpClient *http.Client,
	rawURL string,
	expectedSHA256 string,
	destinationPath string,
) error {
	matches, err := artifactMatches(destinationPath, expectedSHA256)
	if err != nil {
		return err
	}
	if matches {
		return nil
	}

	if _, statErr := os.Stat(destinationPath); statErr == nil {
		slog.WarnContext(
			ctx,
			"offline embedding artifact checksum mismatch; downloading replacement",
			"path",
			destinationPath,
		)
	}
	if err := os.MkdirAll(
		filepath.Dir(destinationPath),
		offlineModelDirectoryMode,
	); err != nil {
		slog.ErrorContext(
			ctx,
			"create offline embedding cache directory failed",
			"path",
			filepath.Dir(destinationPath),
			"err",
			err,
		)
		return fmt.Errorf(
			"create offline embedding cache directory %s: %w",
			filepath.Dir(destinationPath),
			err,
		)
	}
	return downloadAndInstallArtifact(
		ctx,
		httpClient,
		rawURL,
		expectedSHA256,
		destinationPath,
	)
}

func downloadAndInstallArtifact(
	ctx context.Context,
	httpClient *http.Client,
	rawURL string,
	expectedSHA256 string,
	destinationPath string,
) error {
	slog.InfoContext(
		ctx,
		"download offline embedding artifact",
		"url",
		rawURL,
		"path",
		destinationPath,
	)
	response, err := downloadArtifactResponse(ctx, httpClient, rawURL)
	if err != nil {
		return err
	}
	temporaryPath, actualSHA256, err := writeArtifactResponse(
		response,
		destinationPath,
	)
	if err != nil {
		return err
	}
	defer func() {
		_ = os.Remove(temporaryPath)
	}()

	if actualSHA256 != expectedSHA256 {
		checksumErr := fmt.Errorf(
			"offline embedding artifact checksum mismatch for %s: got %s, want %s",
			rawURL,
			actualSHA256,
			expectedSHA256,
		)
		slog.ErrorContext(
			ctx,
			"offline embedding artifact checksum mismatch",
			"url",
			rawURL,
			"actual_sha256",
			actualSHA256,
			"expected_sha256",
			expectedSHA256,
			"err",
			checksumErr,
		)
		return checksumErr
	}
	if err := os.Chmod(temporaryPath, offlineModelFileMode); err != nil {
		slog.ErrorContext(
			ctx,
			"set offline embedding artifact permissions failed",
			"path",
			temporaryPath,
			"err",
			err,
		)
		return fmt.Errorf("set offline embedding artifact permissions: %w", err)
	}
	if err := os.Rename(temporaryPath, destinationPath); err != nil {
		slog.ErrorContext(
			ctx,
			"install offline embedding artifact failed",
			"path",
			destinationPath,
			"err",
			err,
		)
		return fmt.Errorf(
			"install offline embedding artifact %s: %w",
			destinationPath,
			err,
		)
	}
	return nil
}

func writeArtifactResponse(
	response *http.Response,
	destinationPath string,
) (string, string, error) {
	slog.Info("write offline embedding artifact response", "path", destinationPath)
	temporaryFile, err := os.CreateTemp(
		filepath.Dir(destinationPath),
		"."+filepath.Base(destinationPath)+".download-*",
	)
	if err != nil {
		slog.Error(
			"create offline embedding download file failed",
			"path",
			destinationPath,
			"err",
			err,
		)
		createErr := fmt.Errorf(
			"create offline embedding download file: %w",
			err,
		)
		if closeErr := response.Body.Close(); closeErr != nil {
			slog.Error(
				"close offline embedding response failed",
				"path",
				destinationPath,
				"err",
				closeErr,
			)
			return "", "", errors.Join(
				createErr,
				fmt.Errorf("close offline embedding response: %w", closeErr),
			)
		}
		return "", "", createErr
	}
	temporaryPath := temporaryFile.Name()

	hash := sha256.New()
	_, copyErr := io.Copy(io.MultiWriter(temporaryFile, hash), response.Body)
	responseCloseErr := response.Body.Close()
	fileCloseErr := temporaryFile.Close()
	if copyErr != nil {
		_ = os.Remove(temporaryPath)
		slog.Error(
			"write offline embedding artifact failed",
			"path",
			temporaryPath,
			"err",
			copyErr,
		)
		return "", "", fmt.Errorf("write offline embedding artifact: %w", copyErr)
	}
	if responseCloseErr != nil {
		_ = os.Remove(temporaryPath)
		slog.Error(
			"close offline embedding response failed",
			"path",
			temporaryPath,
			"err",
			responseCloseErr,
		)
		return "", "", fmt.Errorf(
			"close offline embedding response: %w",
			responseCloseErr,
		)
	}
	if fileCloseErr != nil {
		_ = os.Remove(temporaryPath)
		slog.Error(
			"close offline embedding artifact failed",
			"path",
			temporaryPath,
			"err",
			fileCloseErr,
		)
		return "", "", fmt.Errorf(
			"close offline embedding artifact: %w",
			fileCloseErr,
		)
	}
	actualSHA256 := hex.EncodeToString(hash.Sum(nil))
	return temporaryPath, actualSHA256, nil
}

func downloadArtifactResponse(
	ctx context.Context,
	httpClient *http.Client,
	rawURL string,
) (*http.Response, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"create offline embedding download request failed",
			"url",
			rawURL,
			"err",
			err,
		)
		return nil, fmt.Errorf(
			"create offline embedding download request: %w",
			err,
		)
	}
	response, err := httpClient.Do(request)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"download offline embedding artifact failed",
			"url",
			rawURL,
			"err",
			err,
		)
		return nil, fmt.Errorf(
			"download offline embedding artifact %s: %w",
			rawURL,
			err,
		)
	}
	if response.StatusCode >= http.StatusOK &&
		response.StatusCode < http.StatusMultipleChoices {
		return response, nil
	}
	closeErr := response.Body.Close()
	if closeErr != nil {
		slog.ErrorContext(
			ctx,
			"close failed offline embedding response failed",
			"url",
			rawURL,
			"status",
			response.Status,
			"err",
			closeErr,
		)
		return nil, fmt.Errorf(
			"download offline embedding artifact %s: HTTP status %s; close response: %w",
			rawURL,
			response.Status,
			closeErr,
		)
	}
	return nil, fmt.Errorf(
		"download offline embedding artifact %s: HTTP status %s",
		rawURL,
		response.Status,
	)
}

func artifactMatches(artifactPath string, expectedSHA256 string) (bool, error) {
	artifact, err := os.Open(artifactPath)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		slog.Error(
			"open offline embedding artifact failed",
			"path",
			artifactPath,
			"err",
			err,
		)
		return false, fmt.Errorf(
			"open offline embedding artifact %s: %w",
			artifactPath,
			err,
		)
	}
	hash := sha256.New()
	_, copyErr := io.Copy(hash, artifact)
	closeErr := artifact.Close()
	if copyErr != nil {
		slog.Error(
			"hash offline embedding artifact failed",
			"path",
			artifactPath,
			"err",
			copyErr,
		)
		return false, fmt.Errorf(
			"hash offline embedding artifact %s: %w",
			artifactPath,
			copyErr,
		)
	}
	if closeErr != nil {
		slog.Error(
			"close offline embedding artifact failed",
			"path",
			artifactPath,
			"err",
			closeErr,
		)
		return false, fmt.Errorf(
			"close offline embedding artifact %s: %w",
			artifactPath,
			closeErr,
		)
	}
	actualSHA256 := hex.EncodeToString(hash.Sum(nil))
	return actualSHA256 == expectedSHA256, nil
}
