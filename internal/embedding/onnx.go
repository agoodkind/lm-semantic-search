package embedding

/*
#cgo linux LDFLAGS: -Wl,-rpath,$ORIGIN
#cgo pkg-config: onnxruntime
#include <stdlib.h>
#include "onnx_bridge.h"
*/
import "C"

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"sync"
	"unsafe"

	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/metrics"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

const (
	onnxProviderName     = "onnx"
	onnxErrorBufferBytes = 2048
	onnxHealthToken      = "x"
)

var (
	onnxRuntimesMutex sync.Mutex
	onnxRuntimes      = make(map[string]*inProcessONNXRuntime)
)

type onnxProvider struct {
	runtime *inProcessONNXRuntime
}

type inProcessONNXRuntime struct {
	session   *C.lms_onnx_session
	tokenizer *genericTokenizer
	preset    offlinemodel.Preset
	mutex     sync.Mutex
}

func newONNXProvider(
	ctx context.Context,
	cfg config.Config,
) (Provider, error) {
	preset, err := offlinemodel.Resolve(cfg.OfflineEmbeddingModel)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"resolve offline embedding model failed",
			"model",
			cfg.OfflineEmbeddingModel,
			"err",
			err,
		)
		return nil, fmt.Errorf("resolve offline embedding model: %w", err)
	}
	files, err := ensureModelFiles(
		ctx,
		http.DefaultClient,
		cfg.StateRoot,
		preset,
	)
	if err != nil {
		return nil, err
	}

	onnxRuntimesMutex.Lock()
	defer onnxRuntimesMutex.Unlock()
	if runtime, found := onnxRuntimes[files.modelPath]; found {
		return &onnxProvider{runtime: runtime}, nil
	}
	runtime, err := initializeONNXRuntime(files, preset)
	if err != nil {
		return nil, err
	}
	onnxRuntimes[files.modelPath] = runtime
	return &onnxProvider{runtime: runtime}, nil
}

func initializeONNXRuntime(
	files cachedModelFiles,
	preset offlinemodel.Preset,
) (*inProcessONNXRuntime, error) {
	tokenizer, err := newGenericTokenizer(
		files.tokenizerPath,
		preset.MaximumTokens,
	)
	if err != nil {
		return nil, err
	}

	modelPath := C.CString(files.modelPath)
	defer C.free(unsafe.Pointer(modelPath))
	errorBuffer := make([]byte, onnxErrorBufferBytes)
	session := C.lms_onnx_session_create(
		modelPath,
		(*C.char)(unsafe.Pointer(&errorBuffer[0])),
	)
	if session == nil {
		sessionErr := fmt.Errorf(
			"initialize ONNX Runtime for %q: %s",
			preset.Name,
			cErrorMessage(errorBuffer),
		)
		if closeErr := tokenizer.Close(); closeErr != nil {
			slog.Error(
				"close tokenizer after ONNX Runtime initialization failed",
				"model",
				preset.Name,
				"err",
				closeErr,
			)
			return nil, errors.Join(sessionErr, closeErr)
		}
		return nil, sessionErr
	}
	return &inProcessONNXRuntime{
		session:   session,
		tokenizer: tokenizer,
		preset:    preset,
		mutex:     sync.Mutex{},
	}, nil
}

func (provider *onnxProvider) ProviderName() string {
	return onnxProviderName
}

func (provider *onnxProvider) Health(ctx context.Context) error {
	if _, err := provider.Embed(ctx, onnxHealthToken); err != nil {
		slog.WarnContext(ctx, "ONNX embedding provider health probe failed", "err", err)
		return fmt.Errorf("ONNX embedding provider health probe: %w", err)
	}
	return nil
}

func (provider *onnxProvider) Embed(
	ctx context.Context,
	text string,
) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		slog.WarnContext(ctx, "ONNX embedding cancelled before start", "err", err)
		return nil, fmt.Errorf("generate ONNX embedding: %w", err)
	}

	provider.runtime.mutex.Lock()
	defer provider.runtime.mutex.Unlock()

	if err := ctx.Err(); err != nil {
		slog.WarnContext(
			ctx,
			"ONNX embedding cancelled while waiting for runtime",
			"err",
			err,
		)
		return nil, fmt.Errorf("generate ONNX embedding: %w", err)
	}

	encoded, err := provider.runtime.tokenizer.encode(text)
	if err != nil {
		return nil, err
	}
	preset := provider.runtime.preset
	embeddingDimension := int(preset.Dimension)
	tokenEmbeddings := make(
		[]float32,
		len(encoded.inputIDs)*embeddingDimension,
	)
	var outputCount C.size_t
	errorBuffer := make([]byte, onnxErrorBufferBytes)
	useTokenTypeIDs := C.int(0)
	if preset.UsesTokenTypeIDs {
		useTokenTypeIDs = 1
	}
	result := C.lms_onnx_run(
		provider.runtime.session,
		(*C.int64_t)(unsafe.Pointer(&encoded.inputIDs[0])),
		(*C.int64_t)(unsafe.Pointer(&encoded.attentionMask[0])),
		(*C.int64_t)(unsafe.Pointer(&encoded.tokenTypeIDs[0])),
		useTokenTypeIDs,
		C.size_t(len(encoded.inputIDs)),
		C.size_t(embeddingDimension),
		(*C.float)(unsafe.Pointer(&tokenEmbeddings[0])),
		C.size_t(len(tokenEmbeddings)),
		&outputCount,
		(*C.char)(unsafe.Pointer(&errorBuffer[0])),
		C.size_t(len(errorBuffer)),
	)
	if result != 0 {
		return nil, fmt.Errorf(
			"run ONNX embedding model %q: %s",
			preset.Name,
			cErrorMessage(errorBuffer),
		)
	}
	if int(outputCount) != len(tokenEmbeddings) {
		return nil, fmt.Errorf(
			"ONNX embedding model %q returned %d values, want %d",
			preset.Name,
			outputCount,
			len(tokenEmbeddings),
		)
	}
	return poolAndNormalize(
		tokenEmbeddings,
		encoded.attentionMask,
		embeddingDimension,
		preset.Pooling,
	)
}

func (provider *onnxProvider) EmbedBatch(
	ctx context.Context,
	texts []string,
) (vectors [][]float32, err error) {
	if len(texts) == 0 {
		return nil, nil
	}

	start := clock.Now()
	metrics.EmbedBatchStarted()
	defer func() {
		metrics.EmbedBatchDone(len(texts), clock.Now().Sub(start), err != nil)
	}()

	vectors = make([][]float32, 0, len(texts))
	for _, text := range texts {
		vector, embedErr := provider.Embed(ctx, text)
		if embedErr != nil {
			return nil, embedErr
		}
		vectors = append(vectors, vector)
	}
	return vectors, nil
}

func poolAndNormalize(
	tokenEmbeddings []float32,
	attentionMask []int64,
	dimension int,
	pooling offlinemodel.Pooling,
) ([]float32, error) {
	if dimension <= 0 ||
		len(tokenEmbeddings) < dimension ||
		len(tokenEmbeddings)%dimension != 0 {
		return nil, fmt.Errorf("ONNX embedding tensor has an invalid shape")
	}
	tokenCount := len(tokenEmbeddings) / dimension
	if len(attentionMask) != tokenCount {
		return nil, fmt.Errorf(
			"ONNX attention mask has %d values for %d tokens",
			len(attentionMask),
			tokenCount,
		)
	}

	vector := make([]float32, dimension)
	switch pooling {
	case offlinemodel.PoolingCLS:
		copy(vector, tokenEmbeddings[:dimension])
	case offlinemodel.PoolingMean:
		attendedTokenCount := 0
		for tokenIndex, attended := range attentionMask {
			if attended == 0 {
				continue
			}
			offset := tokenIndex * dimension
			for dimensionIndex := range vector {
				vector[dimensionIndex] += tokenEmbeddings[offset+dimensionIndex]
			}
			attendedTokenCount++
		}
		if attendedTokenCount == 0 {
			return nil, fmt.Errorf("ONNX attention mask contains no attended tokens")
		}
		inverseTokenCount := float32(1) / float32(attendedTokenCount)
		for dimensionIndex := range vector {
			vector[dimensionIndex] *= inverseTokenCount
		}
	default:
		return nil, fmt.Errorf("ONNX pooling mode %q is not supported", pooling)
	}

	var squaredNorm float64
	for dimensionIndex := range vector {
		squaredNorm += float64(vector[dimensionIndex]) *
			float64(vector[dimensionIndex])
	}
	if squaredNorm == 0 {
		return nil, fmt.Errorf("ONNX embedding model returned a zero vector")
	}
	inverseNorm := float32(1 / math.Sqrt(squaredNorm))
	for dimensionIndex := range vector {
		vector[dimensionIndex] *= inverseNorm
	}
	return vector, nil
}

func cErrorMessage(buffer []byte) string {
	for index, value := range buffer {
		if value == 0 {
			return string(buffer[:index])
		}
	}
	return string(buffer)
}
