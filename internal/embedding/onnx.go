package embedding

/*
#cgo darwin LDFLAGS: -Wl,-rpath,@loader_path
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

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/metrics"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

const (
	onnxErrorBufferBytes = 2048
	onnxHealthToken      = "x"
)

// onnxProviderName is what this runtime calls itself, taken from the closed set
// so the name it reports is the same value the configuration parses to.
const onnxProviderName = model.EmbeddingProviderONNX

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
		cfg.ModelCacheRoot,
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

func (provider *onnxProvider) ProviderName() model.EmbeddingProvider {
	return onnxProviderName
}

func (provider *onnxProvider) Health(ctx context.Context) error {
	if _, err := provider.Embed(ctx, onnxHealthToken); err != nil {
		slog.WarnContext(ctx, "ONNX embedding provider health probe failed", "err", err)
		return fmt.Errorf("ONNX embedding provider health probe: %w", err)
	}
	return nil
}

// onnxEmbedOutcome is the result of embedding one input in process. An input the
// provider refused comes back carrying the rejection reason, a nil vector, and
// the token count when the input got far enough to be measured, so the caller
// reports it as skipped rather than embedding part of the content.
type onnxEmbedOutcome struct {
	vector     []float32
	tokenCount int
	rejection  onnxInputRejection
}

// Embed returns the vector for one input. An input the provider refuses is an
// error rather than a vector over part of the content, which matches the
// OpenAI-compatible provider's behavior for an input the endpoint rejects.
func (provider *onnxProvider) Embed(
	ctx context.Context,
	text string,
) ([]float32, error) {
	outcome, err := provider.embedOne(ctx, text)
	if err != nil {
		return nil, err
	}
	if outcome.rejection != onnxInputAccepted {
		preset := provider.runtime.preset
		rejectionErr := provider.explainRejection(outcome, len(text))
		slog.WarnContext(
			ctx,
			"ONNX embedding input rejected without embedding",
			"model", preset.Name,
			"reason", string(outcome.rejection),
			"input_bytes", len(text),
			"reported_tokens", outcome.tokenCount,
			"model_max_tokens", preset.MaximumTokens,
			"err", rejectionErr,
		)
		return nil, adapterr.NewEmbedInputRejected(
			provider.clientRejection(outcome, len(text)),
			fmt.Errorf(
				"generate ONNX embedding for %q: %w",
				preset.Name,
				rejectionErr,
			),
		)
	}
	return outcome.vector, nil
}

// clientRejection renders one refused input for the client-visible error. Each
// reason names the limit that actually refused the input: the model's token
// window for an input the tokenizer measured, the byte ceiling for an input
// refused before tokenizing, and no limit at all for a NUL byte, which is not a
// size problem and would send the caller after the wrong fix if a limit were
// quoted beside it.
func (provider *onnxProvider) clientRejection(
	outcome onnxEmbedOutcome,
	inputBytes int,
) adapterr.EmbedInputRejection {
	reason := adapterr.EmbedRejectionReason(outcome.rejection)
	switch outcome.rejection {
	case onnxInputOverTokenLimit:
		return adapterr.EmbedInputRejection{
			Reason:   reason,
			Limit:    adapterr.EmbedLimitTokens,
			Measured: adapterr.ReportedFigure(outcome.tokenCount),
			Maximum:  adapterr.ReportedFigure(int(provider.runtime.preset.MaximumTokens)),
		}
	case onnxInputBytesExceeded:
		return adapterr.EmbedInputRejection{
			Reason:   reason,
			Limit:    adapterr.EmbedLimitBytes,
			Measured: adapterr.ReportedFigure(inputBytes),
			Maximum:  adapterr.ReportedFigure(provider.runtime.tokenizer.maximumInputBytes()),
		}
	case onnxInputContainsNUL, onnxInputAccepted:
		return adapterr.EmbedInputRejection{
			Reason:   reason,
			Limit:    adapterr.EmbedLimitNone,
			Measured: adapterr.UnreportedFigure(),
			Maximum:  adapterr.UnreportedFigure(),
		}
	default:
		return adapterr.EmbedInputRejection{
			Reason:   reason,
			Limit:    adapterr.EmbedLimitNone,
			Measured: adapterr.UnreportedFigure(),
			Maximum:  adapterr.UnreportedFigure(),
		}
	}
}

// skippedInput renders one refused input for the batch's Skipped list. Both token
// figures travel only with a rejection the tokenizer measured against the model's
// window. A NUL byte and an over-long byte count are both refused before
// tokenizing, so neither figure exists for them and both come back unreported
// rather than as a zero the caller would read as a measurement.
func (provider *onnxProvider) skippedInput(
	index int,
	outcome onnxEmbedOutcome,
) SkippedInput {
	reportedTokens := adapterr.UnreportedFigure()
	maximumTokens := adapterr.UnreportedFigure()
	if outcome.rejection == onnxInputOverTokenLimit {
		reportedTokens = adapterr.ReportedFigure(outcome.tokenCount)
		maximumTokens = adapterr.ReportedFigure(int(provider.runtime.preset.MaximumTokens))
	}
	return SkippedInput{
		Index:          index,
		Reason:         adapterr.EmbedRejectionReason(outcome.rejection),
		ReportedTokens: reportedTokens,
		MaxTokens:      maximumTokens,
	}
}

// explainRejection states why one input was not embedded, naming the figure that
// blocked it so an operator can size the offending content.
func (provider *onnxProvider) explainRejection(
	outcome onnxEmbedOutcome,
	inputBytes int,
) error {
	switch outcome.rejection {
	case onnxInputContainsNUL:
		return errors.New("input contains a NUL byte, which the tokenizer cannot read past")
	case onnxInputBytesExceeded:
		return fmt.Errorf(
			"input is %d bytes, over the %d-byte tokenizer bound",
			inputBytes,
			provider.runtime.tokenizer.maximumInputBytes(),
		)
	case onnxInputOverTokenLimit:
		return fmt.Errorf(
			"input is %d tokens, over the %d-token limit",
			outcome.tokenCount,
			provider.runtime.preset.MaximumTokens,
		)
	case onnxInputAccepted:
		// Unreachable: the caller asks for an explanation only after seeing a
		// rejection, and an accepted input carries none.
		return errors.New("input carries no rejection")
	default:
		return fmt.Errorf("input was rejected as %q", string(outcome.rejection))
	}
}

// embedOne tokenizes one input and runs the model over it when it fits the
// model's token limit. It never shortens the input to make it fit.
func (provider *onnxProvider) embedOne(
	ctx context.Context,
	text string,
) (onnxEmbedOutcome, error) {
	if err := ctx.Err(); err != nil {
		slog.WarnContext(ctx, "ONNX embedding cancelled before start", "err", err)
		return failedONNXEmbedOutcome(), fmt.Errorf("generate ONNX embedding: %w", err)
	}

	// Classify the input before taking the runtime lock. An input carrying a NUL
	// byte or past the tokenizer's byte ceiling can never yield a vector over the
	// whole content, and tokenizing it anyway would build a complete encoding,
	// discard it, and hold the shared runtime (with every other embedding and
	// health probe queued behind it) for as long as that took.
	if rejection := provider.runtime.tokenizer.classifyInput(text); rejection != onnxInputAccepted {
		return onnxEmbedOutcome{
			vector:     nil,
			tokenCount: 0,
			rejection:  rejection,
		}, nil
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
		return failedONNXEmbedOutcome(), fmt.Errorf("generate ONNX embedding: %w", err)
	}

	encoded, err := provider.runtime.tokenizer.encode(text)
	if err != nil {
		return failedONNXEmbedOutcome(), err
	}
	if encoded.rejection != onnxInputAccepted {
		return onnxEmbedOutcome{
			vector:     nil,
			tokenCount: encoded.tokenCount,
			rejection:  encoded.rejection,
		}, nil
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
		return failedONNXEmbedOutcome(), fmt.Errorf(
			"run ONNX embedding model %q: %s",
			preset.Name,
			cErrorMessage(errorBuffer),
		)
	}
	if int(outputCount) != len(tokenEmbeddings) {
		return failedONNXEmbedOutcome(), fmt.Errorf(
			"ONNX embedding model %q returned %d values, want %d",
			preset.Name,
			outputCount,
			len(tokenEmbeddings),
		)
	}
	vector, poolErr := poolAndNormalize(
		tokenEmbeddings,
		encoded.attentionMask,
		embeddingDimension,
		preset.Pooling,
	)
	if poolErr != nil {
		return failedONNXEmbedOutcome(), poolErr
	}
	return onnxEmbedOutcome{
		vector:     vector,
		tokenCount: encoded.tokenCount,
		rejection:  onnxInputAccepted,
	}, nil
}

// failedONNXEmbedOutcome is the zero outcome returned alongside an error, so
// no caller mistakes a failed run for an embedded input.
func failedONNXEmbedOutcome() onnxEmbedOutcome {
	return onnxEmbedOutcome{vector: nil, tokenCount: 0, rejection: onnxInputAccepted}
}

func (provider *onnxProvider) EmbedBatch(
	ctx context.Context,
	texts []string,
) (result BatchResult, err error) {
	if len(texts) == 0 {
		return BatchResult{Vectors: nil, Skipped: nil}, nil
	}

	start := clock.Now()
	metrics.EmbedBatchStarted()
	defer func() {
		metrics.EmbedBatchDone(len(texts), clock.Now().Sub(start), err != nil)
	}()

	// Every input the provider refuses is reported as skipped with a nil vector and
	// its reason code, exactly as the OpenAI-compatible provider reports a
	// context_length_exceeded rejection. Both implementations of Provider therefore
	// honor the same promise: a returned vector always covers the whole input, and
	// the caller's split-and-retry loop divides anything that does not fit.
	vectors := make([][]float32, len(texts))
	var skipped []SkippedInput
	for index, text := range texts {
		outcome, embedErr := provider.embedOne(ctx, text)
		if embedErr != nil {
			return BatchResult{}, embedErr
		}
		if outcome.rejection != onnxInputAccepted {
			skipped = append(skipped, provider.skippedInput(index, outcome))
			continue
		}
		vectors[index] = outcome.vector
	}
	return BatchResult{Vectors: vectors, Skipped: skipped}, nil
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
