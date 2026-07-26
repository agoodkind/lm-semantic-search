package embedding

import (
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/daulet/tokenizers"
)

// onnxMaximumInputBytesPerToken caps the input the tokenizer is asked to measure,
// expressed as bytes per token the model accepts. The binding materializes and
// copies the identifier, attention-mask, and type-identifier arrays before any
// limit check, so measuring an unbounded input builds token records and duplicate
// buffers that are discarded moments later. Ordinary prose and code average a few
// bytes per token, so a ceiling of 64 bytes per allowed token sits far above the
// content this cap is meant to stop.
//
// The bound may over-reject. A long unbroken run, such as a base64 value or a
// minified identifier, can collapse largely to one unknown token, so an input
// past this ceiling might in fact have fit the model. That direction is
// deliberate: a refused input is reported through the rejection channel and can
// be split, while a measured one could still only ever be embedded whole.
const onnxMaximumInputBytesPerToken = 64

// onnxInputRejection names why an input must not be embedded. Its values are the
// reason codes carried on SkippedInput, so one rejection reads the same whether
// the in-process tokenizer or a hosted endpoint refused the input.
type onnxInputRejection string

const (
	// onnxInputAccepted marks an input the provider may tokenize and embed whole.
	onnxInputAccepted onnxInputRejection = ""
	// onnxInputContainsNUL marks an input carrying a NUL byte. The binding passes
	// the text as a NUL-terminated C string, so the tokenizer would measure only
	// the bytes before that NUL and any vector would cover that prefix alone.
	onnxInputContainsNUL onnxInputRejection = embedCodeInputContainsNUL
	// onnxInputBytesExceeded marks an input past onnxMaximumInputBytesPerToken per
	// allowed token, rejected before tokenization rather than measured exactly.
	onnxInputBytesExceeded onnxInputRejection = embedCodeInputBytesExceeded
	// onnxInputOverTokenLimit marks an input the tokenizer measured past the
	// model's maximum token count.
	onnxInputOverTokenLimit onnxInputRejection = embedCodeContextLengthExceeded
)

// encodedONNXInput carries one tokenized input. tokenCount is the input's full
// token count, never a truncated one, and is zero for an input rejected before it
// could be measured. A rejected input carries no tensors, so no caller can run
// the model over a shortened copy of the content.
type encodedONNXInput struct {
	inputIDs      []int64
	attentionMask []int64
	tokenTypeIDs  []int64
	tokenCount    int
	rejection     onnxInputRejection
}

type genericTokenizer struct {
	tokenizer     *tokenizers.Tokenizer
	maximumTokens int
}

func newGenericTokenizer(
	tokenizerPath string,
	maximumTokens uint32,
) (*genericTokenizer, error) {
	if maximumTokens == 0 {
		return nil, fmt.Errorf("ONNX tokenizer maximum token count must be positive")
	}
	tokenizerData, err := os.ReadFile(tokenizerPath)
	if err != nil {
		slog.Error(
			"read ONNX tokenizer failed",
			"path",
			tokenizerPath,
			"err",
			err,
		)
		return nil, fmt.Errorf("read ONNX tokenizer %s: %w", tokenizerPath, err)
	}
	// The tokenizer deliberately carries no truncation setting. Truncating here
	// would hide the overflow and let the provider return a vector for content
	// the model never saw; encode instead reports the input's true token count
	// and marks it over the limit so the provider skips it and the shared
	// split-and-retry loop divides it into pieces that fit.
	loadedTokenizer, err := tokenizers.FromBytes(tokenizerData)
	if err != nil {
		slog.Error(
			"load ONNX tokenizer failed",
			"path",
			tokenizerPath,
			"err",
			err,
		)
		return nil, fmt.Errorf("load ONNX tokenizer %s: %w", tokenizerPath, err)
	}
	return &genericTokenizer{
		tokenizer:     loadedTokenizer,
		maximumTokens: int(maximumTokens),
	}, nil
}

// maximumInputBytes is the largest input this tokenizer will measure. It scales
// with the model's token limit, so a model with a larger context accepts a
// proportionally larger input.
func (tokenizer *genericTokenizer) maximumInputBytes() int {
	return tokenizer.maximumTokens * onnxMaximumInputBytesPerToken
}

// classifyInput reports whether text can be tokenized as a whole. It reads only
// settings fixed when the tokenizer loaded and never calls the binding, so a
// caller can reject an input before taking the runtime lock and without
// allocating an encoding it would discard.
func (tokenizer *genericTokenizer) classifyInput(text string) onnxInputRejection {
	if strings.ContainsRune(text, 0) {
		return onnxInputContainsNUL
	}
	if len(text) > tokenizer.maximumInputBytes() {
		return onnxInputBytesExceeded
	}
	return onnxInputAccepted
}

// encode tokenizes text without truncation. An input classifyInput refuses, and
// an input whose measurement lands past the model's maximum token count, both
// come back carrying the reason and no tensors, so the caller reports the input
// as skipped instead of embedding a shortened copy. The provider classifies the
// same input before locking the runtime, and this call repeats the check so no
// caller of the tokenizer can reach the NUL-terminated binding with an input it
// would only read part of.
func (tokenizer *genericTokenizer) encode(text string) (encodedONNXInput, error) {
	if rejection := tokenizer.classifyInput(text); rejection != onnxInputAccepted {
		return rejectedEncodedONNXInput(rejection, 0), nil
	}
	encoding, err := tokenizer.tokenizer.EncodeWithOptionsErr(
		text,
		true,
		tokenizers.WithReturnAttentionMask(),
		tokenizers.WithReturnTypeIDs(),
	)
	if err != nil {
		slog.Error("encode ONNX input failed", "err", err)
		return emptyEncodedONNXInput(), fmt.Errorf("encode ONNX input: %w", err)
	}
	if len(encoding.IDs) == 0 {
		return emptyEncodedONNXInput(), fmt.Errorf("ONNX tokenizer returned no token ids")
	}
	if len(encoding.IDs) > tokenizer.maximumTokens {
		return rejectedEncodedONNXInput(onnxInputOverTokenLimit, len(encoding.IDs)), nil
	}

	inputIDs := uint32sToInt64s(encoding.IDs)
	attentionMask := uint32sToInt64s(encoding.AttentionMask)
	if len(attentionMask) == 0 {
		attentionMask = make([]int64, len(inputIDs))
		for index := range attentionMask {
			attentionMask[index] = 1
		}
	}
	if len(attentionMask) != len(inputIDs) {
		return emptyEncodedONNXInput(), fmt.Errorf(
			"ONNX tokenizer returned %d attention values for %d token ids",
			len(attentionMask),
			len(inputIDs),
		)
	}

	tokenTypeIDs := uint32sToInt64s(encoding.TypeIDs)
	if len(tokenTypeIDs) == 0 {
		tokenTypeIDs = make([]int64, len(inputIDs))
	}
	if len(tokenTypeIDs) != len(inputIDs) {
		return emptyEncodedONNXInput(), fmt.Errorf(
			"ONNX tokenizer returned %d type ids for %d token ids",
			len(tokenTypeIDs),
			len(inputIDs),
		)
	}

	return encodedONNXInput{
		inputIDs:      inputIDs,
		attentionMask: attentionMask,
		tokenTypeIDs:  tokenTypeIDs,
		tokenCount:    len(inputIDs),
		rejection:     onnxInputAccepted,
	}, nil
}

// emptyEncodedONNXInput is the zero encoding returned alongside an error, so no
// caller mistakes a failed tokenization for an embeddable input.
func emptyEncodedONNXInput() encodedONNXInput {
	return encodedONNXInput{
		inputIDs:      nil,
		attentionMask: nil,
		tokenTypeIDs:  nil,
		tokenCount:    0,
		rejection:     onnxInputAccepted,
	}
}

// rejectedEncodedONNXInput is the encoding for an input that must not be
// embedded. It carries the reason and the measured token count when the input got
// far enough to be measured, and never any tensors.
func rejectedEncodedONNXInput(
	rejection onnxInputRejection,
	tokenCount int,
) encodedONNXInput {
	return encodedONNXInput{
		inputIDs:      nil,
		attentionMask: nil,
		tokenTypeIDs:  nil,
		tokenCount:    tokenCount,
		rejection:     rejection,
	}
}

func (tokenizer *genericTokenizer) Close() error {
	if err := tokenizer.tokenizer.Close(); err != nil {
		slog.Error("close ONNX tokenizer failed", "err", err)
		return fmt.Errorf("close ONNX tokenizer: %w", err)
	}
	return nil
}

func uint32sToInt64s(values []uint32) []int64 {
	converted := make([]int64, 0, len(values))
	for _, value := range values {
		converted = append(converted, int64(value))
	}
	return converted
}
