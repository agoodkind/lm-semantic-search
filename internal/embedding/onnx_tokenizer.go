package embedding

import (
	"fmt"
	"log/slog"
	"os"

	"github.com/daulet/tokenizers"
)

// encodedONNXInput carries one tokenized input. tokenCount is the input's full
// token count, never a truncated one. When overLimit is true the input
// tokenizes past the model's maximum and the tensors are left empty, so no
// caller can run the model over a shortened copy of the content.
type encodedONNXInput struct {
	inputIDs      []int64
	attentionMask []int64
	tokenTypeIDs  []int64
	tokenCount    int
	overLimit     bool
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

// encode tokenizes text without truncation. An input past the model's maximum
// token count comes back marked over the limit with its full token count and no
// tensors, so the caller reports it as skipped instead of embedding a shortened
// copy.
func (tokenizer *genericTokenizer) encode(text string) (encodedONNXInput, error) {
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
		overLimit := emptyEncodedONNXInput()
		overLimit.tokenCount = len(encoding.IDs)
		overLimit.overLimit = true
		return overLimit, nil
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
		overLimit:     false,
	}, nil
}

// emptyEncodedONNXInput is the zero encoding returned when tokenizing failed or
// when the input is over the model's limit and must not be embedded.
func emptyEncodedONNXInput() encodedONNXInput {
	return encodedONNXInput{
		inputIDs:      nil,
		attentionMask: nil,
		tokenTypeIDs:  nil,
		tokenCount:    0,
		overLimit:     false,
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
