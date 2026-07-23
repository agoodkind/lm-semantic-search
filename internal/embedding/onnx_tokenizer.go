package embedding

import (
	"fmt"
	"log/slog"
	"os"

	"github.com/daulet/tokenizers"
)

type encodedONNXInput struct {
	inputIDs      []int64
	attentionMask []int64
	tokenTypeIDs  []int64
}

type genericTokenizer struct {
	tokenizer *tokenizers.Tokenizer
}

func newGenericTokenizer(
	tokenizerPath string,
	maximumTokens uint32,
) (*genericTokenizer, error) {
	if maximumTokens <= 0 {
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
	loadedTokenizer, err := tokenizers.FromBytesWithTruncation(
		tokenizerData,
		maximumTokens,
		tokenizers.TruncationDirectionRight,
	)
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
	return &genericTokenizer{tokenizer: loadedTokenizer}, nil
}

func (tokenizer *genericTokenizer) encode(text string) (encodedONNXInput, error) {
	encoding, err := tokenizer.tokenizer.EncodeWithOptionsErr(
		text,
		true,
		tokenizers.WithReturnAttentionMask(),
		tokenizers.WithReturnTypeIDs(),
	)
	if err != nil {
		slog.Error("encode ONNX input failed", "err", err)
		return encodedONNXInput{}, fmt.Errorf("encode ONNX input: %w", err)
	}
	if len(encoding.IDs) == 0 {
		return encodedONNXInput{}, fmt.Errorf("ONNX tokenizer returned no token ids")
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
		return encodedONNXInput{}, fmt.Errorf(
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
		return encodedONNXInput{}, fmt.Errorf(
			"ONNX tokenizer returned %d type ids for %d token ids",
			len(tokenTypeIDs),
			len(inputIDs),
		)
	}

	return encodedONNXInput{
		inputIDs:      inputIDs,
		attentionMask: attentionMask,
		tokenTypeIDs:  tokenTypeIDs,
	}, nil
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
