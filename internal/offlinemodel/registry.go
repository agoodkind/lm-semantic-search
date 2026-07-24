// Package offlinemodel defines the downloadable ONNX embedding model presets.
package offlinemodel

import (
	"fmt"
	"strings"
)

const (
	// EmbeddingGemma is the code-capable default offline embedding model.
	EmbeddingGemma = "embeddinggemma"
	// BGESmall is the smaller general-purpose fallback and test model.
	BGESmall = "bge-small"
	// DefaultName is the preset selected when no offline model is configured.
	DefaultName = EmbeddingGemma
)

const (
	embeddingGemmaRevision = "5090578d9565bb06545b4552f76e6bc2c93e4a66"
	embeddingGemmaModelURL = "https://huggingface.co/onnx-community/" +
		"embeddinggemma-300m-ONNX/resolve/" +
		embeddingGemmaRevision + "/onnx/model_q4.onnx"
	embeddingGemmaModelDataURL = "https://huggingface.co/onnx-community/" +
		"embeddinggemma-300m-ONNX/resolve/" +
		embeddingGemmaRevision + "/onnx/model_q4.onnx_data"
	embeddingGemmaVocabularyURL = "https://huggingface.co/onnx-community/" +
		"embeddinggemma-300m-ONNX/resolve/" +
		embeddingGemmaRevision + "/tokenizer.json"

	bgeSmallRevision = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
	bgeSmallModelURL = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/" +
		bgeSmallRevision + "/onnx/model.onnx"
	bgeSmallVocabularyURL = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/" +
		bgeSmallRevision + "/tokenizer.json"
)

func artifactDigest(parts ...string) string {
	return strings.Join(parts, "")
}

// Pooling identifies how token embeddings become one sentence vector.
type Pooling string

const (
	// PoolingCLS selects the first token embedding.
	PoolingCLS Pooling = "cls"
	// PoolingMean averages the embeddings for attended tokens.
	PoolingMean Pooling = "mean"
)

// Preset pins every artifact and inference setting for one offline model.
type Preset struct {
	Name             string
	ModelONNXURL     string
	ModelSHA256      string
	ModelDataURL     string
	ModelDataSHA256  string
	TokenizerURL     string
	TokenizerSHA256  string
	Dimension        int32
	Pooling          Pooling
	QueryPrefix      string
	MaximumTokens    uint32
	UsesTokenTypeIDs bool
}

var presets = map[string]Preset{
	EmbeddingGemma: {
		Name:         EmbeddingGemma,
		ModelONNXURL: embeddingGemmaModelURL,
		ModelSHA256: artifactDigest(
			"ad1dfee8", "1a70f794", "4b9b9d1c", "c6e48075",
			"b832881c", "f33fab2f", "2b248be7", "8f3f0043",
		),
		ModelDataURL: embeddingGemmaModelDataURL,
		ModelDataSHA256: artifactDigest(
			"599962c3", "143b040d", "e2dd05e5", "975be3e9",
			"091dd067", "cacc6a8f", "7186e320", "3bab9e02",
		),
		TokenizerURL: embeddingGemmaVocabularyURL,
		TokenizerSHA256: artifactDigest(
			"4dda02fa", "af32bc91", "031dc8c8", "8457ac27",
			"2b00c101", "6cc67975", "7d1c441b", "248b9c47",
		),
		Dimension:        768,
		Pooling:          PoolingMean,
		QueryPrefix:      "task: code retrieval | query: ",
		MaximumTokens:    2048,
		UsesTokenTypeIDs: false,
	},
	BGESmall: {
		Name:         BGESmall,
		ModelONNXURL: bgeSmallModelURL,
		ModelSHA256: artifactDigest(
			"828e1496", "d7fabb79", "cfa4dcd8", "4fa38625",
			"c0d3d21d", "a474a00f", "08db0f55", "9940cf35",
		),
		ModelDataURL:    "",
		ModelDataSHA256: "",
		TokenizerURL:    bgeSmallVocabularyURL,
		TokenizerSHA256: artifactDigest(
			"d241a60d", "5e8f04cc", "1b2b3e9e", "f7a4921b",
			"27bf526d", "9f6050ab", "90f9267a", "1f9e5c66",
		),
		Dimension:        384,
		Pooling:          PoolingCLS,
		QueryPrefix:      "",
		MaximumTokens:    512,
		UsesTokenTypeIDs: true,
	},
}

// Resolve returns a pinned preset by name. An empty name selects DefaultName.
func Resolve(name string) (Preset, error) {
	normalizedName := strings.TrimSpace(strings.ToLower(name))
	if normalizedName == "" {
		normalizedName = DefaultName
	}
	preset, found := presets[normalizedName]
	if !found {
		return Preset{}, fmt.Errorf(
			"offline embedding model %q is not supported; use %q or %q",
			name,
			EmbeddingGemma,
			BGESmall,
		)
	}
	return preset, nil
}
