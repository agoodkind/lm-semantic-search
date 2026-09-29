package library

import (
	"fmt"
	"math"
)

// lexicalRankParameters are the BM25 settings in the float32 precision that
// Milvus 2.6.18 applies. knowhere receives k1 and b as float32 index
// parameters.
type lexicalRankParameters struct {
	k1 float32
	b  float32
}

// newLexicalRankParameters converts resolved configuration values to the
// float32 settings. It rejects a k1 that is not positive and finite in float32
// and a b outside [0, 1], with an error that wraps [ErrInvalidRequest].
func newLexicalRankParameters(k1 float64, b float64) (lexicalRankParameters, error) {
	converted := lexicalRankParameters{k1: float32(k1), b: float32(b)}
	k1Finite := !math.IsNaN(float64(converted.k1)) && !math.IsInf(float64(converted.k1), 0)
	if !k1Finite || converted.k1 <= 0 {
		return lexicalRankParameters{}, invalidRequest(fmt.Sprintf(
			"BM25 k1 %v must be positive and finite as a float32",
			k1,
		))
	}
	if math.IsNaN(b) || converted.b < 0 || converted.b > 1 {
		return lexicalRankParameters{}, invalidRequest(fmt.Sprintf("BM25 b %v must be between 0 and 1", b))
	}
	return converted, nil
}

// lexicalCorpus is the namespace corpus statistics that one query freezes.
// Size counts every committed occurrence in the namespace, including
// occurrences with no terms, and totalTokens sums their document lengths.
type lexicalCorpus struct {
	size        uint64
	totalTokens uint64
}

// lexicalScorer computes Milvus 2.6.18 BM25 scores for one analyzed query
// against one frozen corpus.
type lexicalScorer struct {
	parameters    lexicalRankParameters
	averageLength float32
	weights       map[uint32]float32
}

// newLexicalScorer builds the query weights. It reports false when the query
// has no terms or the corpus average document length is zero, and the caller
// then omits the lexical ranking.
//
// Each weight follows Milvus BM25Stats.BuildIDF: the query term frequency as a
// float32 times the float32 conversion of ln(1+(N-df+0.5)/(df+0.5)), where the
// logarithm is computed in float64. A repeated query token raises the query
// term frequency, and tokens that share a hash share one weight.
func newLexicalScorer(
	parameters lexicalRankParameters,
	corpus lexicalCorpus,
	query []lexicalTerm,
	documentFrequencies map[uint32]uint64,
) (lexicalScorer, bool) {
	if len(query) == 0 || corpus.size == 0 || corpus.totalTokens == 0 {
		return lexicalScorer{}, false
	}
	corpusSize := float64(corpus.size)
	weights := make(map[uint32]float32, len(query))
	for _, term := range query {
		documentFrequency := float64(documentFrequencies[term.hash])
		inverseFrequency := math.Log(1 + (corpusSize-documentFrequency+0.5)/(documentFrequency+0.5))
		weights[term.hash] = float32(term.frequency) * float32(inverseFrequency)
	}
	return lexicalScorer{
		parameters:    parameters,
		averageLength: float32(float64(corpus.totalTokens) / corpusSize),
		weights:       weights,
	}, true
}

// documentFactor is the knowhere GetDocValueBM25Computer expression
// tf*(k1+1)/(tf+k1*(1-b+b*(doc_len/avgdl))) in float32 with the fused
// multiply-adds that the Milvus 2.6.18 build contracts: b*ratio+(1-b) and
// k1*normalization+tf each round once.
func (scorer lexicalScorer) documentFactor(termFrequency float32, documentLength float32) float32 {
	k1 := scorer.parameters.k1
	b := scorer.parameters.b
	lengthRatio := documentLength / scorer.averageLength
	normalization := fusedMultiplyAdd32(b, lengthRatio, 1-b)
	denominator := fusedMultiplyAdd32(k1, normalization, termFrequency)
	numerator := float32(termFrequency * float32(k1+1))
	return numerator / denominator
}

// scoreDocument returns the BM25 score of one lexical content from its
// postings in ascending term hash order. It computes score = weight*factor +
// score as one fused float32 operation per query term in that order, as the
// knowhere accumulation compiles. A posting for a term outside the query
// contributes nothing.
func (scorer lexicalScorer) scoreDocument(postings []lexicalTerm, documentLength uint64) float32 {
	length := float32(documentLength)
	var score float32
	for _, posting := range postings {
		weight, queried := scorer.weights[posting.hash]
		if !queried {
			continue
		}
		score = fusedMultiplyAdd32(weight, scorer.documentFactor(float32(posting.frequency), length), score)
	}
	return score
}

// fusedMultiplyAdd32 returns a*b+c rounded once to the nearest float32, ties
// to even. The float64 product of two float32 values is exact. TwoSum then
// splits the float64 sum into its rounded value and the exact residual. A
// rounded sum on a float32 midpoint is resolved by the sign of the residual.
func fusedMultiplyAdd32(a float32, b float32, c float32) float32 {
	product := float64(a) * float64(b)
	addend := float64(c)
	sum := product + addend
	virtual := sum - product
	residual := (product - (sum - virtual)) + (addend - virtual)
	rounded := float32(sum)
	if residual == 0 || float64(rounded) == sum {
		return rounded
	}
	below, above := rounded, rounded
	if float64(rounded) < sum {
		above = math.Nextafter32(rounded, float32(math.Inf(1)))
	} else {
		below = math.Nextafter32(rounded, float32(math.Inf(-1)))
	}
	if sum-float64(below) != float64(above)-sum {
		return rounded
	}
	if residual > 0 {
		return above
	}
	return below
}
