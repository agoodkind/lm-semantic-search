package library

import (
	"errors"
	"math"
	"math/big"
	"math/rand/v2"
	"testing"
)

// exactFusedMultiplyAdd32 rounds a*b+c once through arbitrary precision.
func exactFusedMultiplyAdd32(a float32, b float32, c float32) float32 {
	const precision = 256
	product := new(big.Float).SetPrec(precision).SetFloat64(float64(a))
	product.Mul(product, new(big.Float).SetPrec(precision).SetFloat64(float64(b)))
	product.Add(product, new(big.Float).SetPrec(precision).SetFloat64(float64(c)))
	rounded, _ := product.Float32()
	return rounded
}

func TestFusedMultiplyAdd32RoundsOnce(t *testing.T) {
	t.Parallel()
	// 1+2^-11+2^-24 is the midpoint between two adjacent float32 values. The
	// 2^-60 addend decides the rounding, and a float64 sum loses it.
	nearOne := float32(1 + 1.0/4096)
	midpointCase := [3]float32{nearOne, nearOne, float32(math.Ldexp(1, -60))}
	if doubleRounded := float32(float64(nearOne)*float64(nearOne) + float64(midpointCase[2])); doubleRounded == exactFusedMultiplyAdd32(midpointCase[0], midpointCase[1], midpointCase[2]) {
		t.Fatal("the midpoint case does not distinguish one rounding from two")
	}
	cases := [][3]float32{midpointCase, {nearOne, nearOne, -float32(math.Ldexp(1, -60))}, {0.75, 1.3333334, 0.25}, {1, 1, 0}}
	random := rand.New(rand.NewPCG(712, 2))
	for range 200000 {
		cases = append(cases, [3]float32{
			float32(math.Ldexp(random.Float64(), random.IntN(40)-20)),
			float32(math.Ldexp(random.Float64(), random.IntN(40)-20)),
			float32(math.Ldexp(random.Float64()-0.5, random.IntN(60)-30)),
		})
	}
	for _, operands := range cases {
		got := fusedMultiplyAdd32(operands[0], operands[1], operands[2])
		want := exactFusedMultiplyAdd32(operands[0], operands[1], operands[2])
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("fusedMultiplyAdd32(%v, %v, %v) = %v, want %v", operands[0], operands[1], operands[2], got, want)
		}
	}
}

func TestNewLexicalRankParametersValidatesFloat32Settings(t *testing.T) {
	t.Parallel()
	for _, testCase := range []struct {
		name  string
		k1    float64
		b     float64
		valid bool
	}{
		{name: "defaults", k1: 1.2, b: 0.75, valid: true},
		{name: "b zero", k1: 1.2, b: 0, valid: true},
		{name: "b one", k1: 1.2, b: 1, valid: true},
		{name: "k1 zero", k1: 0, b: 0.75, valid: false},
		{name: "k1 negative", k1: -1, b: 0.75, valid: false},
		{name: "k1 not a number", k1: math.NaN(), b: 0.75, valid: false},
		{name: "k1 overflows float32", k1: 1e300, b: 0.75, valid: false},
		{name: "k1 at the maximum", k1: maxBM25K1, b: 0.75, valid: true},
		{name: "k1 above the maximum", k1: math.Nextafter(maxBM25K1, math.Inf(1)), b: 0.75, valid: false},
		{name: "k1 with NaN scores", k1: 3e38, b: 0.75, valid: false},
		{name: "k1 infinite", k1: math.Inf(1), b: 0.75, valid: false},
		{name: "k1 zero as float32", k1: 1e-50, b: 0.75, valid: false},
		{name: "b negative", k1: 1.2, b: -0.1, valid: false},
		{name: "b above one", k1: 1.2, b: 1.1, valid: false},
		{name: "b not a number", k1: 1.2, b: math.NaN(), valid: false},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			_, err := newLexicalRankParameters(testCase.k1, testCase.b)
			if testCase.valid && err != nil {
				t.Fatalf("newLexicalRankParameters(%v, %v) = %v, want nil", testCase.k1, testCase.b, err)
			}
			if !testCase.valid && !errors.Is(err, ErrInvalidRequest) {
				t.Fatalf("newLexicalRankParameters(%v, %v) = %v, want ErrInvalidRequest", testCase.k1, testCase.b, err)
			}
		})
	}
}

func TestNewLexicalScorerOmitsUnrankableQueries(t *testing.T) {
	t.Parallel()
	parameters, err := newLexicalRankParameters(1.2, 0.75)
	if err != nil {
		t.Fatal(err)
	}
	populated := lexicalCorpus{size: 4, totalTokens: 12}
	if _, ranked := newLexicalScorer(parameters, populated, analyzeLexical("!!!").terms, nil); ranked {
		t.Fatal("a query without tokens produced a lexical ranking")
	}
	if _, ranked := newLexicalScorer(parameters, lexicalCorpus{size: 3, totalTokens: 0}, analyzeLexical("alpha").terms, nil); ranked {
		t.Fatal("a corpus with zero average length produced a lexical ranking")
	}
	if _, ranked := newLexicalScorer(parameters, lexicalCorpus{size: 0, totalTokens: 0}, analyzeLexical("alpha").terms, nil); ranked {
		t.Fatal("an empty corpus produced a lexical ranking")
	}
}

func TestLexicalScorerMultipliesRepeatedQueryTerms(t *testing.T) {
	t.Parallel()
	parameters, err := newLexicalRankParameters(1.2, 0.75)
	if err != nil {
		t.Fatal(err)
	}
	corpus := lexicalCorpus{size: 10, totalTokens: 40}
	alpha := lexicalTermHash("alpha")
	frequencies := map[uint32]uint64{alpha: 3}
	single, ranked := newLexicalScorer(parameters, corpus, analyzeLexical("alpha").terms, frequencies)
	if !ranked {
		t.Fatal("single-term query was not ranked")
	}
	repeated, ranked := newLexicalScorer(parameters, corpus, analyzeLexical("alpha ALPHA alpha").terms, frequencies)
	if !ranked {
		t.Fatal("repeated-term query was not ranked")
	}
	if repeated.weights[alpha] != 3*single.weights[alpha] {
		t.Fatalf("repeated weight %v, want 3 times %v", repeated.weights[alpha], single.weights[alpha])
	}
	document := analyzeLexical("alpha beta")
	singleScore := single.scoreDocument(document.terms, document.length)
	repeatedScore := repeated.scoreDocument(document.terms, document.length)
	if singleScore <= 0 || repeatedScore <= singleScore {
		t.Fatalf("scores single %v repeated %v, want 0 < single < repeated", singleScore, repeatedScore)
	}
}

func TestLexicalScorerStaysFiniteAtTheMaximumK1(t *testing.T) {
	t.Parallel()
	for _, b := range []float64{0, 0.75, 1} {
		parameters, err := newLexicalRankParameters(maxBM25K1, b)
		if err != nil {
			t.Fatal(err)
		}
		alpha := lexicalTermHash("alpha")
		query := []lexicalTerm{{hash: alpha, frequency: lexicalMaxTermFrequency}}
		corpus := lexicalCorpus{size: math.MaxInt64, totalTokens: math.MaxInt64}
		scorer, ranked := newLexicalScorer(parameters, corpus, query, map[uint32]uint64{alpha: 1})
		if !ranked {
			t.Fatal("query was not ranked")
		}
		for _, length := range []uint64{lexicalMaxTermFrequency, math.MaxInt64} {
			score := scorer.scoreDocument([]lexicalTerm{{hash: alpha, frequency: lexicalMaxTermFrequency}}, length)
			if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) || score <= 0 {
				t.Fatalf("b %v document length %d: score %v, want a finite positive score", b, length, score)
			}
		}
	}
}
