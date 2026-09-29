package library_test

import (
	"errors"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/library"
)

func validDescriptor(t *testing.T) library.StoreDescriptor {
	t.Helper()
	root := t.TempDir()
	return library.StoreDescriptor{
		CatalogPath:       filepath.Join(root, "catalog.sqlite"),
		LockPath:          filepath.Join(root, "catalog.lock"),
		PoolID:            "pool-a",
		EmbeddingModel:    "nvidia/NV-EmbedCode-7b-v1",
		EmbeddingRevision: "rev-1",
		Dimension:         4096,
		Normalization:     "l2",
	}
}

func TestConfigValidateAcceptsZeroBudgetsAndExplicitZeroBM25B(t *testing.T) {
	t.Parallel()
	zero := 0.0
	one := 1.0
	for _, testCase := range []struct {
		name     string
		bm25B    *float64
		bm25K1   float64
		block    int
		analyzer string
	}{
		{name: "default b", bm25B: nil, bm25K1: 0, block: 0, analyzer: ""},
		{name: "explicit zero b", bm25B: &zero, bm25K1: 0, block: 0, analyzer: ""},
		{name: "explicit one b", bm25B: &one, bm25K1: 0, block: 0, analyzer: ""},
		{name: "standard analyzer", bm25B: nil, bm25K1: 0, block: 0, analyzer: library.StandardAnalyzer},
		{name: "maximum k1", bm25B: nil, bm25K1: 1e6, block: 0, analyzer: ""},
		{name: "maximum query block", bm25B: nil, bm25K1: 0, block: 16384, analyzer: ""},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			config := library.Config{Store: validDescriptor(t), BM25B: testCase.bm25B, BM25K1: testCase.bm25K1, QueryBlockSize: testCase.block, AnalyzerIdentity: testCase.analyzer}
			if err := config.Validate(); err != nil {
				t.Fatalf("Validate() = %v, want nil", err)
			}
		})
	}
}

func TestConfigValidateRejectsInvalidDescriptors(t *testing.T) {
	t.Parallel()
	for _, testCase := range []struct {
		name   string
		mutate func(*library.StoreDescriptor)
		want   string
	}{
		{name: "empty catalog path", mutate: func(d *library.StoreDescriptor) { d.CatalogPath = "" }, want: "CatalogPath is empty"},
		{name: "relative catalog path", mutate: func(d *library.StoreDescriptor) { d.CatalogPath = "catalog.sqlite" }, want: "not absolute"},
		{name: "unclean lock path", mutate: func(d *library.StoreDescriptor) { d.LockPath += "/../catalog.lock" }, want: "not clean"},
		{name: "lock equals catalog", mutate: func(d *library.StoreDescriptor) { d.LockPath = d.CatalogPath }, want: "equals CatalogPath"},
		{name: "empty pool", mutate: func(d *library.StoreDescriptor) { d.PoolID = "" }, want: "PoolID is empty"},
		{name: "empty model", mutate: func(d *library.StoreDescriptor) { d.EmbeddingModel = "" }, want: "EmbeddingModel is empty"},
		{name: "padded revision", mutate: func(d *library.StoreDescriptor) { d.EmbeddingRevision = " rev-1" }, want: "surrounding whitespace"},
		{name: "empty normalization", mutate: func(d *library.StoreDescriptor) { d.Normalization = "" }, want: "Normalization is empty"},
		{name: "zero dimension", mutate: func(d *library.StoreDescriptor) { d.Dimension = 0 }, want: "Dimension 0 must be positive"},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			descriptor := validDescriptor(t)
			testCase.mutate(&descriptor)
			err := library.Config{Store: descriptor}.Validate()
			assertInvalidRequest(t, err, testCase.want)
		})
	}
}

func TestConfigValidateRejectsNegativeBudgetsAndInvalidRanking(t *testing.T) {
	t.Parallel()
	negativeB := -0.1
	overOneB := 1.5
	for _, testCase := range []struct {
		name   string
		mutate func(*library.Config)
		want   string
	}{
		{name: "negative batch rows", mutate: func(c *library.Config) { c.MaxBatchRows = -1 }, want: "MaxBatchRows is -1"},
		{name: "negative query timeout", mutate: func(c *library.Config) { c.QueryTimeout = -1 }, want: "QueryTimeout is -1"},
		{name: "negative page limit", mutate: func(c *library.Config) { c.MaxPageSize = -1 }, want: "MaxPageSize is -1"},
		{name: "unknown search mode", mutate: func(c *library.Config) { c.SearchMode = 9 }, want: "SearchMode 9"},
		{name: "negative k1", mutate: func(c *library.Config) { c.BM25K1 = -1 }, want: "BM25K1"},
		{name: "k1 zero as float32", mutate: func(c *library.Config) { c.BM25K1 = 1e-50 }, want: "BM25K1"},
		{name: "k1 infinite as float32", mutate: func(c *library.Config) { c.BM25K1 = 1e39 }, want: "BM25K1"},
		{name: "query block above the Milvus limit", mutate: func(c *library.Config) { c.QueryBlockSize = 16385 }, want: "QueryBlockSize"},
		{name: "k1 finite float32 overflow", mutate: func(c *library.Config) { c.BM25K1 = 3e38 }, want: "BM25K1"},
		{name: "k1 just above the maximum", mutate: func(c *library.Config) { c.BM25K1 = math.Nextafter(1e6, 2e6) }, want: "BM25K1"},
		{name: "negative b", mutate: func(c *library.Config) { c.BM25B = &negativeB }, want: "BM25B"},
		{name: "b over one", mutate: func(c *library.Config) { c.BM25B = &overOneB }, want: "BM25B"},
		{name: "negative rrf k", mutate: func(c *library.Config) { c.RRFK = -60 }, want: "RRFK"},
		{name: "unknown analyzer", mutate: func(c *library.Config) { c.AnalyzerIdentity = "english-stemmer" }, want: `AnalyzerIdentity "english-stemmer" is not supported`},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			config := library.Config{Store: validDescriptor(t)}
			testCase.mutate(&config)
			assertInvalidRequest(t, config.Validate(), testCase.want)
		})
	}
}

func conversationNamespace() library.NamespaceSpec {
	return library.NamespaceSpec{
		ID:     "clyde-conversations",
		Policy: library.AppendOnly,
		Scalars: []library.ScalarColumn{
			{Name: "workspace", Type: library.String, Nullable: true, Mutable: true, MaxLength: 64},
			{Name: "archived", Type: library.Bool, Mutable: true},
			{Name: "message_index", Type: library.Int64},
		},
	}
}

func TestNamespaceValidateRejectsInvalidDeclarations(t *testing.T) {
	t.Parallel()
	if err := conversationNamespace().Validate(); err != nil {
		t.Fatalf("Validate() = %v, want nil", err)
	}
	for _, testCase := range []struct {
		name   string
		mutate func(*library.NamespaceSpec)
		want   string
	}{
		{name: "empty id", mutate: func(s *library.NamespaceSpec) { s.ID = "" }, want: "namespace ID is empty"},
		{name: "zero policy", mutate: func(s *library.NamespaceSpec) { s.Policy = 0 }, want: "policy 0"},
		{name: "duplicate column", mutate: func(s *library.NamespaceSpec) { s.Scalars = append(s.Scalars, s.Scalars[0]) }, want: "twice"},
		{name: "expression column name", mutate: func(s *library.NamespaceSpec) { s.Scalars[0].Name = "a or 1=1" }, want: "must match"},
		{name: "unbounded string", mutate: func(s *library.NamespaceSpec) { s.Scalars[0].MaxLength = 0 }, want: "MaxLength 0 must be positive"},
		{name: "bool with length", mutate: func(s *library.NamespaceSpec) { s.Scalars[1].MaxLength = 4 }, want: "non-string type"},
		{name: "zero type", mutate: func(s *library.NamespaceSpec) { s.Scalars[2].Type = 0 }, want: "type 0"},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			spec := conversationNamespace()
			testCase.mutate(&spec)
			assertInvalidRequest(t, spec.Validate(), testCase.want)
		})
	}
}

func TestNamespaceValidateOccurrenceChecksScalarsAgainstDeclaration(t *testing.T) {
	t.Parallel()
	valid := library.Occurrence{
		RowKey:         "message-1/chat/0",
		SortKey:        "0001",
		SourceText:     "alpha",
		SearchText:     "alpha",
		EmbeddingInput: "alpha",
		Scalars: map[string]library.ScalarValue{
			"workspace":     {Type: library.String, Null: true},
			"archived":      {Type: library.Bool, Bool: false},
			"message_index": {Type: library.Int64, Int64: 7},
		},
	}
	if err := conversationNamespace().ValidateOccurrence(valid); err != nil {
		t.Fatalf("ValidateOccurrence(valid) = %v, want nil", err)
	}
	absent := valid
	absent.Scalars = map[string]library.ScalarValue{}
	if err := conversationNamespace().ValidateOccurrence(absent); err != nil {
		t.Fatalf("ValidateOccurrence(absent scalars) = %v, want nil", err)
	}

	for _, testCase := range []struct {
		name   string
		key    string
		value  library.ScalarValue
		mutate func(*library.Occurrence)
		want   string
	}{
		{name: "unknown column", key: "provider", value: library.ScalarValue{Type: library.String, String: "claude"}, want: `undeclared column "provider"`},
		{name: "type mismatch", key: "message_index", value: library.ScalarValue{Type: library.String, String: "7"}, want: "does not match declared type"},
		{name: "null in non-nullable", key: "archived", value: library.ScalarValue{Type: library.Bool, Null: true}, want: "not nullable"},
		{name: "null with value", key: "workspace", value: library.ScalarValue{Type: library.String, Null: true, String: "x"}, want: "null value also sets"},
		{name: "string over length", key: "workspace", value: library.ScalarValue{Type: library.String, String: strings.Repeat("w", 65)}, want: "65 bytes, over the declared 64"},
		{name: "two tagged fields", key: "message_index", value: library.ScalarValue{Type: library.Int64, Int64: 1, Bool: true}, want: "also sets a string or bool"},
		{name: "empty row key", mutate: func(o *library.Occurrence) { o.RowKey = "" }, want: "row key is empty"},
		{name: "whitespace embedding input", mutate: func(o *library.Occurrence) { o.EmbeddingInput = " \n" }, want: "no non-whitespace"},
		{name: "invalid source text", mutate: func(o *library.Occurrence) { o.SourceText = "\xff" }, want: "SourceText is not valid UTF-8"},
		{name: "NUL in search text", mutate: func(o *library.Occurrence) { o.SearchText = "before\x00after" }, want: "SearchText contains a NUL byte"},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			occurrence := valid
			occurrence.Scalars = map[string]library.ScalarValue{}
			if testCase.key != "" {
				occurrence.Scalars[testCase.key] = testCase.value
			}
			if testCase.mutate != nil {
				testCase.mutate(&occurrence)
			}
			assertInvalidRequest(t, conversationNamespace().ValidateOccurrence(occurrence), testCase.want)
		})
	}
}

func assertInvalidRequest(t *testing.T, err error, wantSubstring string) {
	t.Helper()
	if !errors.Is(err, library.ErrInvalidRequest) {
		t.Fatalf("error = %v, want one wrapping ErrInvalidRequest", err)
	}
	if !strings.Contains(err.Error(), wantSubstring) {
		t.Fatalf("error = %q, want it to contain %q", err.Error(), wantSubstring)
	}
}
