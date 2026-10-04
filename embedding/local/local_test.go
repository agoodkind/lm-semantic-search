package local_test

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/collection"
	"goodkind.io/lm-semantic-search/collection/memory"
	"goodkind.io/lm-semantic-search/embedding"
	"goodkind.io/lm-semantic-search/embedding/local"
)

const (
	testModel            = "bge-small"
	testCollection       = "local_code_snippets"
	languageColumn       = "language"
	cacheRootEnvironment = "LOCAL_EMBEDDING_TEST_CACHE_ROOT"
	// unreachableProxy refuses every connection. A process that uses it as its
	// HTTP proxy cannot complete any HTTP request.
	unreachableProxy = "http://127.0.0.1:1"
)

// modelCacheRoot returns the cache root the ONNX provider tests also use. The
// pinned model files download once per machine.
func modelCacheRoot(t *testing.T) string {
	t.Helper()
	root := filepath.Join(os.TempDir(), "lm-semantic-search-offline-model-test-cache")
	if err := os.MkdirAll(root, 0o700); err != nil {
		t.Fatalf("create model cache %s: %v", root, err)
	}
	return root
}

func newProvider(t *testing.T, cacheRoot string) embedding.Provider {
	t.Helper()
	provider, err := local.New(context.Background(), local.Options{Model: testModel, CacheRoot: cacheRoot})
	if errors.Is(err, local.ErrModelUnavailable) {
		t.Skipf("model files are not cached and the download failed: %v", err)
	}
	if err != nil {
		t.Fatalf("local.New: %v", err)
	}
	return provider
}

type snippet struct {
	id       string
	language string
	content  string
}

func snippets() []snippet {
	return []snippet{
		{id: "go-server", language: "go", content: "func main() {\n\thttp.HandleFunc(\"/health\", func(w http.ResponseWriter, r *http.Request) {\n\t\tw.WriteHeader(http.StatusOK)\n\t})\n\tlog.Fatal(http.ListenAndServe(\":8080\", nil))\n}"},
		{id: "python-csv", language: "python", content: "def read_rows(path):\n    with open(path, newline=\"\") as handle:\n        return list(csv.DictReader(handle))"},
		{id: "sql-join", language: "sql", content: "SELECT customers.name, SUM(orders.total) FROM orders JOIN customers ON customers.id = orders.customer_id GROUP BY customers.name;"},
		{id: "rust-fibonacci", language: "rust", content: "fn fibonacci(n: u64) -> u64 {\n    if n < 2 { return n; }\n    fibonacci(n - 1) + fibonacci(n - 2)\n}"},
		{id: "bash-archive", language: "bash", content: "#!/usr/bin/env bash\ntar -czf backup.tar.gz \"$1\"\necho \"compressed $1 into backup.tar.gz\""},
		{id: "js-debounce", language: "javascript", content: "function debounce(callback, delay) {\n  let timer;\n  return (...args) => {\n    clearTimeout(timer);\n    timer = setTimeout(() => callback(...args), delay);\n  };\n}"},
	}
}

func snippetDeclaration() collection.Declaration {
	return collection.Declaration{
		ItemIDColumn: "",
		Scalars: []collection.ScalarColumn{
			{Name: languageColumn, Type: collection.ScalarTypeString, Nullable: false, MaxLength: 32},
		},
	}
}

// TestLocalProviderSearchesMemoryStore embeds code snippets with the in-process
// model, stores them in the in-memory store, and proves a natural-language
// query returns the snippet that answers it first. The test uses no Milvus
// server and no embedding service.
func TestLocalProviderSearchesMemoryStore(t *testing.T) {
	ctx := context.Background()
	provider := newProvider(t, modelCacheRoot(t))
	described, err := local.Describe(testModel)
	if err != nil {
		t.Fatalf("Describe: %v", err)
	}

	corpus := snippets()
	contents := make([]string, 0, len(corpus))
	for _, entry := range corpus {
		contents = append(contents, entry.content)
	}
	batch, err := provider.EmbedBatch(ctx, contents)
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	if len(batch.Skipped) != 0 {
		t.Fatalf("EmbedBatch skipped %+v, want every snippet embedded", batch.Skipped)
	}

	store := memory.New(memory.Options{EmbeddingModel: described.Name})
	declaration := snippetDeclaration()
	ensure := collection.EnsureRequest{Collection: testCollection, Declaration: declaration, Dimension: described.Dimension}
	if err := store.EnsureCollection(ctx, ensure); err != nil {
		t.Fatalf("EnsureCollection: %v", err)
	}
	rows := make([]collection.Row, 0, len(corpus))
	for index, entry := range corpus {
		rows = append(rows, collection.Row{
			ID:                entry.id,
			Content:           entry.content,
			RelativePath:      "snippets/" + entry.id,
			StartLine:         1,
			EndLine:           1,
			FileExtension:     "",
			Metadata:          "{}",
			SplitPart:         0,
			SplitPartRecorded: false,
			Vector:            batch.Vectors[index],
			Scalars:           map[string]collection.ScalarValue{languageColumn: collection.StringScalar(entry.language)},
		})
	}
	if err := store.Upsert(ctx, testCollection, declaration, rows); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	search := func(query string, filter *collection.Filter) []collection.Hit {
		t.Helper()
		vector, embedErr := provider.Embed(ctx, described.QueryPrefix+query)
		if embedErr != nil {
			t.Fatalf("Embed %q: %v", query, embedErr)
		}
		hits, searchErr := store.Search(ctx, collection.SearchRequest{
			Collection:    testCollection,
			Query:         query,
			Vector:        vector,
			Limit:         3,
			MinScore:      0,
			Filter:        filter,
			GroupBy:       "",
			PerGroupLimit: 0,
			Declaration:   declaration,
		})
		if searchErr != nil {
			t.Fatalf("Search %q: %v", query, searchErr)
		}
		if len(hits) == 0 {
			t.Fatalf("Search %q returned no hit", query)
		}
		return hits
	}

	expected := map[string]string{
		"read the rows of a csv file":                 "python-csv",
		"http server with a health check endpoint":    "go-server",
		"compute a fibonacci number recursively":      "rust-fibonacci",
		"compress a directory into a tar archive":     "bash-archive",
		"total order amount for each customer":        "sql-join",
		"delay a callback until events stop arriving": "js-debounce",
	}
	for query, wantID := range expected {
		hits := search(query, nil)
		if hits[0].ID != wantID {
			t.Errorf("query %q ranked %s first with score %.3f, want %s", query, hits[0].ID, hits[0].Score, wantID)
		}
	}

	scripting := collection.ColumnIn(languageColumn, collection.StringValues([]string{"python", "bash"}))
	for _, hit := range search("compute a fibonacci number recursively", &scripting) {
		language := hit.Scalars[languageColumn].Value.String
		if language != "python" && language != "bash" {
			t.Errorf("filtered search returned %s with language %q", hit.ID, language)
		}
	}
}

// TestLocalProviderStartsFromCacheWithoutNetwork proves a process that cannot
// complete any HTTP request builds the provider from cached model files and
// embeds a text. The first provider downloads the files when the cache lacks
// them. The child process then runs with an HTTP proxy that refuses every
// connection.
func TestLocalProviderStartsFromCacheWithoutNetwork(t *testing.T) {
	cacheRoot := modelCacheRoot(t)
	newProvider(t, cacheRoot)

	command := exec.CommandContext(context.Background(), os.Args[0], "-test.run=^TestEmbedFromCacheInChildProcess$", "-test.v")
	command.Env = append(
		os.Environ(),
		cacheRootEnvironment+"="+cacheRoot,
		"HTTPS_PROXY="+unreachableProxy,
		"HTTP_PROXY="+unreachableProxy,
		"NO_PROXY=",
	)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("child process without network failed: %v\n%s", err, output)
	}
}

// TestEmbedFromCacheInChildProcess runs only inside the child process that
// TestLocalProviderStartsFromCacheWithoutNetwork starts.
func TestEmbedFromCacheInChildProcess(t *testing.T) {
	cacheRoot := os.Getenv(cacheRootEnvironment)
	if cacheRoot == "" {
		t.Skip("runs only as a child process of TestLocalProviderStartsFromCacheWithoutNetwork")
	}
	provider, err := local.New(context.Background(), local.Options{Model: testModel, CacheRoot: cacheRoot})
	if err != nil {
		t.Fatalf("local.New without network: %v", err)
	}
	vector, err := provider.Embed(context.Background(), "func main() {}")
	if err != nil {
		t.Fatalf("Embed without network: %v", err)
	}
	if len(vector) != 384 {
		t.Fatalf("vector width = %d, want 384", len(vector))
	}
}
