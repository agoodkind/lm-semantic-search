//go:build live

package live

import (
	"context"
	"crypto/sha256"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
)

type codebaseBaselineResult struct {
	chunks int32
	hits   map[string][]string
}

func TestLibraryCodebaseMatchesHealthyBaseline(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "sub"), 0o700); err != nil {
		t.Fatalf("create fixture subdirectory: %v", err)
	}
	duplicate := goFile(goFunction("Shared", "sharedmarker"))
	files := map[string]string{
		"first.go":    duplicate,
		"sub/copy.go": duplicate,
		"notes.md":    "# Shared marker\n\nThe sharedmarker notes describe the fixture.\n",
	}
	for name, content := range files {
		writeCodebaseFile(t, root, name, content)
	}
	results := map[config.CodebaseStoreKind]codebaseBaselineResult{}
	for _, backend := range []config.CodebaseStoreKind{config.CodebaseStoreSemantic, config.CodebaseStoreLibrary} {
		t.Run(string(backend), func(t *testing.T) {
			codebaseDaemon := newCodebaseLiveDaemon(t, backend)
			codebaseDaemon.index(t, root)
			status, err := codebaseDaemon.client.GetIndex(context.Background(), &pb.GetIndexRequest{Path: root})
			if err != nil {
				t.Fatalf("read codebase status: %v", err)
			}
			result := codebaseBaselineResult{
				chunks: status.GetCodebase().GetLastSuccessfulRun().GetTotalChunks(),
				hits:   map[string][]string{},
			}
			if int(result.chunks) != len(files) {
				t.Fatalf("published %d chunks, want %d file owners", result.chunks, len(files))
			}
			cases := []struct {
				label      string
				path       string
				extensions []string
				wantPaths  []string
			}{
				{"all", root, nil, []string{"first.go", "notes.md", "sub/copy.go"}},
				{"go", root, []string{".go"}, []string{"first.go", "sub/copy.go"}},
				{"sub", filepath.Join(root, "sub"), nil, []string{"sub/copy.go"}},
			}
			for _, searchCase := range cases {
				response, err := codebaseDaemon.client.SearchCode(context.Background(), &pb.SearchCodeRequest{
					Path: searchCase.path, Query: "sharedmarker", Limit: codebaseSearchLimit,
					ExtensionFilter: searchCase.extensions,
				})
				if err != nil {
					t.Fatalf("%s search: %v", searchCase.label, err)
				}
				result.hits[searchCase.label] = baselineHitIdentities(t, response.GetResults(), files, searchCase.wantPaths)
				t.Logf("%s %s ordered identities: %v", backend, searchCase.label, result.hits[searchCase.label])
			}
			t.Logf("model=%s dimension=%d hybrid=%t chunks=%d", codebaseDaemon.config.EmbeddingModel,
				codebaseDaemon.config.EmbeddingDimension, codebaseDaemon.config.HybridMode, result.chunks)
			results[backend] = result
		})
	}
	baseline, baselineOK := results[config.CodebaseStoreSemantic]
	libraryResult, libraryOK := results[config.CodebaseStoreLibrary]
	if !baselineOK || !libraryOK {
		t.Fatal("both isolated backends must complete before comparison")
	}
	if baseline.chunks != libraryResult.chunks {
		t.Fatalf("baseline published %d chunks, library published %d", baseline.chunks, libraryResult.chunks)
	}
	for label, expected := range baseline.hits {
		actual := libraryResult.hits[label]
		if !maps.Equal(baselineIdentitySet(expected), baselineIdentitySet(actual)) {
			t.Fatalf("%s identity sets differ: baseline %v, library %v", label, expected, actual)
		}
		if !slices.Equal(expected, actual) {
			t.Fatalf("%s ordered identities differ: baseline %v, library %v", label, expected, actual)
		}
		t.Logf("%s has %d exact matching identities; ordered identity equality=%t", label, len(actual), slices.Equal(expected, actual))
	}
}

func baselineHitIdentities(t *testing.T, hits []*pb.SearchResult, files map[string]string, wantPaths []string) []string {
	t.Helper()
	identities := make([]string, 0, len(hits))
	paths := map[string]bool{}
	for index, hit := range hits {
		path := hit.GetRelativePath()
		content, exists := files[path]
		if !exists || !strings.Contains(content, hit.GetContent()) {
			t.Fatalf("hit from %s is not an indexed source excerpt", path)
		}
		if index > 0 && hit.GetScore() > hits[index-1].GetScore() {
			t.Fatal("search hits are not in descending score order")
		}
		paths[path] = true
		identities = append(identities, fmt.Sprintf("%s:%d:%d:%x", path, hit.GetStartLine(), hit.GetEndLine(), sha256.Sum256([]byte(hit.GetContent()))))
	}
	if len(identities) != len(wantPaths) || len(baselineIdentitySet(identities)) != len(identities) {
		t.Fatalf("search returned %d hits with %d distinct identities, want %d", len(identities), len(baselineIdentitySet(identities)), len(wantPaths))
	}
	if !maps.Equal(paths, baselineIdentitySet(wantPaths)) {
		t.Fatalf("search returned owners %v, want %v", paths, wantPaths)
	}
	return identities
}

func baselineIdentitySet(identities []string) map[string]bool {
	result := make(map[string]bool, len(identities))
	for _, identity := range identities {
		result[identity] = true
	}
	return result
}
