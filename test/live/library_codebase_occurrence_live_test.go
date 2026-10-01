//go:build live

package live

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"goodkind.io/gksyntax/chunk"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/library"
)

func TestLibraryCodebaseRepeatedSameLineOccurrences(t *testing.T) {
	codebaseDaemon := newLibraryCodebaseDaemon(t)
	root := t.TempDir()
	source := "{\"text\":\"" + strings.Repeat("duplicateordinalmarker ", 800) + "\"}\n"
	if path := os.Getenv("LMS_CODEBASE_OCCURRENCE_SOURCE"); path != "" {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		source = string(data)
	}
	writeCodebaseFile(t, root, "repeated.json", source)
	projected, err := chunk.NewDispatcher().SplitFileWithType(t.Context(), filepath.Join(root, "repeated.json"), []byte(source), "ast")
	if err != nil {
		t.Fatal(err)
	}
	counts := make(map[string]int)
	expected := make(map[string]int)
	maxTokens := config.ActiveEmbedTokenLimit(codebaseDaemon.config)
	if configured := codebaseDaemon.config.EmbeddingMaxTokens; configured > 0 {
		maxTokens = min(maxTokens, configured)
	}
	duplicate := false
	for _, item := range projected.Chunks {
		key := fmt.Sprintf("%d:%d:%x", item.StartLine, item.EndLine, sha256.Sum256([]byte(item.Content)))
		counts[key]++
		duplicate = duplicate || counts[key] > 1
		parts, err := library.PrepareText(t.Context(), library.PrepareRequest{
			Text: item.Content, MaxTokens: maxTokens,
		})
		if err != nil {
			t.Fatal(err)
		}
		for _, part := range parts {
			expected[item.Content[part.ByteStart:part.ByteEnd]]++
		}
	}
	if !duplicate {
		t.Fatal("the real parser did not produce distinct same-line repeated segments")
	}
	codebaseDaemon.index(t, root)
	before := readCrossNamespaceOwners(t, codebaseDaemon, root)
	rows := before["repeated.json"]
	wanted := 0
	for _, count := range expected {
		wanted += count
	}
	if len(rows) != wanted {
		t.Fatalf("published %d occurrences, want all %d prepared source parts", len(rows), wanted)
	}
	keys := make(map[string]bool)
	vectors := make(map[string]string)
	for _, row := range rows {
		if expected[row.source] == 0 {
			t.Fatal("publication repeated or invented a prepared source part")
		}
		expected[row.source]--
		if keys[row.rowKey] {
			t.Fatalf("published duplicate row key %s", row.rowKey)
		}
		keys[row.rowKey] = true
		if vectorID, exists := vectors[row.source]; exists && vectorID != row.vectorID {
			t.Fatal("identical source inputs published different canonical vectors")
		}
		vectors[row.source] = row.vectorID
	}
	requireRepeatedSourceSearch(t, codebaseDaemon, root, rows)
	baseline := codebaseRepairCounters(t, codebaseDaemon)["embed_batches_total"]
	codebaseDaemon.sync(t, root)
	requireCrossNamespaceOwnersEqual(t, before, readCrossNamespaceOwners(t, codebaseDaemon, root))
	if actual := codebaseRepairCounters(t, codebaseDaemon)["embed_batches_total"]; actual != baseline {
		t.Fatalf("unchanged sync submitted %d embedding batches", actual-baseline)
	}
	codebaseDaemon.stop()
	child := startCodebaseRestartChild(t, codebaseDaemon)
	codebaseDaemon.sync(t, root)
	requireCrossNamespaceOwnersEqual(t, before, readCrossNamespaceOwners(t, codebaseDaemon, root))
	if actual := codebaseRepairCounters(t, codebaseDaemon)["embed_batches_total"]; actual != 0 {
		t.Fatalf("restart and unchanged sync submitted %d embedding batches", actual)
	}
	requireRepeatedSourceSearch(t, codebaseDaemon, root, rows)
	killCodebaseRestartChild(t, child)
	t.Logf("source SHA256=%x; published %d distinct occurrences; unchanged sync and restart preserved row keys and submitted zero new embedding batches", sha256.Sum256([]byte(source)), len(rows))
}

func requireRepeatedSourceSearch(t *testing.T, codebaseDaemon *libraryCodebaseDaemon, root string, rows []codebaseOccurrence) {
	t.Helper()
	response, err := codebaseDaemon.client.SearchCode(t.Context(), &pb.SearchCodeRequest{Path: root, Query: "repeated text", Limit: codebaseSearchLimit})
	if err != nil {
		t.Fatal(err)
	}
	wanted := min(len(rows), codebaseSearchLimit)
	if len(response.GetResults()) != wanted {
		t.Fatalf("public search returned %d occurrences, want %d", len(response.GetResults()), wanted)
	}
	counts := make(map[string]int)
	for _, row := range rows {
		counts[row.source]++
	}
	for _, result := range response.GetResults() {
		if result.GetRelativePath() != "repeated.json" {
			t.Fatalf("public search returned owner %s", result.GetRelativePath())
		}
		if counts[result.GetContent()] == 0 {
			t.Fatal("public search repeated or invented a source occurrence")
		}
		counts[result.GetContent()]--
	}
}
