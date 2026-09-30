//go:build live

package live

import (
	"context"
	"strings"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

func TestLibraryCodebasePreservesNULSourceAndReusesNormalizedVector(t *testing.T) {
	codebaseDaemon := newLibraryCodebaseDaemon(t)
	root := t.TempDir()
	const owner = "notes.md"
	const content = "# Storage integrity\n\nA storage check before\x00after verifies integrity.\n"
	writeCodebaseFile(t, root, owner, content)
	codebaseDaemon.index(t, root)
	before := requireNULCodebaseOccurrence(t, codebaseDaemon, root, owner, "before\x00after")

	writeCodebaseFile(t, root, owner, strings.ReplaceAll(content, "\x00", " "))
	codebaseDaemon.sync(t, root)
	after := requireNULCodebaseOccurrence(t, codebaseDaemon, root, owner, "before after")
	if before.rowKey == after.rowKey {
		t.Fatal("NUL and space source text have the same occurrence identity")
	}
	if before.vectorID != after.vectorID {
		t.Fatalf("normalized embedding input did not reuse its canonical vector: %s and %s", before.vectorID, after.vectorID)
	}
	if after.generationOrder != before.generationOrder+1 {
		t.Fatalf("replacement generation = %d, want %d", after.generationOrder, before.generationOrder+1)
	}
	if strings.ReplaceAll(before.source, "\x00", " ") != after.source {
		t.Fatal("replacement changed source bytes other than the NUL-to-space edit")
	}
	t.Logf("distinct occurrence identities %s and %s reuse vector %s", before.rowKey, after.rowKey, before.vectorID)
}

func requireNULCodebaseOccurrence(t *testing.T, codebaseDaemon *libraryCodebaseDaemon, root string, owner string, marker string) codebaseOccurrence {
	t.Helper()
	response, err := codebaseDaemon.client.SearchCode(context.Background(), &pb.SearchCodeRequest{
		Path: root, Query: "storage integrity before\x00after", Limit: codebaseSearchLimit,
	})
	if err != nil {
		t.Fatalf("search source excerpt: %v", err)
	}
	if len(response.GetResults()) != 1 {
		t.Fatalf("search returned %d occurrences, want one", len(response.GetResults()))
	}
	hit := response.GetResults()[0]
	if hit.GetRelativePath() != owner || !strings.Contains(hit.GetContent(), marker) {
		t.Fatalf("search returned %q from %q, want original marker %q from %s", hit.GetContent(), hit.GetRelativePath(), marker, owner)
	}
	occurrences := codebaseDaemon.readCodebaseOwners(t)[owner]
	if len(occurrences) != 1 {
		t.Fatalf("owner has %d published occurrences, want one", len(occurrences))
	}
	if occurrences[0].source != hit.GetContent() {
		t.Fatal("search excerpt differs from the published source bytes")
	}
	return occurrences[0]
}
