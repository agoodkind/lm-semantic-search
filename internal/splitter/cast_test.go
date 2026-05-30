package splitter

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

// TestASTChunkerMergesSmallDeclarations proves the cAST walk groups several
// small declarations into one chunk up to the budget, rather than emitting one
// chunk per declaration, while still splitting a file that exceeds the budget
// into more than one chunk.
func TestASTChunkerMergesSmallDeclarations(t *testing.T) {
	t.Parallel()

	var source strings.Builder
	source.WriteString("package main\n\n")
	const functionCount = 8
	for index := 1; index <= functionCount; index++ {
		fmt.Fprintf(&source, "func a%d() int { return %d }\n", index, index)
	}

	dispatcher := &Dispatcher{astChunkSize: 100, astChunkOverlap: 0, fallbackChunkSize: 1000, fallbackOverlap: 0}
	result, err := dispatcher.SplitFile(context.Background(), "merge.go", []byte(source.String()))
	if err != nil {
		t.Fatalf("SplitFile returned error: %v", err)
	}
	if result.Strategy != "ast" {
		t.Fatalf("strategy = %q (expected ast)", result.Strategy)
	}
	if len(result.Chunks) < 2 {
		t.Fatalf("expected the oversize file to split into multiple chunks, got %d", len(result.Chunks))
	}
	if len(result.Chunks) >= functionCount {
		t.Fatalf("expected small declarations to merge into fewer chunks than declarations (%d), got %d", functionCount, len(result.Chunks))
	}
	joined := ""
	for _, chunk := range result.Chunks {
		joined += chunk.Content + "\n"
	}
	if got := strings.Count(joined, "func a"); got != functionCount {
		t.Fatalf("expected all %d functions preserved across chunks, found %d", functionCount, got)
	}
}

// TestSwiftDeclarationsChunkByAST proves a Swift file of two functions and one
// struct chunks at declaration boundaries through the AST path, rather than
// collapsing to one whole-file chunk.
func TestSwiftDeclarationsChunkByAST(t *testing.T) {
	t.Parallel()

	source := strings.Join([]string{
		"struct Point {",
		"    let x: Int",
		"    let y: Int",
		"}",
		"",
		"func add(a: Int, b: Int) -> Int {",
		"    return a + b",
		"}",
		"",
		"func mul(a: Int, b: Int) -> Int {",
		"    return a * b",
		"}",
	}, "\n")

	dispatcher := &Dispatcher{astChunkSize: 40, astChunkOverlap: 0, fallbackChunkSize: 1000, fallbackOverlap: 0}
	result, err := dispatcher.SplitFile(context.Background(), "Math.swift", []byte(source))
	if err != nil {
		t.Fatalf("SplitFile returned error: %v", err)
	}
	if result.Strategy != "ast" {
		t.Fatalf("strategy = %q (expected ast; Swift grammar did not load)", result.Strategy)
	}
	if len(result.Chunks) < 2 {
		t.Fatalf("expected Swift declarations to split into multiple chunks, got %d", len(result.Chunks))
	}
	for _, chunk := range result.Chunks {
		if strings.Contains(chunk.Content, "struct Point") && strings.Contains(chunk.Content, "func mul") {
			t.Fatalf("a chunk spanned the whole file instead of aligning to declarations: %q", chunk.Content)
		}
	}
}

// TestASTChunkerSplitsOversizeDeclaration proves a single declaration larger
// than the budget is split into more than one chunk by recursing into its
// children, rather than emitted as one oversize chunk.
func TestASTChunkerSplitsOversizeDeclaration(t *testing.T) {
	t.Parallel()

	var source strings.Builder
	source.WriteString("package main\n\nfunc big() {\n")
	const statementCount = 40
	for index := 0; index < statementCount; index++ {
		fmt.Fprintf(&source, "\tresult%d := compute(%d) + offset\n", index, index)
	}
	source.WriteString("}\n")

	dispatcher := &Dispatcher{astChunkSize: 120, astChunkOverlap: 0, fallbackChunkSize: 1000, fallbackOverlap: 0}
	result, err := dispatcher.SplitFile(context.Background(), "big.go", []byte(source.String()))
	if err != nil {
		t.Fatalf("SplitFile returned error: %v", err)
	}
	if result.Strategy != "ast" {
		t.Fatalf("strategy = %q (expected ast)", result.Strategy)
	}
	if len(result.Chunks) < 2 {
		t.Fatalf("expected an oversize function to split into multiple chunks, got %d", len(result.Chunks))
	}
}
