package splitter

import (
	"context"
	"strings"
	"testing"
)

func TestLangchainSplitJavaScriptBreaksOnFunction(t *testing.T) {
	t.Parallel()

	content := strings.Join([]string{
		"function alpha() {",
		"  return 1;",
		"}",
		"",
		"function bravo() {",
		"  return 2;",
		"}",
		"",
		"function charlie() {",
		"  return 3;",
		"}",
	}, "\n")

	dispatcher := &Dispatcher{
		astChunkSize:      2500,
		astChunkOverlap:   0,
		fallbackChunkSize: 40,
		fallbackOverlap:   0,
	}
	result, err := dispatcher.SplitFileWithType(context.Background(), "example.js", []byte(content), "langchain")
	if err != nil {
		t.Fatalf("SplitFileWithType returned error: %v", err)
	}
	if result.Strategy != "langchain" {
		t.Fatalf("strategy = %q", result.Strategy)
	}
	if len(result.Chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d: %#v", len(result.Chunks), result.Chunks)
	}
	for _, chunk := range result.Chunks[1:] {
		if !strings.HasPrefix(strings.TrimSpace(chunk.Content), "function") {
			t.Fatalf("chunk does not start at a function boundary: %q", chunk.Content)
		}
	}
}

func TestLangchainSplitPythonBreaksOnDef(t *testing.T) {
	t.Parallel()

	content := strings.Join([]string{
		"def alpha():",
		"    return 1",
		"",
		"def bravo():",
		"    return 2",
		"",
		"def charlie():",
		"    return 3",
	}, "\n")

	dispatcher := &Dispatcher{
		astChunkSize:      2500,
		astChunkOverlap:   0,
		fallbackChunkSize: 30,
		fallbackOverlap:   0,
	}
	result, err := dispatcher.SplitFileWithType(context.Background(), "example.py", []byte(content), "langchain")
	if err != nil {
		t.Fatalf("SplitFileWithType returned error: %v", err)
	}
	if len(result.Chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(result.Chunks))
	}
	if !strings.Contains(result.Chunks[0].Content, "def alpha") {
		t.Fatalf("first chunk missing def alpha: %q", result.Chunks[0].Content)
	}
}

func TestLangchainSplitFallsBackForUnknownLanguage(t *testing.T) {
	t.Parallel()

	content := "alpha\nbravo\ncharlie\n"
	chunks := langchainSplit(content, "unknown", "example.unk", 8, 0)
	if len(chunks) == 0 {
		t.Fatal("langchainSplit returned no chunks")
	}
}

func TestCSharpASTSupported(t *testing.T) {
	t.Parallel()

	content := strings.Join([]string{
		"namespace Demo {",
		"  public class Greeter {",
		"    public string Hello() {",
		"      return \"hi\";",
		"    }",
		"  }",
		"}",
	}, "\n")

	dispatcher := NewDispatcher()
	result, err := dispatcher.SplitFile(context.Background(), "Greeter.cs", []byte(content))
	if err != nil {
		t.Fatalf("SplitFile returned error: %v", err)
	}
	if result.Strategy != "ast" {
		t.Fatalf("strategy = %q (expected ast)", result.Strategy)
	}
	if len(result.Chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
}

// TestSplitFileFallbackUsesMarkdownHeadings proves a .md file, which has no AST
// grammar, routes through the recursive separator fallback and cuts one chunk
// per second-level heading rather than packing sections together.
func TestSplitFileFallbackUsesMarkdownHeadings(t *testing.T) {
	t.Parallel()

	content := strings.Join([]string{
		"# Document Title",
		"",
		"Introductory paragraph that sets up the document context here.",
		"",
		"## Alpha",
		"The alpha section explains the very first concept in brief.",
		"",
		"## Bravo",
		"The bravo section explains the second concept in brief here.",
		"",
		"## Charlie",
		"The charlie section explains the third concept in brief now.",
	}, "\n")

	dispatcher := &Dispatcher{
		astChunkSize:      2500,
		astChunkOverlap:   0,
		fallbackChunkSize: 100,
		fallbackOverlap:   0,
	}
	result, err := dispatcher.SplitFile(context.Background(), "doc.md", []byte(content))
	if err != nil {
		t.Fatalf("SplitFile returned error: %v", err)
	}
	if result.Strategy != "fallback" {
		t.Fatalf("strategy = %q (expected fallback; markdown has no AST grammar)", result.Strategy)
	}
	headingChunks := 0
	for _, chunk := range result.Chunks {
		trimmed := strings.TrimSpace(chunk.Content)
		if strings.HasPrefix(trimmed, "## ") {
			headingChunks++
		}
		if strings.Count(trimmed, "## ") > 1 {
			t.Fatalf("a single chunk merged multiple headings: %q", chunk.Content)
		}
	}
	if headingChunks != 3 {
		t.Fatalf("expected 3 section chunks beginning at a heading, got %d: %#v", headingChunks, result.Chunks)
	}
}

// TestSplitFileFallbackForUnknownExtensionBreaksOnBlankLines proves an
// extension absent from extensionLanguages routes through the recursive
// fallback and cuts on paragraph boundaries rather than spanning non-adjacent
// paragraphs.
func TestSplitFileFallbackForUnknownExtensionBreaksOnBlankLines(t *testing.T) {
	t.Parallel()

	paragraph := func(word string) string {
		return strings.TrimRight(strings.Repeat(word+" ", 9), " ")
	}
	content := strings.Join([]string{paragraph("alpha"), "", paragraph("bravo"), "", paragraph("charlie")}, "\n")

	dispatcher := &Dispatcher{
		astChunkSize:      2500,
		astChunkOverlap:   0,
		fallbackChunkSize: 60,
		fallbackOverlap:   0,
	}
	result, err := dispatcher.SplitFile(context.Background(), "notes.unknownext", []byte(content))
	if err != nil {
		t.Fatalf("SplitFile returned error: %v", err)
	}
	if result.Strategy != "fallback" {
		t.Fatalf("strategy = %q (expected fallback)", result.Strategy)
	}
	if len(result.Chunks) < 2 {
		t.Fatalf("expected the recursive fallback to produce multiple chunks, got %d", len(result.Chunks))
	}
	for _, chunk := range result.Chunks {
		if strings.Contains(chunk.Content, "alpha") && strings.Contains(chunk.Content, "charlie") {
			t.Fatalf("a chunk spanned non-adjacent paragraphs, suggesting blind windowing: %q", chunk.Content)
		}
	}
}
