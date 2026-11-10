package semantic

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
)

func TestQueryTextForEmbeddingAppliesConfiguredPrefix(t *testing.T) {
	service := &Service{cfg: config.Config{
		QueryInstructionPrefix: "Instruct: task.\nQuery: ",
	}}
	got := service.queryTextForEmbedding("find the retry loop")
	want := "Instruct: task.\nQuery: find the retry loop"
	if got != want {
		t.Fatalf("queryTextForEmbedding = %q, want %q", got, want)
	}
}

func TestQueryTextForEmbeddingNoPrefixPassesThrough(t *testing.T) {
	service := &Service{cfg: config.Config{}}
	if got := service.queryTextForEmbedding("q"); got != "q" {
		t.Fatalf("queryTextForEmbedding = %q, want %q", got, "q")
	}
}
