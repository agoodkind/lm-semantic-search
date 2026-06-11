package semantic

import (
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

func chunkOfBytes(n int) model.StoredChunk {
	return model.StoredChunk{Content: strings.Repeat("a", n)}
}

func TestPackChunksEmptyInputYieldsNoGroups(t *testing.T) {
	groups := packChunksByEstimatedTokens(nil, 32, 6000)
	if len(groups) != 0 {
		t.Fatalf("groups = %d, want 0", len(groups))
	}
}

func TestPackChunksSingleOversizeChunkShipsAlone(t *testing.T) {
	chunks := []model.StoredChunk{chunkOfBytes(100_000), chunkOfBytes(4)}
	groups := packChunksByEstimatedTokens(chunks, 32, 6000)
	if len(groups) != 2 {
		t.Fatalf("groups = %d, want 2", len(groups))
	}
	if len(groups[0]) != 1 {
		t.Fatalf("first group rows = %d, want 1 (oversize ships alone)", len(groups[0]))
	}
}

func TestPackChunksClosesOnTokenBudget(t *testing.T) {
	// 10 chunks of 400 bytes = 100 estimated tokens each; budget 250 packs 2 per group.
	chunks := make([]model.StoredChunk, 10)
	for i := range chunks {
		chunks[i] = chunkOfBytes(400)
	}
	groups := packChunksByEstimatedTokens(chunks, 32, 250)
	if len(groups) != 5 {
		t.Fatalf("groups = %d, want 5", len(groups))
	}
}

func TestPackChunksClosesOnRowCap(t *testing.T) {
	chunks := make([]model.StoredChunk, 10)
	for i := range chunks {
		chunks[i] = chunkOfBytes(4)
	}
	groups := packChunksByEstimatedTokens(chunks, 4, 6000)
	want := []int{4, 4, 2}
	if len(groups) != len(want) {
		t.Fatalf("groups = %d, want %d", len(groups), len(want))
	}
	for i, group := range groups {
		if len(group) != want[i] {
			t.Fatalf("group %d rows = %d, want %d", i, len(group), want[i])
		}
	}
}

func TestPackChunksPreservesOrderAndCoverage(t *testing.T) {
	chunks := make([]model.StoredChunk, 25)
	for i := range chunks {
		chunks[i] = chunkOfBytes((i*53)%900 + 1)
	}
	groups := packChunksByEstimatedTokens(chunks, 8, 300)
	var flattened []model.StoredChunk
	for _, group := range groups {
		flattened = append(flattened, group...)
	}
	if len(flattened) != len(chunks) {
		t.Fatalf("flattened = %d chunks, want %d", len(flattened), len(chunks))
	}
	for i := range chunks {
		if flattened[i].Content != chunks[i].Content {
			t.Fatalf("chunk %d out of order", i)
		}
	}
}
