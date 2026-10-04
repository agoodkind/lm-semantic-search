package memory_test

import (
	"context"
	"fmt"
	"math/rand/v2"
	"runtime"
	"testing"

	"goodkind.io/lm-semantic-search/collection"
	"goodkind.io/lm-semantic-search/collection/memory"
)

const (
	benchCollection   = "bench_rows"
	benchDimension    = 768
	benchContentBytes = 1200
	benchBatchRows    = 1000
)

// BenchmarkSearch measures one unfiltered Search over a collection of 768-wide
// vectors and reports the Go heap bytes each stored row uses.
func BenchmarkSearch(b *testing.B) {
	for _, rowCount := range []int{10_000, 100_000} {
		b.Run(fmt.Sprintf("rows=%d", rowCount), func(b *testing.B) {
			ctx := context.Background()
			generator := rand.New(rand.NewPCG(1, 2))
			declaration := collection.Declaration{ItemIDColumn: "", Scalars: nil}

			var before runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&before)

			store := memory.New(memory.Options{EmbeddingModel: ""})
			ensure := collection.EnsureRequest{Collection: benchCollection, Declaration: declaration, Dimension: benchDimension}
			if err := store.EnsureCollection(ctx, ensure); err != nil {
				b.Fatalf("EnsureCollection: %v", err)
			}
			content := string(make([]byte, benchContentBytes))
			for start := 0; start < rowCount; start += benchBatchRows {
				rows := make([]collection.Row, 0, benchBatchRows)
				for index := start; index < start+benchBatchRows; index++ {
					rows = append(rows, collection.Row{
						ID:                fmt.Sprintf("row-%07d", index),
						Content:           content + fmt.Sprint(index),
						RelativePath:      fmt.Sprintf("bench/%07d", index),
						StartLine:         0,
						EndLine:           0,
						FileExtension:     "",
						Metadata:          "{}",
						SplitPart:         0,
						SplitPartRecorded: false,
						Vector:            randomVector(generator),
						Scalars:           nil,
					})
				}
				if err := store.Upsert(ctx, benchCollection, declaration, rows); err != nil {
					b.Fatalf("Upsert: %v", err)
				}
			}

			var after runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&after)
			heapPerRow := float64(after.HeapAlloc-before.HeapAlloc) / float64(rowCount)

			request := collection.SearchRequest{
				Collection:    benchCollection,
				Query:         "",
				Vector:        randomVector(generator),
				Limit:         10,
				MinScore:      0,
				Filter:        nil,
				GroupBy:       "",
				PerGroupLimit: 0,
				Declaration:   declaration,
			}
			b.ResetTimer()
			for b.Loop() {
				hits, err := store.Search(ctx, request)
				if err != nil || len(hits) != 10 {
					b.Fatalf("Search returned %d hits and error %v", len(hits), err)
				}
			}
			b.ReportMetric(heapPerRow, "heap-bytes/row")
			runtime.KeepAlive(store)
		})
	}
}

func randomVector(generator *rand.Rand) []float32 {
	vector := make([]float32, benchDimension)
	for index := range vector {
		vector[index] = generator.Float32()*2 - 1
	}
	return vector
}
