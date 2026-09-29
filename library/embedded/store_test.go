package embedded_test

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/embedded"
	"goodkind.io/lm-semantic-search/library/internal/storebinding"
	"goodkind.io/lm-semantic-search/library/internal/vectorcodec"
)

const (
	// testDimension matches a common production embedding width.
	testDimension = 1024
	// scoreTolerance bounds the difference from the independent cosine.
	scoreTolerance = 1e-9
	// bindRaceAttempts repeats the first-bind race on fresh roots.
	bindRaceAttempts = 32
	// fanOutVectorCount is the number of library-format IDs in the fan-out test.
	fanOutVectorCount = 64
	catalogUUIDA      = "11111111-1111-4111-8111-111111111111"
	catalogUUIDB      = "22222222-2222-4222-8222-222222222222"
)

func encodeBinding(t *testing.T, catalogUUID string, writerHost string, dimension int) string {
	t.Helper()
	text, err := storebinding.Encode(storebinding.Binding{
		CatalogUUID:       catalogUUID,
		CatalogPath:       "/var/lib/lms/catalog.sqlite",
		WriterHost:        writerHost,
		PoolID:            "pool-test",
		EmbeddingModel:    "test-model",
		EmbeddingRevision: "rev-1",
		Dimension:         dimension,
		Normalization:     "l2",
	})
	if err != nil {
		t.Fatalf("encode binding: %v", err)
	}
	return text
}

func newStore(t *testing.T, root string) library.VectorStore {
	t.Helper()
	store, err := embedded.New(embedded.Config{Root: root})
	if err != nil {
		t.Fatalf("New(%s): %v", root, err)
	}
	return store
}

func newBoundStore(t *testing.T) (library.VectorStore, string) {
	t.Helper()
	root := filepath.Join(t.TempDir(), "pool")
	store := newStore(t, root)
	if err := store.BindCatalog(context.Background(), encodeBinding(t, catalogUUIDA, "host-a", testDimension)); err != nil {
		t.Fatalf("BindCatalog: %v", err)
	}
	return store, root
}

func randomVector(random *rand.Rand) []float32 {
	values := make([]float32, testDimension)
	for index := range values {
		values[index] = float32(random.NormFloat64())
	}
	return values
}

func newRecord(id string, values []float32) library.VectorRecord {
	return library.VectorRecord{
		ID:             id,
		IdentityDigest: "digest-" + id,
		Checksum:       vectorcodec.Checksum(values),
		Values:         values,
	}
}

func identityOf(record library.VectorRecord) library.VectorIdentity {
	return library.VectorIdentity{
		ID:             record.ID,
		IdentityDigest: record.IdentityDigest,
		Checksum:       record.Checksum,
	}
}

func putRecords(t *testing.T, store library.VectorStore, records []library.VectorRecord) {
	t.Helper()
	for _, record := range records {
		if err := store.PutCanonical(context.Background(), record); err != nil {
			t.Fatalf("PutCanonical(%s): %v", record.ID, err)
		}
	}
}

// vectorFiles returns every regular file under root except the binding file.
func vectorFiles(t *testing.T, root string) []string {
	t.Helper()
	var paths []string
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.Type().IsRegular() && path != filepath.Join(root, "binding.json") {
			paths = append(paths, path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", root, err)
	}
	return paths
}

func readFile(t *testing.T, path string) []byte {
	t.Helper()
	content, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return content
}

func independentCosine(left []float32, right []float32) float64 {
	var dotProduct, leftSquares, rightSquares float64
	for index := range left {
		leftValue := float64(left[index])
		rightValue := float64(right[index])
		dotProduct += leftValue * rightValue
		leftSquares += leftValue * leftValue
		rightSquares += rightValue * rightValue
	}
	return dotProduct / (math.Sqrt(leftSquares) * math.Sqrt(rightSquares))
}

func TestNewValidatesRootAndCreatesDirectory(t *testing.T) {
	t.Parallel()
	parent := t.TempDir()
	for _, root := range []string{"", "relative/pool", parent + "/pool/../pool", parent + "/pool/"} {
		if _, err := embedded.New(embedded.Config{Root: root}); !errors.Is(err, library.ErrInvalidRequest) {
			t.Fatalf("New(%q) error = %v, want ErrInvalidRequest", root, err)
		}
	}
	root := filepath.Join(parent, "pool")
	store := newStore(t, root)
	info, err := os.Stat(root)
	if err != nil {
		t.Fatalf("stat root: %v", err)
	}
	if !info.IsDir() || info.Mode().Perm() != 0o700 {
		t.Fatalf("root mode = %v, want a directory with mode 0700", info.Mode())
	}
	if got := store.PoolIdentity(); got != "embedded:"+root {
		t.Fatalf("PoolIdentity = %q, want %q", got, "embedded:"+root)
	}
}

func TestBindCatalogRejectsSecondCatalogUUID(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	root := filepath.Join(t.TempDir(), "pool")
	first := newStore(t, root)
	second := newStore(t, root)
	firstBinding := encodeBinding(t, catalogUUIDA, "host-a", testDimension)
	if err := first.BindCatalog(ctx, firstBinding); err != nil {
		t.Fatalf("first BindCatalog: %v", err)
	}
	err := second.BindCatalog(ctx, encodeBinding(t, catalogUUIDB, "host-b", testDimension))
	if !errors.Is(err, library.ErrStoreMismatch) {
		t.Fatalf("second BindCatalog error = %v, want ErrStoreMismatch", err)
	}
	if got := string(readFile(t, filepath.Join(root, "binding.json"))); got != firstBinding {
		t.Fatalf("binding file = %q, want the first binding %q", got, firstBinding)
	}
	values := randomVector(rand.New(rand.NewPCG(1, 1)))
	if err := second.PutCanonical(ctx, newRecord("v1", values)); !errors.Is(err, library.ErrInvalidRequest) {
		t.Fatalf("PutCanonical on the losing store error = %v, want ErrInvalidRequest", err)
	}
	if files := vectorFiles(t, root); len(files) != 0 {
		t.Fatalf("losing store wrote vector files %v", files)
	}
}

func TestBindCatalogRaceHasOneWinner(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	for attempt := range bindRaceAttempts {
		root := filepath.Join(t.TempDir(), fmt.Sprintf("pool-%d", attempt))
		stores := []library.VectorStore{newStore(t, root), newStore(t, root)}
		bindings := []string{
			encodeBinding(t, catalogUUIDA, "host-a", testDimension),
			encodeBinding(t, catalogUUIDB, "host-b", testDimension),
		}
		results := make([]error, len(stores))
		start := make(chan struct{})
		var group sync.WaitGroup
		for index := range stores {
			group.Go(func() {
				<-start
				results[index] = stores[index].BindCatalog(ctx, bindings[index])
			})
		}
		close(start)
		group.Wait()

		winner := -1
		for index, err := range results {
			switch {
			case err == nil && winner < 0:
				winner = index
			case err == nil:
				t.Fatalf("attempt %d: both BindCatalog calls succeeded", attempt)
			case !errors.Is(err, library.ErrStoreMismatch):
				t.Fatalf("attempt %d: BindCatalog %d error = %v, want ErrStoreMismatch", attempt, index, err)
			}
		}
		if winner < 0 {
			t.Fatalf("attempt %d: no BindCatalog call succeeded: %v", attempt, results)
		}
		if got := string(readFile(t, filepath.Join(root, "binding.json"))); got != bindings[winner] {
			t.Fatalf("attempt %d: binding file = %q, want the winning binding %q", attempt, got, bindings[winner])
		}
	}
}

func TestBindCatalogAcceptsSameUUIDAndRejectsOtherDimension(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	root := filepath.Join(t.TempDir(), "pool")
	if err := newStore(t, root).BindCatalog(ctx, encodeBinding(t, catalogUUIDA, "host-a", testDimension)); err != nil {
		t.Fatalf("first BindCatalog: %v", err)
	}
	rebound := newStore(t, root)
	if err := rebound.BindCatalog(ctx, encodeBinding(t, catalogUUIDA, "host-b", testDimension)); err != nil {
		t.Fatalf("BindCatalog with the same UUID: %v", err)
	}
	values := randomVector(rand.New(rand.NewPCG(2, 2)))
	if err := rebound.PutCanonical(ctx, newRecord("v1", values)); err != nil {
		t.Fatalf("PutCanonical after rebind: %v", err)
	}
	err := newStore(t, root).BindCatalog(ctx, encodeBinding(t, catalogUUIDA, "host-a", testDimension+1))
	if !errors.Is(err, library.ErrStoreMismatch) {
		t.Fatalf("BindCatalog with another dimension error = %v, want ErrStoreMismatch", err)
	}
}

func TestPutCanonicalThenVerifyStrong(t *testing.T) {
	t.Parallel()
	store, _ := newBoundStore(t)
	random := rand.New(rand.NewPCG(3, 3))
	records := make([]library.VectorRecord, 0, 16)
	identities := make([]library.VectorIdentity, 0, 16)
	for index := range 16 {
		record := newRecord(fmt.Sprintf("%064x", index), randomVector(random))
		records = append(records, record)
		identities = append(identities, identityOf(record))
	}
	putRecords(t, store, records)
	if err := store.VerifyStrong(context.Background(), identities); err != nil {
		t.Fatalf("VerifyStrong: %v", err)
	}
}

// isFanOutName reports whether name is two lowercase hex characters.
func isFanOutName(name string) bool {
	if len(name) != 2 {
		return false
	}
	for _, character := range name {
		if !strings.ContainsRune("0123456789abcdef", character) {
			return false
		}
	}
	return true
}

// vectorDirectoryCounts walks root/vectors and returns the number of vector
// files in each fan-out directory, keyed by the path relative to root/vectors.
// It fails t when a file is not at <2 hex>/<2 hex>/<id>.vec for an ID in ids.
func vectorDirectoryCounts(t *testing.T, root string, ids map[string]struct{}) map[string]int {
	t.Helper()
	vectorsRoot := filepath.Join(root, "vectors")
	counts := make(map[string]int)
	for _, path := range vectorFiles(t, vectorsRoot) {
		relative, err := filepath.Rel(vectorsRoot, path)
		if err != nil {
			t.Fatalf("relative path of %s: %v", path, err)
		}
		parts := strings.Split(relative, string(filepath.Separator))
		if len(parts) != 3 || !isFanOutName(parts[0]) || !isFanOutName(parts[1]) {
			t.Fatalf("vector file %s is not at vectors/<2 hex>/<2 hex>/<id>.vec", relative)
		}
		id, found := strings.CutSuffix(parts[2], ".vec")
		if _, known := ids[id]; !found || !known {
			t.Fatalf("vector file %s does not name a written vector ID", relative)
		}
		counts[filepath.Join(parts[0], parts[1])]++
	}
	return counts
}

func TestLibraryVectorIDsSpreadAcrossFanOutDirectories(t *testing.T) {
	t.Parallel()
	store, root := newBoundStore(t)
	random := rand.New(rand.NewPCG(13, 13))
	records := make([]library.VectorRecord, 0, fanOutVectorCount)
	identities := make([]library.VectorIdentity, 0, fanOutVectorCount)
	ids := make(map[string]struct{}, fanOutVectorCount)
	for range fanOutVectorCount {
		id := fmt.Sprintf("v1_%016x%016x", random.Uint64(), random.Uint64())
		record := newRecord(id, randomVector(random))
		records = append(records, record)
		identities = append(identities, identityOf(record))
		ids[id] = struct{}{}
	}
	putRecords(t, store, records)
	if err := store.VerifyStrong(context.Background(), identities); err != nil {
		t.Fatalf("VerifyStrong: %v", err)
	}

	counts := vectorDirectoryCounts(t, root, ids)
	firstLevel := make(map[string]struct{})
	fileCount := 0
	for directory, count := range counts {
		firstLevel[filepath.Dir(directory)] = struct{}{}
		fileCount += count
		if count == fanOutVectorCount {
			t.Fatalf("directory vectors/%s contains all %d vector files", directory, count)
		}
	}
	if fileCount != fanOutVectorCount {
		t.Fatalf("found %d vector files, want %d", fileCount, fanOutVectorCount)
	}
	if len(firstLevel) < 2 || len(counts) < 2 {
		t.Fatalf(
			"vector files occupy %d first-level and %d second-level directories, want more than one of each: %v",
			len(firstLevel),
			len(counts),
			counts,
		)
	}
}

func TestPutCanonicalReplayDoesNotRewrite(t *testing.T) {
	t.Parallel()
	store, root := newBoundStore(t)
	record := newRecord("replayed", randomVector(rand.New(rand.NewPCG(4, 4))))
	putRecords(t, store, []library.VectorRecord{record})
	files := vectorFiles(t, root)
	if len(files) != 1 {
		t.Fatalf("vector files = %v, want one", files)
	}
	before, err := os.Stat(files[0])
	if err != nil {
		t.Fatalf("stat vector file: %v", err)
	}
	putRecords(t, store, []library.VectorRecord{record})
	after, err := os.Stat(files[0])
	if err != nil {
		t.Fatalf("stat vector file after replay: %v", err)
	}
	if !os.SameFile(before, after) || !after.ModTime().Equal(before.ModTime()) {
		t.Fatal("replayed PutCanonical replaced the vector file")
	}
}

func TestPutCanonicalConflictKeepsStoredVector(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store, root := newBoundStore(t)
	random := rand.New(rand.NewPCG(5, 5))
	original := newRecord("conflict", randomVector(random))
	putRecords(t, store, []library.VectorRecord{original})
	stored := readFile(t, vectorFiles(t, root)[0])

	otherValues := newRecord("conflict", randomVector(random))
	otherDigest := original
	otherDigest.IdentityDigest = "digest-other"
	for _, conflicting := range []library.VectorRecord{otherValues, otherDigest} {
		if err := store.PutCanonical(ctx, conflicting); !errors.Is(err, library.ErrVectorCorrupt) {
			t.Fatalf("conflicting PutCanonical error = %v, want ErrVectorCorrupt", err)
		}
	}
	if got := readFile(t, vectorFiles(t, root)[0]); string(got) != string(stored) {
		t.Fatal("conflicting PutCanonical changed the stored vector file")
	}
	if err := store.VerifyStrong(ctx, []library.VectorIdentity{identityOf(original)}); err != nil {
		t.Fatalf("VerifyStrong of the original identity: %v", err)
	}
}

func TestFlippedValueByteIsCorrupt(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store, root := newBoundStore(t)
	random := rand.New(rand.NewPCG(6, 6))
	record := newRecord("flipped", randomVector(random))
	putRecords(t, store, []library.VectorRecord{record})
	path := vectorFiles(t, root)[0]
	content := readFile(t, path)
	content[len(content)-1] ^= 0x01
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatalf("write flipped vector file: %v", err)
	}

	if err := store.VerifyStrong(ctx, []library.VectorIdentity{identityOf(record)}); !errors.Is(err, library.ErrVectorCorrupt) {
		t.Fatalf("VerifyStrong error = %v, want ErrVectorCorrupt", err)
	}
	if _, err := store.ScoreExact(ctx, randomVector(random), []string{record.ID}); !errors.Is(err, library.ErrVectorCorrupt) {
		t.Fatalf("ScoreExact error = %v, want ErrVectorCorrupt", err)
	}
}

func TestMissingVectorIsReported(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store, _ := newBoundStore(t)
	random := rand.New(rand.NewPCG(7, 7))
	present := newRecord("present", randomVector(random))
	putRecords(t, store, []library.VectorRecord{present})
	missing := newRecord("absent-id", randomVector(random))

	err := store.VerifyStrong(ctx, []library.VectorIdentity{identityOf(present), identityOf(missing)})
	if !errors.Is(err, library.ErrVectorMissing) || !strings.Contains(err.Error(), missing.ID) {
		t.Fatalf("VerifyStrong error = %v, want ErrVectorMissing naming %s", err, missing.ID)
	}
	scores, err := store.ScoreExact(ctx, randomVector(random), []string{present.ID, missing.ID})
	if !errors.Is(err, library.ErrVectorMissing) || scores != nil {
		t.Fatalf("ScoreExact = %v, %v, want no scores and ErrVectorMissing", scores, err)
	}
}

func TestScoreExactMatchesIndependentCosine(t *testing.T) {
	t.Parallel()
	store, _ := newBoundStore(t)
	random := rand.New(rand.NewPCG(8, 8))
	records := make([]library.VectorRecord, 0, 24)
	byID := make(map[string][]float32, 24)
	for index := range 24 {
		record := newRecord(fmt.Sprintf("vector-%02d", index), randomVector(random))
		records = append(records, record)
		byID[record.ID] = record.Values
	}
	putRecords(t, store, records)
	ids := make([]string, 0, len(records))
	for _, index := range random.Perm(len(records)) {
		ids = append(ids, records[index].ID)
	}
	query := randomVector(random)

	scores, err := store.ScoreExact(context.Background(), query, ids)
	if err != nil {
		t.Fatalf("ScoreExact: %v", err)
	}
	if len(scores) != len(ids) {
		t.Fatalf("ScoreExact returned %d scores, want %d", len(scores), len(ids))
	}
	for index, score := range scores {
		if score.ID != ids[index] {
			t.Fatalf("score %d ID = %s, want %s", index, score.ID, ids[index])
		}
		want := independentCosine(query, byID[score.ID])
		if math.Abs(score.Score-want) > scoreTolerance {
			t.Fatalf("score for %s = %.17g, want %.17g", score.ID, score.Score, want)
		}
	}
}

func TestInvalidVectorsAreRejected(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store, root := newBoundStore(t)
	random := rand.New(rand.NewPCG(9, 9))
	stored := newRecord("stored", randomVector(random))
	putRecords(t, store, []library.VectorRecord{stored})

	invalid := map[string][]float32{
		"nan":           randomVector(random),
		"infinite":      randomVector(random),
		"zero norm":     make([]float32, testDimension),
		"short":         randomVector(random)[:testDimension-1],
		"negative-infs": randomVector(random),
	}
	invalid["nan"][3] = float32(math.NaN())
	invalid["infinite"][5] = float32(math.Inf(1))
	invalid["negative-infs"][0] = float32(math.Inf(-1))
	for name, values := range invalid {
		if err := store.PutCanonical(ctx, newRecord("invalid", values)); !errors.Is(err, library.ErrInvalidRequest) {
			t.Fatalf("PutCanonical with %s values error = %v, want ErrInvalidRequest", name, err)
		}
		if _, err := store.ScoreExact(ctx, values, []string{stored.ID}); !errors.Is(err, library.ErrInvalidRequest) {
			t.Fatalf("ScoreExact with %s query error = %v, want ErrInvalidRequest", name, err)
		}
	}
	wrongChecksum := newRecord("wrong-checksum", randomVector(random))
	wrongChecksum.Checksum = stored.Checksum
	if err := store.PutCanonical(ctx, wrongChecksum); !errors.Is(err, library.ErrInvalidRequest) {
		t.Fatalf("PutCanonical with a wrong checksum error = %v, want ErrInvalidRequest", err)
	}
	if files := vectorFiles(t, root); len(files) != 1 {
		t.Fatalf("vector files = %v, want only the valid vector", files)
	}
}

func TestInvalidIDsAreRejected(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store, _ := newBoundStore(t)
	random := rand.New(rand.NewPCG(10, 10))
	values := randomVector(random)
	for _, id := range []string{"", "a/b", `a\b`, "..", "x..y"} {
		if err := store.PutCanonical(ctx, newRecord(id, values)); !errors.Is(err, library.ErrInvalidRequest) {
			t.Fatalf("PutCanonical(%q) error = %v, want ErrInvalidRequest", id, err)
		}
		if _, err := store.ScoreExact(ctx, values, []string{id}); !errors.Is(err, library.ErrInvalidRequest) {
			t.Fatalf("ScoreExact(%q) error = %v, want ErrInvalidRequest", id, err)
		}
	}
	record := newRecord("repeated", values)
	putRecords(t, store, []library.VectorRecord{record})
	if _, err := store.ScoreExact(ctx, values, []string{record.ID, record.ID}); !errors.Is(err, library.ErrInvalidRequest) {
		t.Fatalf("ScoreExact with a repeated ID error = %v, want ErrInvalidRequest", err)
	}
}

func TestUnboundStoreRejectsVectorCalls(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newStore(t, filepath.Join(t.TempDir(), "pool"))
	values := randomVector(rand.New(rand.NewPCG(11, 11)))
	record := newRecord("unbound", values)
	if err := store.PutCanonical(ctx, record); !errors.Is(err, library.ErrInvalidRequest) {
		t.Fatalf("PutCanonical error = %v, want ErrInvalidRequest", err)
	}
	if err := store.VerifyStrong(ctx, []library.VectorIdentity{identityOf(record)}); !errors.Is(err, library.ErrInvalidRequest) {
		t.Fatalf("VerifyStrong error = %v, want ErrInvalidRequest", err)
	}
	if _, err := store.ScoreExact(ctx, values, []string{record.ID}); !errors.Is(err, library.ErrInvalidRequest) {
		t.Fatalf("ScoreExact error = %v, want ErrInvalidRequest", err)
	}
}

func TestScoreExactReturnsContextError(t *testing.T) {
	t.Parallel()
	store, _ := newBoundStore(t)
	random := rand.New(rand.NewPCG(12, 12))
	record := newRecord("cancelled", randomVector(random))
	putRecords(t, store, []library.VectorRecord{record})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	scores, err := store.ScoreExact(ctx, randomVector(random), []string{record.ID})
	if !errors.Is(err, context.Canceled) || scores != nil {
		t.Fatalf("ScoreExact = %v, %v, want no scores and context.Canceled", scores, err)
	}
}
