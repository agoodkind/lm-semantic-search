package localvec

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestReadRowsToleratesTornTrailingLine proves that a truncated final line,
// which an append interrupted by a crash can leave, does not make every earlier
// committed row unreadable. Only the torn line is dropped.
func TestReadRowsToleratesTornTrailingLine(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "collection.jsonl")
	valid := `{"id":"chunk_a","relativePath":"a.go","content":"a","contentVectorKey":"ka","vector":[1,0]}` + "\n"
	valid += `{"id":"chunk_b","relativePath":"b.go","content":"b","contentVectorKey":"kb","vector":[0,1]}` + "\n"
	torn := `{"id":"chunk_c","relativePath":"c.go","conte`
	if err := os.WriteFile(path, []byte(valid+torn), 0o600); err != nil {
		t.Fatalf("write collection: %v", err)
	}

	rows, exists, healed, err := readRows(path)
	if err != nil {
		t.Fatalf("readRows returned error on torn trailing line: %v", err)
	}
	if !exists {
		t.Fatal("readRows reported collection absent")
	}
	if !healed {
		t.Fatal("readRows did not report the torn line as healed")
	}
	if len(rows) != 2 {
		t.Fatalf("rows = %d, want 2 (torn final line dropped)", len(rows))
	}
	if rows[0].ID != "chunk_a" || rows[1].ID != "chunk_b" {
		t.Fatalf("unexpected rows: %+v", rows)
	}
}

// TestReadRowsFailsOnMidFileCorruption proves that a decode failure on a line
// that is not the last one is genuine corruption and remains a hard error,
// distinct from a torn trailing line.
func TestReadRowsFailsOnMidFileCorruption(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "collection.jsonl")
	corrupt := `{"id":"chunk_a"` + "\n"
	corrupt += `{"id":"chunk_b","relativePath":"b.go","content":"b","contentVectorKey":"kb","vector":[0,1]}` + "\n"
	if err := os.WriteFile(path, []byte(corrupt), 0o600); err != nil {
		t.Fatalf("write collection: %v", err)
	}

	if _, _, _, err := readRows(path); err == nil {
		t.Fatal("readRows accepted mid-file corruption, want error")
	}
}

// TestLoadHealsTornTailOnDisk proves the torn trailing fragment is physically
// removed when a collection loads, so a later append cannot land after it and
// corrupt the file. It loads a collection with a torn tail through the store,
// then reads the raw file back and confirms the fragment is gone.
func TestLoadHealsTornTailOnDisk(t *testing.T) {
	t.Parallel()
	root := filepath.Join(t.TempDir(), "localvec")
	if err := os.MkdirAll(root, 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	collectionPath := filepath.Join(root, "collection")
	if err := os.MkdirAll(collectionPath, 0o700); err != nil {
		t.Fatalf("mkdir collection: %v", err)
	}
	rows := []row{{
		ID:               "chunk_a",
		RelativePath:     "a.go",
		Content:          "a",
		ContentVectorKey: "ka",
		Vector:           []float32{1, 0},
	}}
	if err := assignLabels(rows); err != nil {
		t.Fatalf("assignLabels: %v", err)
	}
	vectorIndex, err := buildVectorIndex(rows, 2)
	if err != nil {
		t.Fatalf("buildVectorIndex: %v", err)
	}
	defer vectorIndex.Close()
	if err := vectorIndex.Save(filepath.Join(collectionPath, indexFileName)); err != nil {
		t.Fatalf("save index: %v", err)
	}
	path := filepath.Join(collectionPath, metadataFileName)
	if err := rewriteRows(path, rows); err != nil {
		t.Fatalf("write metadata: %v", err)
	}
	torn := `{"id":"chunk_b","relativePath":"b.go","conte`
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		t.Fatalf("open metadata: %v", err)
	}
	if _, err := file.WriteString(torn); err != nil {
		file.Close()
		t.Fatalf("append torn metadata: %v", err)
	}
	if err := file.Close(); err != nil {
		t.Fatalf("close metadata: %v", err)
	}

	loaded := newCollection("collection", collectionPath)
	loaded.mutex.Lock()
	if err := loaded.loadLocked(); err != nil {
		loaded.mutex.Unlock()
		t.Fatalf("loadLocked: %v", err)
	}
	loaded.mutex.Unlock()

	reread, _, healed, err := readRows(path)
	if err != nil {
		t.Fatalf("reread after heal: %v", err)
	}
	if healed {
		t.Fatal("torn tail still present on disk after load heal")
	}
	if len(reread) != 1 || reread[0].ID != "chunk_a" {
		t.Fatalf("healed file rows = %+v, want only chunk_a", reread)
	}
}

// TestNormalizeVectorLargeNormKeepsDirection proves the normalization divides in
// float64 so a vector whose norm exceeds the float32 range is not flattened to
// zero. A float32 norm would overflow to +Inf and zero every component.
func TestNormalizeVectorLargeNormKeepsDirection(t *testing.T) {
	t.Parallel()
	normalized, err := normalizeVector([]float32{math.MaxFloat32, math.MaxFloat32})
	if err != nil {
		t.Fatalf("normalizeVector: %v", err)
	}
	var squaredNorm float64
	for _, value := range normalized {
		squaredNorm += float64(value) * float64(value)
	}
	if math.Abs(squaredNorm-1.0) > 1e-3 {
		t.Fatalf("normalized vector not unit length: squaredNorm=%v components=%v", squaredNorm, normalized)
	}
}
