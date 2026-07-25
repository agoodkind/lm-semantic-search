package localvec

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"strconv"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

const (
	rowIDHashLength  = 16
	jsonLineMaxBytes = 16 * 1024 * 1024
)

type row struct {
	Label                uint64    `json:"label"`
	ID                   string    `json:"id"`
	RelativePath         string    `json:"relativePath"`
	StartLine            int32     `json:"startLine"`
	EndLine              int32     `json:"endLine"`
	Language             string    `json:"language,omitempty"`
	FileExtension        string    `json:"fileExtension"`
	Content              string    `json:"content"`
	ContentVectorKey     string    `json:"contentVectorKey"`
	Vector               []float32 `json:"vector"`
	ConversationID       string    `json:"conversationId,omitempty"`
	ParentConversationID string    `json:"parentConversationId,omitempty"`
	MessageIndex         int32     `json:"messageIndex,omitempty"`
	Role                 string    `json:"role,omitempty"`
	TimestampUnix        int64     `json:"timestampUnix,omitempty"`
	WorkspaceRoot        string    `json:"workspaceRoot,omitempty"`
	Archived             bool      `json:"archived,omitempty"`
}

func newRow(chunk model.StoredChunk, vector []float32) (row, error) {
	normalized, err := normalizeVector(vector)
	if err != nil {
		slog.Error(
			"normalize local vector row failed",
			"relative_path",
			chunk.RelativePath,
			"err",
			err,
		)
		return row{}, fmt.Errorf("normalize vector for %s: %w", chunk.RelativePath, err)
	}
	return row{
		Label:                0,
		ID:                   generateRowID(chunk),
		RelativePath:         chunk.RelativePath,
		StartLine:            chunk.StartLine,
		EndLine:              chunk.EndLine,
		Language:             chunk.Language,
		FileExtension:        chunk.FileExtension,
		Content:              chunk.Content,
		ContentVectorKey:     semantic.ContentVectorKey(chunk.Content),
		Vector:               normalized,
		ConversationID:       chunk.ConversationID,
		ParentConversationID: chunk.ParentConversationID,
		MessageIndex:         chunk.MessageIndex,
		Role:                 chunk.Role,
		TimestampUnix:        chunk.TimestampUnix,
		WorkspaceRoot:        chunk.WorkspaceRoot,
		Archived:             chunk.Archived,
	}, nil
}

func assignLabels(rows []row) error {
	labels := make(map[uint64]string, len(rows))
	for index := range rows {
		if rows[index].Label == 0 {
			continue
		}
		existingID, found := labels[rows[index].Label]
		if found && existingID != rows[index].ID {
			return fmt.Errorf(
				"local vector label %d maps to both %s and %s",
				rows[index].Label,
				existingID,
				rows[index].ID,
			)
		}
		labels[rows[index].Label] = rows[index].ID
	}
	for index := range rows {
		if rows[index].Label != 0 {
			continue
		}
		rows[index].Label = labelForRowID(rows[index].ID, labels)
		labels[rows[index].Label] = rows[index].ID
	}
	return nil
}

func labelForRowID(rowID string, assigned map[uint64]string) uint64 {
	for attempt := uint64(0); ; attempt++ {
		input := []byte(rowID)
		if attempt > 0 {
			input = append(input, 0)
			input = strconv.AppendUint(input, attempt, 10)
		}
		sum := sha256.Sum256(input)
		label := binary.LittleEndian.Uint64(sum[:8])
		if label == 0 {
			continue
		}
		existingID, found := assigned[label]
		if !found || existingID == rowID {
			return label
		}
	}
}

func vectorDimensions(rows []row) (int, error) {
	if len(rows) == 0 {
		return 0, nil
	}
	dimensions := len(rows[0].Vector)
	if dimensions == 0 {
		return 0, fmt.Errorf("local vector row %s has an empty vector", rows[0].ID)
	}
	for _, stored := range rows[1:] {
		if len(stored.Vector) != dimensions {
			return 0, fmt.Errorf(
				"local vector row %s has %d dimensions, want %d",
				stored.ID,
				len(stored.Vector),
				dimensions,
			)
		}
	}
	return dimensions, nil
}

func (stored row) chunk(score float64) model.StoredChunk {
	return model.StoredChunk{
		Content:              stored.Content,
		RelativePath:         stored.RelativePath,
		StartLine:            stored.StartLine,
		EndLine:              stored.EndLine,
		Language:             stored.Language,
		FileExtension:        stored.FileExtension,
		ConversationID:       stored.ConversationID,
		ParentConversationID: stored.ParentConversationID,
		MessageIndex:         stored.MessageIndex,
		Role:                 stored.Role,
		TimestampUnix:        stored.TimestampUnix,
		WorkspaceRoot:        stored.WorkspaceRoot,
		Archived:             stored.Archived,
		Score:                score,
	}
}

func generateRowID(chunk model.StoredChunk) string {
	hashInput := fmt.Sprintf(
		"%s:%d:%d:%s",
		chunk.RelativePath,
		chunk.StartLine,
		chunk.EndLine,
		chunk.Content,
	)
	sum := sha256.Sum256([]byte(hashInput))
	return "chunk_" + hex.EncodeToString(sum[:])[:rowIDHashLength]
}

func normalizeVector(vector []float32) ([]float32, error) {
	if len(vector) == 0 {
		return nil, errors.New("embedding vector is empty")
	}
	var squaredNorm float64
	for _, value := range vector {
		floatValue := float64(value)
		if math.IsNaN(floatValue) || math.IsInf(floatValue, 0) {
			return nil, errors.New("embedding vector contains a non-finite value")
		}
		squaredNorm += floatValue * floatValue
	}
	if squaredNorm == 0 {
		return nil, errors.New("embedding vector has zero norm")
	}
	// Divide in float64 and narrow the result. Narrowing the norm to float32
	// first overflows to +Inf for a vector whose norm exceeds the float32 range,
	// which would normalize every component to zero and destroy the direction.
	norm := math.Sqrt(squaredNorm)
	normalized := make([]float32, 0, len(vector))
	for _, value := range vector {
		normalized = append(normalized, float32(float64(value)/norm))
	}
	return normalized, nil
}

// readRows loads a collection file. The returned healed flag reports that a torn
// trailing line was dropped, so the caller rewrites the file to the clean rows
// before any later append lands after the fragment on disk.
func readRows(path string) (rows []row, exists bool, healed bool, err error) {
	file, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return []row{}, false, false, nil
	}
	if err != nil {
		slog.Error("open local vector collection failed", "path", path, "err", err)
		return nil, false, false, fmt.Errorf("open local vector collection %s: %w", path, err)
	}
	defer file.Close()

	// Read every line into memory first so a decode failure can distinguish a
	// torn trailing line from real mid-file corruption. Writes are append-only,
	// so a crash mid-append can only truncate the final line. Dropping that one
	// line keeps every earlier committed row readable, which lets the next sync
	// re-embed the interrupted file instead of the whole collection becoming
	// unreadable. A decode failure on any earlier line is genuine corruption and
	// stays a hard error.
	lines := make([][]byte, 0)
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), jsonLineMaxBytes)
	for scanner.Scan() {
		line := make([]byte, len(scanner.Bytes()))
		copy(line, scanner.Bytes())
		lines = append(lines, line)
	}
	if scanErr := scanner.Err(); scanErr != nil {
		slog.Error("read local vector collection failed", "path", path, "err", scanErr)
		return nil, false, false, fmt.Errorf("read local vector collection %s: %w", path, scanErr)
	}

	rows = make([]row, 0, len(lines))
	for index, line := range lines {
		var stored row
		if decodeErr := json.Unmarshal(line, &stored); decodeErr != nil {
			isLastLine := index == len(lines)-1
			if isLastLine {
				slog.Warn("dropping torn trailing local vector row", "path", path, "err", decodeErr)
				healed = true
				break
			}
			slog.Error("decode local vector row failed", "path", path, "line", index, "err", decodeErr)
			return nil, false, false, fmt.Errorf("decode local vector row %d from %s: %w", index, path, decodeErr)
		}
		rows = append(rows, stored)
	}
	return rows, true, healed, nil
}

func rewriteRows(path string, rows []row) error {
	tempFile, err := os.CreateTemp(
		filepath.Dir(path),
		filepath.Base(path)+".tmp-*",
	)
	if err != nil {
		slog.Error("create local vector rewrite file failed", "path", path, "err", err)
		return fmt.Errorf("create local vector rewrite file for %s: %w", path, err)
	}
	tempPath := tempFile.Name()
	removeTemp := true
	defer func() {
		if removeTemp {
			_ = os.Remove(tempPath)
		}
	}()

	if err := tempFile.Chmod(0o600); err != nil {
		tempFile.Close()
		slog.Error("set local vector rewrite permissions failed", "path", path, "err", err)
		return fmt.Errorf("set local vector rewrite permissions for %s: %w", path, err)
	}
	encoder := json.NewEncoder(tempFile)
	for _, stored := range rows {
		if err := encoder.Encode(stored); err != nil {
			tempFile.Close()
			slog.Error("encode local vector rewrite row failed", "path", path, "err", err)
			return fmt.Errorf("encode local vector rewrite row for %s: %w", path, err)
		}
	}
	if err := tempFile.Sync(); err != nil {
		tempFile.Close()
		slog.Error("sync local vector rewrite file failed", "path", path, "err", err)
		return fmt.Errorf("sync local vector rewrite file for %s: %w", path, err)
	}
	if err := tempFile.Close(); err != nil {
		slog.Error("close local vector rewrite file failed", "path", path, "err", err)
		return fmt.Errorf("close local vector rewrite file for %s: %w", path, err)
	}
	if err := os.Rename(tempPath, path); err != nil {
		slog.Error("replace local vector collection failed", "path", path, "err", err)
		return fmt.Errorf("replace local vector collection %s: %w", path, err)
	}
	removeTemp = false
	return nil
}

func cloneRows(rows []row) []row {
	cloned := make([]row, 0, len(rows))
	for _, stored := range rows {
		copyRow := stored
		copyRow.Vector = append([]float32(nil), stored.Vector...)
		cloned = append(cloned, copyRow)
	}
	return cloned
}
