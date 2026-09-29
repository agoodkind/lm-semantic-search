package library

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"sort"
	"strconv"
	"strings"
)

// vectorIdentityVersion prefixes every vector identity. A change to the
// identity encoding needs a new version string.
const vectorIdentityVersion = "lms-vector-v1"

// documentRole is the embedding role of every stored vector. Query embeddings
// are never stored.
const documentRole = "document"

// vectorIDHexLength is the number of identity digest characters in a vector
// ID.
const vectorIDHexLength = 32

// vectorIdentity is the full identity of one canonical vector.
type vectorIdentity struct {
	id            string
	digest        string
	inputHash     string
	inputBytes    []byte
	modelIdentity string
}

// newVectorIdentity computes the identity of the exact embedding input under
// the descriptor's model, revision, dimension, and normalization.
func newVectorIdentity(descriptor StoreDescriptor, embeddingInput string) vectorIdentity {
	modelIdentity := descriptor.EmbeddingModel + "@" + descriptor.EmbeddingRevision
	var header strings.Builder
	header.WriteString(vectorIdentityVersion)
	header.WriteString("\nmodel=")
	header.WriteString(modelIdentity)
	header.WriteString("\ndimension=")
	header.WriteString(strconv.Itoa(descriptor.Dimension))
	header.WriteString("\nnormalization=")
	header.WriteString(descriptor.Normalization)
	header.WriteString("\nrole=")
	header.WriteString(documentRole)
	header.WriteString("\n\n")
	identityBytes := append([]byte(header.String()), embeddingInput...)
	digest := hexSHA256(identityBytes)
	return vectorIdentity{
		id:            "v1_" + digest[:vectorIDHexLength],
		digest:        digest,
		inputHash:     hexSHA256([]byte(embeddingInput)),
		inputBytes:    []byte(embeddingInput),
		modelIdentity: modelIdentity,
	}
}

// namespaceRecord is the canonical stored form of one [NamespaceSpec].
type namespaceRecord struct {
	ID      string          `json:"id"`
	Policy  NamespacePolicy `json:"policy"`
	Scalars []columnRecord  `json:"scalars"`
}

// columnRecord is the canonical stored form of one [ScalarColumn].
type columnRecord struct {
	Name      string     `json:"name"`
	Type      ScalarType `json:"type"`
	Nullable  bool       `json:"nullable"`
	Mutable   bool       `json:"mutable"`
	MaxLength int        `json:"max_length"`
}

func encodeNamespace(spec NamespaceSpec) (string, error) {
	record := namespaceRecord{ID: spec.ID, Policy: spec.Policy, Scalars: make([]columnRecord, 0, len(spec.Scalars))}
	for _, column := range spec.Scalars {
		record.Scalars = append(record.Scalars, columnRecord(column))
	}
	encoded, err := json.Marshal(record)
	if err != nil {
		slog.Error("encode namespace declaration failed", "namespace", spec.ID, "err", err)
		return "", fmt.Errorf("encode namespace %q declaration: %w", spec.ID, err)
	}
	return string(encoded), nil
}

func decodeNamespace(encoded string) (NamespaceSpec, error) {
	var record namespaceRecord
	if err := json.Unmarshal([]byte(encoded), &record); err != nil {
		slog.Error("decode namespace declaration failed", "err", err)
		return NamespaceSpec{}, fmt.Errorf("decode namespace declaration: %w", err)
	}
	spec := NamespaceSpec{ID: record.ID, Policy: record.Policy, Scalars: make([]ScalarColumn, 0, len(record.Scalars))}
	for _, column := range record.Scalars {
		spec.Scalars = append(spec.Scalars, ScalarColumn(column))
	}
	return spec, nil
}

// occurrenceRecord is the canonical stored form of one occurrence. JSON
// encoding sorts map keys, so equal occurrences encode to equal bytes.
type occurrenceRecord struct {
	RowKey         string                 `json:"row_key"`
	SortKey        string                 `json:"sort_key"`
	SourceText     string                 `json:"source_text"`
	SearchText     string                 `json:"search_text"`
	EmbeddingInput string                 `json:"embedding_input"`
	Scalars        map[string]scalarValue `json:"scalars"`
}

// scalarValue is the canonical stored form of one [ScalarValue].
type scalarValue struct {
	Type   ScalarType `json:"type"`
	Null   bool       `json:"null"`
	String string     `json:"string"`
	Bool   bool       `json:"bool"`
	Int64  int64      `json:"int64"`
}

func newOccurrenceRecord(occurrence Occurrence) occurrenceRecord {
	scalars := make(map[string]scalarValue, len(occurrence.Scalars))
	for name, value := range occurrence.Scalars {
		scalars[name] = scalarValue(value)
	}
	return occurrenceRecord{
		RowKey:         occurrence.RowKey,
		SortKey:        occurrence.SortKey,
		SourceText:     occurrence.SourceText,
		SearchText:     occurrence.SearchText,
		EmbeddingInput: occurrence.EmbeddingInput,
		Scalars:        scalars,
	}
}

func (record occurrenceRecord) occurrence() Occurrence {
	scalars := make(map[string]ScalarValue, len(record.Scalars))
	for name, value := range record.Scalars {
		scalars[name] = ScalarValue(value)
	}
	return Occurrence{
		RowKey:         record.RowKey,
		SortKey:        record.SortKey,
		SourceText:     record.SourceText,
		SearchText:     record.SearchText,
		EmbeddingInput: record.EmbeddingInput,
		Scalars:        scalars,
	}
}

// encodeOccurrence returns the canonical bytes of occurrence and their hex
// SHA-256.
func encodeOccurrence(occurrence Occurrence) ([]byte, string, error) {
	encoded, err := json.Marshal(newOccurrenceRecord(occurrence))
	if err != nil {
		slog.Error("encode occurrence failed", "row_key", occurrence.RowKey, "err", err)
		return nil, "", fmt.Errorf("encode occurrence %q: %w", occurrence.RowKey, err)
	}
	return encoded, hexSHA256(encoded), nil
}

// decodeOccurrence parses bytes produced by [encodeOccurrence].
func decodeOccurrence(encoded []byte) (Occurrence, error) {
	var record occurrenceRecord
	if err := json.Unmarshal(encoded, &record); err != nil {
		slog.Error("decode occurrence failed", "err", err)
		return Occurrence{}, fmt.Errorf("decode occurrence: %w", err)
	}
	return record.occurrence(), nil
}

// manifestEntry is one row of a generation manifest.
type manifestEntry struct {
	rowKey         string
	occurrenceHash string
}

// manifestHash returns the hex SHA-256 of entries sorted by row key.
func manifestHash(entries []manifestEntry) string {
	sorted := append([]manifestEntry(nil), entries...)
	sort.Slice(sorted, func(left int, right int) bool {
		return sorted[left].rowKey < sorted[right].rowKey
	})
	var manifest strings.Builder
	for _, entry := range sorted {
		manifest.WriteString(entry.rowKey)
		manifest.WriteByte(0)
		manifest.WriteString(entry.occurrenceHash)
		manifest.WriteByte('\n')
	}
	return hexSHA256([]byte(manifest.String()))
}

// SealRows returns the [GenerationSeal] of a complete owner generation. The
// seal contains the row count and the SHA-256 of the canonical occurrence
// encodings sorted by row key. A caller passes the seal of every staged row of
// one generation to CommitGeneration. Duplicate row keys return an error that
// wraps [ErrInvalidRequest].
func SealRows(rows []Occurrence) (GenerationSeal, error) {
	entries := make([]manifestEntry, 0, len(rows))
	seen := make(map[string]bool, len(rows))
	for _, row := range rows {
		if seen[row.RowKey] {
			return GenerationSeal{}, invalidRequest(fmt.Sprintf("seal rows: row key %q appears twice", row.RowKey))
		}
		seen[row.RowKey] = true
		_, occurrenceHash, err := encodeOccurrence(row)
		if err != nil {
			return GenerationSeal{}, err
		}
		entries = append(entries, manifestEntry{rowKey: row.RowKey, occurrenceHash: occurrenceHash})
	}
	return GenerationSeal{RowCount: uint64(len(rows)), ManifestHash: manifestHash(entries)}, nil
}

// generationFingerprint identifies one committed owner generation.
func generationFingerprint(key GenerationKey, manifest string) string {
	return hexSHA256([]byte(strings.Join([]string{
		"generation",
		key.Namespace,
		key.OwnerID,
		strconv.FormatUint(key.GenerationOrder, 10),
		key.IdempotencyToken,
		manifest,
	}, "\x00")))
}

// searchTextHash identifies equal lexical text across occurrences.
func searchTextHash(text string) string {
	return hexSHA256([]byte(text))
}

// sourceBlobID identifies equal source excerpts across occurrences.
func sourceBlobID(text string) string {
	return hexSHA256([]byte(text))
}

func hexSHA256(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
