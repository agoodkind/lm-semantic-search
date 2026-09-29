package library

import (
	"fmt"
	"log/slog"
	"math"
	"path/filepath"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"
)

// Resource defaults for zero [Config] values. They bound resource use and are
// not latency targets.
const (
	defaultMaxBatchRows      = 256
	defaultMaxBatchBytes     = 8 << 20
	defaultQueryBlockSize    = 512
	defaultQueryWorkers      = 2
	defaultMaxTemporaryBytes = 1 << 30
	defaultMaxSnapshotBytes  = 256 << 20
	defaultSnapshotTTL       = 10 * time.Minute
	defaultQueryTimeout      = 30 * time.Second
	defaultBM25K1            = 1.2
	defaultBM25B             = 0.75
	defaultRRFK              = 60
	defaultSearchMode        = Hybrid
)

// scalarColumnNamePattern accepts a column name that starts with a letter or an
// underscore and continues with letters, digits, or underscores, up to 64
// bytes.
var scalarColumnNamePattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]{0,63}$`)

// Validate reports whether the configuration opens a library. It checks the
// store descriptor, every budget, the ranking settings, and the search mode,
// and returns an error that wraps [ErrInvalidRequest] for the first violation.
// It does not contact the vector backend or the embedder.
func (config Config) Validate() error {
	_, err := config.resolved()
	return err
}

// resolved returns the configuration with every zero budget replaced by its
// default, or the first validation error.
func (config Config) resolved() (Config, error) {
	if err := config.Store.Validate(); err != nil {
		return Config{}, err
	}
	if err := validateBudgets(config); err != nil {
		return Config{}, err
	}
	if err := validateRequestLimits(config); err != nil {
		return Config{}, err
	}
	if err := validateRanking(config); err != nil {
		return Config{}, err
	}

	resolved := config
	resolved.MaxBatchRows = defaultInt(config.MaxBatchRows, defaultMaxBatchRows)
	resolved.MaxBatchBytes = defaultInt64(config.MaxBatchBytes, defaultMaxBatchBytes)
	resolved.QueryBlockSize = defaultInt(config.QueryBlockSize, defaultQueryBlockSize)
	resolved.QueryWorkers = defaultInt(config.QueryWorkers, defaultQueryWorkers)
	resolved.MaxTemporaryBytes = defaultInt64(config.MaxTemporaryBytes, defaultMaxTemporaryBytes)
	resolved.MaxSnapshotBytes = defaultInt64(config.MaxSnapshotBytes, defaultMaxSnapshotBytes)
	resolved.SnapshotTTL = defaultDuration(config.SnapshotTTL, defaultSnapshotTTL)
	resolved.QueryTimeout = defaultDuration(config.QueryTimeout, defaultQueryTimeout)
	if config.SearchMode == 0 {
		resolved.SearchMode = defaultSearchMode
	}
	if config.BM25K1 == 0 {
		resolved.BM25K1 = defaultBM25K1
	}
	bm25B := defaultBM25B
	if config.BM25B != nil {
		bm25B = *config.BM25B
	}
	resolved.BM25B = &bm25B
	resolved.RRFK = defaultInt(config.RRFK, defaultRRFK)
	if config.AnalyzerIdentity == "" {
		resolved.AnalyzerIdentity = StandardAnalyzer
	}
	return resolved, nil
}

func validateBudgets(config Config) error {
	budgets := []struct {
		name  string
		value int64
	}{
		{name: "MaxBatchRows", value: int64(config.MaxBatchRows)},
		{name: "MaxBatchBytes", value: config.MaxBatchBytes},
		{name: "QueryBlockSize", value: int64(config.QueryBlockSize)},
		{name: "QueryWorkers", value: int64(config.QueryWorkers)},
		{name: "MaxTemporaryBytes", value: config.MaxTemporaryBytes},
		{name: "MaxSnapshotBytes", value: config.MaxSnapshotBytes},
		{name: "SnapshotTTL", value: int64(config.SnapshotTTL)},
		{name: "QueryTimeout", value: int64(config.QueryTimeout)},
	}
	for _, budget := range budgets {
		if budget.value < 0 {
			return invalidRequest(fmt.Sprintf(
				"config %s is %d; use zero for the default or a positive value",
				budget.name,
				budget.value,
			))
		}
	}
	return nil
}

func validateRequestLimits(config Config) error {
	limits := []struct {
		name  string
		value int
	}{
		{name: "MaxPageSize", value: config.MaxPageSize},
		{name: "MaxQueryBytes", value: config.MaxQueryBytes},
		{name: "MaxFilterDepth", value: config.MaxFilterDepth},
		{name: "MaxFilterValues", value: config.MaxFilterValues},
	}
	for _, limit := range limits {
		if limit.value < 0 {
			return invalidRequest(fmt.Sprintf(
				"config %s is %d; use zero to disable the limit or a positive value",
				limit.name,
				limit.value,
			))
		}
	}
	return nil
}

func validateRanking(config Config) error {
	switch config.SearchMode {
	case 0, Dense, Hybrid:
	default:
		return invalidRequest(fmt.Sprintf("config SearchMode %d is neither Dense nor Hybrid", config.SearchMode))
	}
	if math.IsNaN(config.BM25K1) || math.IsInf(config.BM25K1, 0) || config.BM25K1 < 0 {
		return invalidRequest(fmt.Sprintf("config BM25K1 %v must be finite and positive", config.BM25K1))
	}
	if config.BM25B != nil {
		bm25B := *config.BM25B
		if math.IsNaN(bm25B) || bm25B < 0 || bm25B > 1 {
			return invalidRequest(fmt.Sprintf("config BM25B %v must be between 0 and 1", bm25B))
		}
	}
	if config.RRFK < 0 {
		return invalidRequest(fmt.Sprintf("config RRFK %d must be positive", config.RRFK))
	}
	if config.AnalyzerIdentity != "" && config.AnalyzerIdentity != StandardAnalyzer {
		return invalidRequest(fmt.Sprintf(
			"config AnalyzerIdentity %q is not supported; use %q",
			config.AnalyzerIdentity,
			StandardAnalyzer,
		))
	}
	return nil
}

// Validate reports whether the descriptor identifies a pool and catalog. Both
// paths are absolute, clean, and distinct. The pool, model, revision, and
// normalization identities are nonempty without surrounding whitespace, and
// the dimension is positive. A violation returns an error that wraps
// [ErrInvalidRequest].
func (descriptor StoreDescriptor) Validate() error {
	if err := validateStorePath("CatalogPath", descriptor.CatalogPath); err != nil {
		return err
	}
	if err := validateStorePath("LockPath", descriptor.LockPath); err != nil {
		return err
	}
	if descriptor.CatalogPath == descriptor.LockPath {
		return invalidRequest(fmt.Sprintf("store descriptor LockPath %q equals CatalogPath", descriptor.LockPath))
	}
	identities := []struct {
		name  string
		value string
	}{
		{name: "PoolID", value: descriptor.PoolID},
		{name: "EmbeddingModel", value: descriptor.EmbeddingModel},
		{name: "EmbeddingRevision", value: descriptor.EmbeddingRevision},
		{name: "Normalization", value: descriptor.Normalization},
	}
	for _, identity := range identities {
		if err := validateIdentity("store descriptor "+identity.name, identity.value); err != nil {
			return err
		}
	}
	if descriptor.Dimension <= 0 {
		return invalidRequest(fmt.Sprintf("store descriptor Dimension %d must be positive", descriptor.Dimension))
	}
	return nil
}

func validateStorePath(name string, path string) error {
	if path == "" {
		return invalidRequest(fmt.Sprintf("store descriptor %s is empty", name))
	}
	if !filepath.IsAbs(path) {
		return invalidRequest(fmt.Sprintf("store descriptor %s %q is not absolute", name, path))
	}
	if filepath.Clean(path) != path {
		return invalidRequest(fmt.Sprintf(
			"store descriptor %s %q is not clean; use %q",
			name,
			path,
			filepath.Clean(path),
		))
	}
	return nil
}

func validateIdentity(name string, value string) error {
	if value == "" {
		return invalidRequest(name + " is empty")
	}
	if !utf8.ValidString(value) {
		return invalidRequest(name + " is not valid UTF-8")
	}
	if strings.TrimSpace(value) != value {
		return invalidRequest(fmt.Sprintf("%s %q has surrounding whitespace", name, value))
	}
	if strings.ContainsFunc(value, isControlRune) {
		return invalidRequest(fmt.Sprintf("%s %q contains a control character", name, value))
	}
	return nil
}

// Validate reports whether the declaration can be registered. The ID is a
// nonempty identity, the policy is [AppendOnly] or [ReplaceAllowed], and every
// column has a unique name, a declared type, and a maximum length that is
// positive for a [String] column and zero otherwise. A violation returns an
// error that wraps [ErrInvalidRequest].
func (spec NamespaceSpec) Validate() error {
	if err := validateIdentity("namespace ID", spec.ID); err != nil {
		return err
	}
	switch spec.Policy {
	case AppendOnly, ReplaceAllowed:
	default:
		return invalidRequest(fmt.Sprintf(
			"namespace %q policy %d is neither AppendOnly nor ReplaceAllowed",
			spec.ID,
			spec.Policy,
		))
	}
	seen := make(map[string]bool, len(spec.Scalars))
	for _, column := range spec.Scalars {
		if err := validateColumn(spec.ID, column); err != nil {
			return err
		}
		if seen[column.Name] {
			return invalidRequest(fmt.Sprintf("namespace %q declares column %q twice", spec.ID, column.Name))
		}
		seen[column.Name] = true
	}
	return nil
}

func validateColumn(namespace string, column ScalarColumn) error {
	if !scalarColumnNamePattern.MatchString(column.Name) {
		return invalidRequest(fmt.Sprintf(
			"namespace %q column name %q must match %s",
			namespace,
			column.Name,
			scalarColumnNamePattern,
		))
	}
	switch column.Type {
	case String:
		if column.MaxLength <= 0 {
			return invalidRequest(fmt.Sprintf(
				"namespace %q string column %q MaxLength %d must be positive",
				namespace,
				column.Name,
				column.MaxLength,
			))
		}
	case Bool, Int64:
		if column.MaxLength != 0 {
			return invalidRequest(fmt.Sprintf(
				"namespace %q column %q declares MaxLength %d for a non-string type",
				namespace,
				column.Name,
				column.MaxLength,
			))
		}
	default:
		return invalidRequest(fmt.Sprintf(
			"namespace %q column %q type %d is not String, Bool, or Int64",
			namespace,
			column.Name,
			column.Type,
		))
	}
	return nil
}

// ValidateOccurrence reports whether the occurrence can be written to the
// namespace. The row key is nonempty, every text field is valid UTF-8, the
// search text contains no NUL byte, the embedding input contains a
// non-whitespace character, and every scalar key is
// a declared column with a value of that column's type. A null value requires
// a nullable column. A violation returns an error that wraps
// [ErrInvalidRequest].
func (spec NamespaceSpec) ValidateOccurrence(occurrence Occurrence) error {
	if occurrence.RowKey == "" {
		return invalidRequest(fmt.Sprintf("namespace %q occurrence row key is empty", spec.ID))
	}
	texts := []struct {
		name  string
		value string
	}{
		{name: "RowKey", value: occurrence.RowKey},
		{name: "SortKey", value: occurrence.SortKey},
		{name: "SourceText", value: occurrence.SourceText},
		{name: "SearchText", value: occurrence.SearchText},
		{name: "EmbeddingInput", value: occurrence.EmbeddingInput},
	}
	for _, text := range texts {
		if !utf8.ValidString(text.value) {
			return invalidRequest(fmt.Sprintf(
				"namespace %q row %q %s is not valid UTF-8",
				spec.ID,
				occurrence.RowKey,
				text.name,
			))
		}
	}
	if strings.ContainsRune(occurrence.SearchText, 0) {
		return invalidRequest(fmt.Sprintf(
			"namespace %q row %q SearchText contains a NUL byte, which the analyzer reads as the end of the text",
			spec.ID,
			occurrence.RowKey,
		))
	}
	if strings.TrimSpace(occurrence.EmbeddingInput) == "" {
		return invalidRequest(fmt.Sprintf(
			"namespace %q row %q EmbeddingInput contains no non-whitespace character",
			spec.ID,
			occurrence.RowKey,
		))
	}
	columns := make(map[string]ScalarColumn, len(spec.Scalars))
	for _, column := range spec.Scalars {
		columns[column.Name] = column
	}
	for name, value := range occurrence.Scalars {
		column, declared := columns[name]
		if !declared {
			return invalidRequest(fmt.Sprintf(
				"namespace %q row %q sets undeclared column %q",
				spec.ID,
				occurrence.RowKey,
				name,
			))
		}
		if message := scalarValueViolation(column, value); message != "" {
			return invalidRequest(fmt.Sprintf("namespace %q row %q: %s", spec.ID, occurrence.RowKey, message))
		}
	}
	return nil
}

// scalarValueViolation describes why value does not satisfy column, or returns
// an empty string when it does.
func scalarValueViolation(column ScalarColumn, value ScalarValue) string {
	if value.Type != column.Type {
		return fmt.Sprintf("column %q value type %d does not match declared type %d", column.Name, value.Type, column.Type)
	}
	if value.Null {
		if !column.Nullable {
			return fmt.Sprintf("column %q is not nullable", column.Name)
		}
		if value.String != "" || value.Bool || value.Int64 != 0 {
			return fmt.Sprintf("column %q null value also sets a value field", column.Name)
		}
		return ""
	}
	switch column.Type {
	case String:
		if value.Bool || value.Int64 != 0 {
			return fmt.Sprintf("column %q string value also sets a bool or int64 field", column.Name)
		}
		if !utf8.ValidString(value.String) {
			return fmt.Sprintf("column %q value is not valid UTF-8", column.Name)
		}
		if len(value.String) > column.MaxLength {
			return fmt.Sprintf(
				"column %q value is %d bytes, over the declared %d",
				column.Name,
				len(value.String),
				column.MaxLength,
			)
		}
	case Bool:
		if value.String != "" || value.Int64 != 0 {
			return fmt.Sprintf("column %q bool value also sets a string or int64 field", column.Name)
		}
	case Int64:
		if value.String != "" || value.Bool {
			return fmt.Sprintf("column %q int64 value also sets a string or bool field", column.Name)
		}
	default:
		return fmt.Sprintf("column %q type %d is not String, Bool, or Int64", column.Name, column.Type)
	}
	return ""
}

// invalidRequest logs and returns a validation failure that wraps
// [ErrInvalidRequest].
func invalidRequest(message string) error {
	err := fmt.Errorf("%w: %s", ErrInvalidRequest, message)
	slog.Warn("library request rejected", "err", err)
	return err
}

func isControlRune(character rune) bool {
	return character < ' ' || character == 0x7f
}

func defaultInt(value int, fallback int) int {
	if value == 0 {
		return fallback
	}
	return value
}

func defaultInt64(value int64, fallback int64) int64 {
	if value == 0 {
		return fallback
	}
	return value
}

func defaultDuration(value time.Duration, fallback time.Duration) time.Duration {
	if value == 0 {
		return fallback
	}
	return value
}
