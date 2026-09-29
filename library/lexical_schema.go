package library

// lexicalSchemaStatements create the lexical tables and their indexes. The
// catalog migration runs them in order inside its schema transaction.
//
// lexical_content and lexical_terms store the analyzed form of one distinct
// SearchText once per analyzer identity. lexical_occurrences maps each
// committed occurrence to its content. lexical_stats and lexical_df store the
// occurrence-weighted corpus statistics of each namespace: an occurrence adds 1
// to the corpus size, its document length to the token total, and 1 to the
// document frequency of each distinct term hash, even when another occurrence
// has identical text.
var lexicalSchemaStatements = []string{
	`CREATE TABLE IF NOT EXISTS lexical_content (
		search_hash TEXT PRIMARY KEY,
		analyzer_identity TEXT NOT NULL,
		document_length INTEGER NOT NULL CHECK (document_length >= 0)
	) STRICT, WITHOUT ROWID`,
	`CREATE TABLE IF NOT EXISTS lexical_terms (
		search_hash TEXT NOT NULL REFERENCES lexical_content (search_hash),
		term_hash INTEGER NOT NULL CHECK (term_hash >= 0 AND term_hash < 4294967295),
		tf INTEGER NOT NULL CHECK (tf >= 1 AND tf <= 16777216),
		PRIMARY KEY (search_hash, term_hash)
	) STRICT, WITHOUT ROWID`,
	`CREATE INDEX IF NOT EXISTS lexical_terms_by_term ON lexical_terms (term_hash, search_hash)`,
	`CREATE TABLE IF NOT EXISTS lexical_occurrences (
		namespace TEXT NOT NULL,
		owner_id TEXT NOT NULL,
		row_key TEXT NOT NULL,
		search_hash TEXT NOT NULL REFERENCES lexical_content (search_hash),
		PRIMARY KEY (namespace, owner_id, row_key)
	) STRICT, WITHOUT ROWID`,
	`CREATE INDEX IF NOT EXISTS lexical_occurrences_by_content ON lexical_occurrences (search_hash)`,
	`CREATE TABLE IF NOT EXISTS lexical_stats (
		namespace TEXT PRIMARY KEY,
		generation INTEGER NOT NULL CHECK (generation >= 0),
		corpus_size INTEGER NOT NULL CHECK (corpus_size >= 0),
		total_tokens INTEGER NOT NULL CHECK (total_tokens >= 0)
	) STRICT, WITHOUT ROWID`,
	`CREATE TABLE IF NOT EXISTS lexical_df (
		namespace TEXT NOT NULL,
		term_hash INTEGER NOT NULL,
		df INTEGER NOT NULL CHECK (df > 0),
		PRIMARY KEY (namespace, term_hash)
	) STRICT, WITHOUT ROWID`,
}
