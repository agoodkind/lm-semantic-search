package collection

import (
	"strings"

	"goodkind.io/lm-semantic-search/internal/tshash"
)

const documentNamePrefix = "conv_chunks_"

// DocumentName returns the stored collection name for a caller's collection
// ID: the document prefix plus the first 8 hex characters of the MD5 of the
// trimmed ID. Stored collections already use this format. Do not change it.
func DocumentName(collectionID string) string {
	return documentNamePrefix + tshash.PathPrefix(strings.TrimSpace(collectionID))
}

// IsDocumentName reports whether a stored name is a document collection name.
func IsDocumentName(collectionName string) bool {
	return strings.HasPrefix(collectionName, documentNamePrefix)
}
