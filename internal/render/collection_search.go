package render

import (
	"fmt"
	"strings"

	"goodkind.io/lm-semantic-search/internal/view"
)

// CollectionSearch formats typed collection search results.
func CollectionSearch(searchView view.CollectionSearchView) string {
	if len(searchView.Results) == 0 {
		return fmt.Sprintf("🔍 No collection results found for query: %q in collection '%s'", searchView.Query, searchView.CollectionID)
	}

	formatted := make([]string, 0, len(searchView.Results))
	for index, result := range searchView.Results {
		formatted = append(formatted, fmt.Sprintf(
			"%d. Collection row [%s]\n   Row key: %s\n   Score: %.4f\n   Content:\n```\n%s\n```",
			index+1,
			searchView.CollectionID,
			result.RowKey,
			result.Score,
			strings.TrimSpace(truncateContent(result.Content, 5000)),
		))
	}

	header := fmt.Sprintf("🔍 Found %d collection results for query: %q in collection '%s'", len(searchView.Results), searchView.Query, searchView.CollectionID)
	return header + "\n\n" + strings.Join(formatted, "\n\n")
}

// CollectionItemState formats one item's indexed fingerprint.
func CollectionItemState(stateView view.CollectionItemStateView) string {
	if stateView.Fingerprint == "" {
		return fmt.Sprintf("Item '%s' in collection '%s' is not indexed", stateView.ItemID, stateView.CollectionID)
	}
	return fmt.Sprintf("Item '%s' in collection '%s' is indexed at fingerprint %s", stateView.ItemID, stateView.CollectionID, stateView.Fingerprint)
}
