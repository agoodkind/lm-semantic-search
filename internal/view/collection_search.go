package view

// CollectionSearchView is the typed collection search response view.
type CollectionSearchView struct {
	CollectionID string
	Query        string
	Results      []CollectionResultView
}

// CollectionResultView is one typed collection search hit.
type CollectionResultView struct {
	RowKey  string
	Score   float64
	Content string
}

// CollectionItemStateView is the collection item state response view. An
// empty Fingerprint means the item is not indexed.
type CollectionItemStateView struct {
	CollectionID string
	ItemID       string
	Fingerprint  string
}
