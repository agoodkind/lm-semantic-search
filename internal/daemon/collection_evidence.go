package daemon

import (
	"context"
	"log/slog"
)

type collectionEvidence struct {
	presence   collectionPresence
	rows       int32
	rowsKnown  bool
	collection string
	nameSource string
}

// probeCollectionEvidence owns the routing invariant: existence and row
// evidence is judged against the STORED codebase.CollectionName whenever the
// registry has one, and a derived name is only a fallback for untracked
// paths. Store errors stay Unknown, and Unknown never routes a job to
// bootstrap, so a transient Milvus failure can never trigger a full rebuild.
func (manager *Manager) probeCollectionEvidence(ctx context.Context, canonicalPath string, caller string) collectionEvidence {
	evidence := collectionEvidence{
		presence:   collectionPresenceUnknown,
		rows:       0,
		rowsKnown:  false,
		collection: "",
		nameSource: "",
	}
	if manager.semantic == nil || !manager.semantic.Available() {
		return evidence
	}

	collectionName := manager.storedCollectionNameForPath(canonicalPath)
	nameSource := "stored"
	if collectionName == "" {
		collectionName = manager.semantic.CollectionName(canonicalPath)
		nameSource = "derived"
	}
	evidence.collection = collectionName
	evidence.nameSource = nameSource
	if collectionName == "" {
		return evidence
	}

	facts, inspectErr := manager.semantic.InspectCollection(ctx, collectionName)
	if inspectErr != nil {
		slog.WarnContext(
			ctx,
			"Milvus InspectCollection failed",
			"caller",
			caller,
			"path",
			canonicalPath,
			"collection",
			evidence.collection,
			"name_source",
			evidence.nameSource,
			"err",
			inspectErr,
		)
		return evidence
	}
	evidence.rows = facts.Rows
	evidence.rowsKnown = facts.RowsKnown
	if facts.Exists {
		evidence.presence = collectionPresencePresent
		return evidence
	}
	evidence.presence = collectionPresenceMissing
	return evidence
}

// storedCollectionNameForPath prefers the registry's persisted collection
// name because it is the name rows were actually written under; deriving
// from the path can diverge from it, and a divergence must never be judged
// as a missing collection.
func (manager *Manager) storedCollectionNameForPath(canonicalPath string) string {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	for _, codebase := range manager.codebases {
		if codebase.CanonicalPath == canonicalPath {
			return codebase.CollectionName
		}
	}
	return ""
}
