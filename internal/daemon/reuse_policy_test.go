package daemon

import (
	"context"
	"errors"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func TestResolveItemReusePolicy(t *testing.T) {
	cases := []struct {
		name          string
		staging       bool
		semanticReady bool
		forced        bool
		presence      collectionPresence
		want          bool
		wantInspect   bool
	}{
		{name: "live ready enables without probing", staging: false, semanticReady: true, forced: true, presence: collectionPresenceMissing, want: true, wantInspect: false},
		{name: "live unavailable disables without probing", staging: false, semanticReady: false, forced: false, presence: collectionPresencePresent, want: false, wantInspect: false},
		{name: "staging present enables", staging: true, semanticReady: true, forced: false, presence: collectionPresencePresent, want: true, wantInspect: true},
		{name: "staging missing disables", staging: true, semanticReady: true, forced: false, presence: collectionPresenceMissing, want: false, wantInspect: true},
		{name: "staging unknown disables", staging: true, semanticReady: true, forced: false, presence: collectionPresenceUnknown, want: false, wantInspect: true},
		{name: "staging forced disables without probing", staging: true, semanticReady: true, forced: true, presence: collectionPresencePresent, want: false, wantInspect: false},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			manager, _, _ := newTestManager(t)
			canonical := t.TempDir()
			collectionName := "cc_policy"
			inspectCalls := 0
			manager.semantic = &fakeSemantic{
				inspectCollection: func(_ context.Context, gotCollection string) (semantic.CollectionFacts, error) {
					inspectCalls++
					if gotCollection != collectionName {
						t.Fatalf("InspectCollection(%q), want %q", gotCollection, collectionName)
					}
					switch testCase.presence {
					case collectionPresencePresent:
						return semantic.CollectionFacts{Exists: true, Rows: 1, RowsKnown: true}, nil
					case collectionPresenceMissing:
						return semantic.CollectionFacts{Exists: false, Rows: 0, RowsKnown: false}, nil
					case collectionPresenceUnknown:
						return semantic.CollectionFacts{}, errors.New("inspect failed")
					default:
						return semantic.CollectionFacts{}, errors.New("unexpected presence")
					}
				},
			}
			manager.mu.Lock()
			manager.codebases["cb-policy"] = model.Codebase{
				ID:             "cb-policy",
				CanonicalPath:  canonical,
				CollectionName: collectionName,
			}
			manager.mu.Unlock()

			job := model.Job{
				ID:            "job-policy",
				CodebaseID:    "cb-policy",
				CanonicalPath: canonical,
				Forced:        testCase.forced,
			}
			got := manager.resolveItemReusePolicy(context.Background(), job, testCase.staging, testCase.semanticReady)
			if got != testCase.want {
				t.Fatalf("resolveItemReusePolicy() = %v, want %v", got, testCase.want)
			}
			if (inspectCalls > 0) != testCase.wantInspect {
				t.Fatalf("InspectCollection calls = %d, want inspect=%v", inspectCalls, testCase.wantInspect)
			}
		})
	}
}
