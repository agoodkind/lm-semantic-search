//go:build restartacceptance

package restartacceptance

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"
)

func TestRunCloneConfirmationProvesEveryRequiredBoundary(t *testing.T) {
	events := make([]string, 0, 7)
	input := cloneConfirmationInput{
		Census: func(context.Context) (cloneMilvusCensus, error) {
			events = append(events, "census")
			return cloneMilvusCensus{
				Databases: []string{"default"},
				Collections: collectionCensus{
					{Database: "default", Collection: "restored"}: "hash",
				},
				Samples: collectionCensus{
					{Database: "default", Collection: "restored"}: "vector-hash",
				},
			}, nil
		},
		ScalarDebt: func(context.Context) ([]string, error) {
			events = append(events, "scalar-debt")
			return nil, nil
		},
		ColdCodeSearch: func(ctx context.Context) (semanticSearchObservation, error) {
			requireBoundedCloneSearch(t, ctx)
			events = append(events, "cold-code")
			return semanticSearchObservation{ResultIDs: []string{"code-row"}}, nil
		},
		ColdConversationSearch: func(ctx context.Context) (semanticSearchObservation, error) {
			requireBoundedCloneSearch(t, ctx)
			events = append(events, "cold-conversation")
			return semanticSearchObservation{ResultIDs: []string{"conversation-row"}}, nil
		},
		Health: func(context.Context) error {
			events = append(events, "health")
			return nil
		},
		ActiveJobs: func(context.Context) (int, error) {
			events = append(events, "jobs")
			return 0, nil
		},
	}

	result, err := runCloneConfirmation(context.Background(), input)
	if err != nil {
		t.Fatalf("run clone confirmation: %v", err)
	}
	wantEvents := []string{
		"census",
		"scalar-debt",
		"cold-code",
		"cold-conversation",
		"health",
		"jobs",
	}
	if !slices.Equal(events, wantEvents) {
		t.Fatalf("confirmation events = %v, want %v", events, wantEvents)
	}
	if len(result.Census.Collections) != 1 || len(result.Census.Samples) != 1 {
		t.Fatalf("confirmation census = %+v", result.Census)
	}
}

func TestRunCloneConfirmationRejectsDebtMissingSamplesEmptySearchAndActiveJobs(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*cloneConfirmationInput)
	}{
		{name: "scalar debt", mutate: func(input *cloneConfirmationInput) {
			input.ScalarDebt = func(context.Context) ([]string, error) { return []string{"conv_chunks_debt"}, nil }
		}},
		{name: "missing vector samples", mutate: func(input *cloneConfirmationInput) {
			input.Census = func(context.Context) (cloneMilvusCensus, error) {
				return cloneMilvusCensus{Collections: collectionCensus{{Collection: "restored"}: "hash"}}, nil
			}
		}},
		{name: "empty code search", mutate: func(input *cloneConfirmationInput) {
			input.ColdCodeSearch = func(context.Context) (semanticSearchObservation, error) {
				return semanticSearchObservation{}, nil
			}
		}},
		{name: "empty conversation search", mutate: func(input *cloneConfirmationInput) {
			input.ColdConversationSearch = func(context.Context) (semanticSearchObservation, error) {
				return semanticSearchObservation{}, nil
			}
		}},
		{name: "active jobs", mutate: func(input *cloneConfirmationInput) {
			input.ActiveJobs = func(context.Context) (int, error) { return 1, nil }
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			input := successfulCloneConfirmationInput()
			test.mutate(&input)
			if _, err := runCloneConfirmation(context.Background(), input); err == nil {
				t.Fatal("invalid clone confirmation passed")
			}
		})
	}
}

func successfulCloneConfirmationInput() cloneConfirmationInput {
	return cloneConfirmationInput{
		Census: func(context.Context) (cloneMilvusCensus, error) {
			return cloneMilvusCensus{
				Collections: collectionCensus{{Collection: "restored"}: "hash"},
				Samples:     collectionCensus{{Collection: "restored"}: "vector-hash"},
			}, nil
		},
		ScalarDebt: func(context.Context) ([]string, error) { return nil, nil },
		ColdCodeSearch: func(context.Context) (semanticSearchObservation, error) {
			return semanticSearchObservation{ResultIDs: []string{"code-row"}}, nil
		},
		ColdConversationSearch: func(context.Context) (semanticSearchObservation, error) {
			return semanticSearchObservation{ResultIDs: []string{"conversation-row"}}, nil
		},
		Health:     func(context.Context) error { return nil },
		ActiveJobs: func(context.Context) (int, error) { return 0, nil },
	}
}

func requireBoundedCloneSearch(t *testing.T, ctx context.Context) {
	t.Helper()
	deadline, exists := ctx.Deadline()
	if !exists {
		t.Fatal("cold search has no deadline")
	}
	remaining := time.Until(deadline)
	if remaining <= 14*time.Second || remaining > 15*time.Second {
		t.Fatalf("cold search deadline = %s, want 15 seconds", remaining)
	}
	if !errors.Is(context.Cause(ctx), nil) {
		t.Fatalf("cold search context starts canceled: %v", context.Cause(ctx))
	}
}

func TestWaitForCloneConversationSeedWaitsForFeederConvergence(t *testing.T) {
	statusCalls := 0
	searchCalls := 0
	err := waitForCloneConversationSeed(
		context.Background(),
		func(context.Context) (clydeStatusObservation, error) {
			statusCalls++
			if statusCalls == 1 {
				return clydeStatusObservation{
					PID: 7, Responding: true, Manifest: 1, Needed: 1, Pending: 1,
				}, nil
			}
			return clydeStatusObservation{
				PID: 7, Responding: true, Manifest: 1, Embedded: 1,
			}, nil
		},
		func(context.Context) (semanticSearchObservation, error) {
			searchCalls++
			if statusCalls < 2 {
				t.Fatal("conversation search ran before feeder convergence")
			}
			return semanticSearchObservation{
				Succeeded: true,
				Source:    "semantic",
				Matches:   1,
				ResultIDs: []string{"conversation:0"},
			}, nil
		},
	)
	if err != nil {
		t.Fatalf("wait for clone conversation seed: %v", err)
	}
	if searchCalls != 1 {
		t.Fatalf("search calls = %d, want 1", searchCalls)
	}
}

func TestPrepareCloneConfirmationColdTargetsStopsDaemonBeforeBothReleases(t *testing.T) {
	events := make([]string, 0, 3)
	err := prepareCloneConfirmationColdTargets(
		context.Background(),
		&daemonRuntime{},
		[]string{"code", "conversation"},
		func(*daemonRuntime) error {
			events = append(events, "stop")
			return nil
		},
		func(_ context.Context, collectionName string) error {
			if len(events) == 0 || events[0] != "stop" {
				t.Fatal("collection released before the daemon stopped")
			}
			events = append(events, "release:"+collectionName)
			return nil
		},
	)
	if err != nil {
		t.Fatalf("prepare clone confirmation cold targets: %v", err)
	}
	want := []string{"stop", "release:code", "release:conversation"}
	if !slices.Equal(events, want) {
		t.Fatalf("events = %v, want %v", events, want)
	}
}
