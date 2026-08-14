//go:build restartacceptance

package restartacceptance

import (
	"context"
	"fmt"
	"time"
)

const cloneColdSearchTimeout = 15 * time.Second

type cloneConfirmationInput struct {
	Census                 collectionCensusFunc
	ScalarDebt             func(context.Context) ([]string, error)
	ColdCodeSearch         func(context.Context) (semanticSearchObservation, error)
	ColdConversationSearch func(context.Context) (semanticSearchObservation, error)
	Health                 func(context.Context) error
	ActiveJobs             func(context.Context) (int, error)
}

type cloneConfirmationResult struct {
	Census cloneMilvusCensus
}

func runCloneConfirmation(
	ctx context.Context,
	input cloneConfirmationInput,
) (cloneConfirmationResult, error) {
	if input.Census == nil || input.ScalarDebt == nil || input.ColdCodeSearch == nil ||
		input.ColdConversationSearch == nil || input.Health == nil || input.ActiveJobs == nil {
		return cloneConfirmationResult{}, fmt.Errorf("clone confirmation operations are incomplete")
	}
	census, err := input.Census(ctx)
	if err != nil {
		return cloneConfirmationResult{}, fmt.Errorf("capture restored clone census: %w", err)
	}
	if len(census.Collections) == 0 {
		return cloneConfirmationResult{}, fmt.Errorf("restored clone census is empty")
	}
	if len(census.Samples) == 0 {
		return cloneConfirmationResult{}, fmt.Errorf("restored clone vector samples are empty")
	}
	debt, err := input.ScalarDebt(ctx)
	if err != nil {
		return cloneConfirmationResult{}, fmt.Errorf("check restored clone scalar debt: %w", err)
	}
	if len(debt) != 0 {
		return cloneConfirmationResult{}, fmt.Errorf("restored clone conversation scalar debt exists: %v", debt)
	}
	codeContext, cancelCode := context.WithTimeout(ctx, cloneColdSearchTimeout)
	codeResult, err := input.ColdCodeSearch(codeContext)
	cancelCode()
	if err != nil {
		return cloneConfirmationResult{}, fmt.Errorf("search cold clone code collection: %w", err)
	}
	if len(codeResult.ResultIDs) == 0 {
		return cloneConfirmationResult{}, fmt.Errorf("cold clone code search returned no results")
	}
	conversationContext, cancelConversation := context.WithTimeout(ctx, cloneColdSearchTimeout)
	conversationResult, err := input.ColdConversationSearch(conversationContext)
	cancelConversation()
	if err != nil {
		return cloneConfirmationResult{}, fmt.Errorf("search cold clone conversation collection: %w", err)
	}
	if len(conversationResult.ResultIDs) == 0 {
		return cloneConfirmationResult{}, fmt.Errorf("cold clone conversation search returned no results")
	}
	if err := input.Health(ctx); err != nil {
		return cloneConfirmationResult{}, fmt.Errorf("check isolated services after cold searches: %w", err)
	}
	activeJobs, err := input.ActiveJobs(ctx)
	if err != nil {
		return cloneConfirmationResult{}, fmt.Errorf("count isolated jobs after cold searches: %w", err)
	}
	if activeJobs != 0 {
		return cloneConfirmationResult{}, fmt.Errorf("isolated active jobs = %d, want 0", activeJobs)
	}
	return cloneConfirmationResult{Census: census}, nil
}
