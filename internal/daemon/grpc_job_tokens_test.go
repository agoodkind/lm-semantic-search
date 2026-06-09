package daemon

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

// A protobuf job carries the resolved display tokens so a machine consumer reads
// the same folded status the human surfaces do, while the raw state and error
// stay on their own fields for debugging. A retryable failure during a degraded
// pipeline folds the same way the single-job and list views do: the state reads
// "failed (retryable)" and the display error is suppressed because the banner
// already carries the cause.
func TestToJobWithTokensEmitsResolvedDisplay(t *testing.T) {
	t.Parallel()
	job := model.Job{
		ID:    "job_x",
		State: model.JobStateFailed,
		Error: &model.JobError{Message: "embedding endpoint is unreachable", Retryable: true},
	}

	degraded := toJobWithTokens(job, true)
	if got := degraded.GetDisplayState(); got != "failed (retryable)" {
		t.Fatalf("degraded display_state = %q, want %q", got, "failed (retryable)")
	}
	if got := degraded.GetDisplayError(); got != "" {
		t.Fatalf("degraded display_error = %q, want empty (banner carries the cause)", got)
	}
	if got := degraded.GetState(); got != string(model.JobStateFailed) {
		t.Fatalf("raw state = %q, want %q kept for machine parsing", got, model.JobStateFailed)
	}

	healthy := toJobWithTokens(job, false)
	if got := healthy.GetDisplayError(); got != "embedding endpoint is unreachable" {
		t.Fatalf("healthy display_error = %q, want the message shown", got)
	}
}
