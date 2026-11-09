package daemon

import (
	"goodkind.io/lm-semantic-search/internal/model"
)

func mergeSchedulingPolicyPatches(
	existing model.SchedulingPolicyPatch,
	incoming model.SchedulingPolicyPatch,
) model.SchedulingPolicyPatch {
	merged := existing
	if incoming.Priority != nil {
		merged.Priority = incoming.Priority
	}

	if incoming.Quiet != nil {
		merged.Quiet = incoming.Quiet
	}

	if incoming.IdleAfterSeconds != nil {
		merged.IdleAfterSeconds = incoming.IdleAfterSeconds
	}

	return merged
}

func applyJobSchedulingPolicy(
	job *model.Job,
	effectivePolicy model.SchedulingPolicy,
	override model.SchedulingPolicyPatch,
	queueSequence uint64,
) {
	job.EffectiveSchedulingPolicy = effectivePolicy
	job.SchedulingOverride = override
	job.QueueSequence = queueSequence
}

func policyUpdateAppliesToJob(job model.Job) bool {
	switch job.State {
	case model.JobStateQueued, model.JobStateRunning, model.JobStatePaused:
		return true
	case model.JobStateCancelling,
		model.JobStateCompleted,
		model.JobStateFailed,
		model.JobStateCancelled:
		return false
	default:
		return false
	}
}

func clearPatchedSchedulingOverrides(
	override model.SchedulingPolicyPatch,
	patch model.SchedulingPolicyPatch,
) model.SchedulingPolicyPatch {
	cleared := override
	if patch.Priority != nil {
		cleared.Priority = nil
	}

	if patch.Quiet != nil {
		cleared.Quiet = nil
	}

	if patch.IdleAfterSeconds != nil {
		cleared.IdleAfterSeconds = nil
	}

	return cleared
}

func policyUpdateTransactionJobs(
	transaction model.PolicyUpdateTransaction,
) []model.Job {
	jobCount := len(transaction.OldDetachedJobs)
	if transaction.OldActiveJob != nil {
		jobCount++
	}

	jobs := make([]model.Job, 0, jobCount)
	if transaction.OldActiveJob != nil {
		jobs = append(jobs, *transaction.OldActiveJob)
	}

	jobs = append(jobs, transaction.OldDetachedJobs...)

	return jobs
}

func fullSchedulingPolicyPatch(
	policy model.SchedulingPolicy,
) model.SchedulingPolicyPatch {
	priority := policy.Priority
	quiet := policy.Quiet
	idleAfterSeconds := policy.IdleAfterSeconds

	return model.SchedulingPolicyPatch{
		Priority:         &priority,
		Quiet:            &quiet,
		IdleAfterSeconds: &idleAfterSeconds,
	}
}
