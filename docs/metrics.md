# Read historical metrics

`lm-semantic-search status --since 1h` reports the daemon's work during the
last hour. It reads the daemon's status reply, structured logs, and job journal.
It does not read index content or external application data.

Use `lm-semantic-search --json status --since 1h` when another program needs
the report. JSON keeps unavailable values as `null`. Human output renders them
as `n/a`.

## Read values

Counters show the daemon's current value, the window delta, and rate per
second. Gauges show the current, minimum, mean, and maximum sampled value.
Gauges without retained samples keep their historical summaries unavailable.
Durations show total time, calls, mean, p50, p95, and maximum. A percentile is
unavailable when the retained records do not contain individual observations.

Aggregate embedding latency appears separately from the time breakdown.
The time breakdown ranks exclusive stage time. A parent stage excludes direct
child stage durations, so nested spans do not count the same time twice.
Unattributed time is measured completed-job duration minus instrumented stage
time. It identifies work that the retained stages do not explain.

## Check coverage

Coverage is complete only when retained samples bracket the requested window
without a detected restart or malformed record. A daemon restart resets process
counters. Retention can remove the sample before a requested window. A malformed
log or journal record also makes coverage incomplete. Warnings explain each
known gap.

The report uses existing log retention and job-journal compaction. Older work
becomes unavailable when those retained files no longer contain it.

Compressed rotated JSONL files are read with the same record rules as active
JSONL files.
