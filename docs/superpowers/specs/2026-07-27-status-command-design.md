# Status command display design

Status: proposed
Date: 2026-07-27

Scope: what `lm-semantic-search status` prints. How the values reach the command is a separate design.

## Rules

Print the counter's own name and its value. No translation, no status words, no glyphs, no bars, no color as meaning. `embed_vectors_total 3946`, not `rate 1.5 vectors per second`.

The names are the identifiers `internal/metrics` already uses as both expvar keys and slog field keys, and the protobuf field names for everything else. A name on screen is greppable in the daemon log and findable in the source.

The command renders a list of name, value, and unit triples and knows none of the names. It cannot fork the status vocabulary that `cmd/lm-semantic-search/display_guard_test.go` guards, because it never calls a named getter. The daemon supplies the triples, their group, and their order.

Order is the declaration order in the source, not alphabetical, so a line stays where it was between releases.

Absent is `null`. Empty string is `""`. Zero is `0`. These are three different facts and read differently.

Derived values are not printed. `embed_latency_ms_sum` and `embed_batches_total` are both on screen; their quotient is the reader's to take.

## Numbers

The terminal groups digits in threes with commas and right-aligns on the last digit, so `2,189,323` and `824` line up and a digit-count change is visible at a glance.

Piped output prints raw digits with no separators, because that output is parsed. Grouping is a terminal-only affordance and the two must not diverge in any other way.

Units live in their own column, taken from the daemon rather than inferred from the name. A unit makes the delta unambiguous: `+16,104 ms` and `+72 vectors` are different kinds of movement.

A value with no unit, such as a timestamp, a boolean, or a string, leaves the column blank. A constant, such as `index_slots_total`, leaves the delta blank.

Byte counts print as bytes. A conversion to a larger unit is one more thing to distrust when a number looks wrong.

## Terminal

Refreshes on an interval, default two seconds. The fourth column is the change since the previous refresh. Two observations, not a derived rate.

```
lm-semantic-search  version=202607270542-fe-6e0a44c  pid=7342  uptime_s=11,539 s
socket=/Users/agoodkind/.local/state/lm-semantic-search/sockets/lm-semantic-search-daemon.sock
read_at=2026-07-27T12:52:31Z  interval=2s

dependency_health.degraded                  false
dependency_health.mode                         ""
dependency_health.since                      null
dependency_health.last_healthy_at            2026-07-27T12:52:20Z

index_slots_in_use                              2  slots               +0
index_slots_total                               4  slots
jobs_active                                     2  jobs                +0
jobs_completed_total                       28,989  jobs                +0
jobs_failed_total                              61  jobs                +0
jobs_cancelled_total                          362  jobs                +0
boot_resumes_total                              0  jobs                +0
sync_skipped_inflight_total                 1,524  requests            +2

embed_inflight                                  1  batches             +0
embed_batches_total                           824  batches            +11
embed_batches_failed                            0  batches             +0
embed_vectors_total                         3,946  vectors            +72
embed_latency_ms_sum                    2,189,323  ms             +16,104
embed_chunks_reused_total                   4,025  chunks             +38

converge_upsert_total                         843  paths               +3
converge_remove_total                           5  paths               +0
converge_copy_chunks_total                      0  paths               +0
sweep_runs_total                              200  runs                +0
sweep_changed_total                             6  runs                +0

num_goroutine                                  34  goroutines          -1
heap_alloc_bytes                      249,278,160  bytes       +8,231,344
heap_inuse_bytes                      289,931,264  bytes               +0
num_gc                                        289  cycles              +2

codebases_total                               128  codebases
codebases.status=indexed                      105  codebases
codebases.status=indexing                       2  codebases
codebases.status=stale                          0  codebases
codebases.status=failed                         0  codebases
codebases.status=missing                       21  codebases
codebases.status=discovered                     0  codebases
codebases.status=quarantined                    0  codebases

activity  running=2 queued=3

[0] job_id=job_1785156767_bf1c45076324
    codebase_id=cb_1785149463_0213d6697d89  operation=index  state=running
    canonical_path=/Users/agoodkind/Sites/clyde-dev/clyde/.claude/worktrees/cursor-ingest-fix
    trigger=changed_files  forced=false
    phase=embedding  overall_percent=40.8  phase_percent=61.2
    files_processed=312  files_total=764  unit=file
    chunks_generated=4,123  chunks_embedded=98  chunks_reused=4,025  chunks_dropped=0
    reuse_vectors_loaded=4,025  collection_rows_written=1,032
    embedding_batches_completed=812  embedding_batches_total=824
    started_at=14:48:02  last_event_at=14:52:27  heartbeat_at=14:52:30

[1] job_id=null  source=watcher
    codebase_id=cb_1780707585_a95525d0b0db  operation=converge  state=running
    canonical_path=/Users/agoodkind/Sites/lm-semantic-search
    pending_paths=8
    started_at=14:52:22

[2] job_id=null  source=watcher
    codebase_id=cb_1785151182_893b769460bc  operation=converge  state=queued
    canonical_path=/Users/agoodkind/Sites/configs
    pending_paths=12

up/down select   enter expand   p pause   r refresh   q quit
```

Activity fields carry their unit in the name (`files_processed`, `chunks_embedded`, `pending_paths`, `overall_percent`, `collection_rows_written`), so they take no unit column. Their numbers are grouped like the counter table.

`job_id=null` states that file-change work carries no job record, so `job get` and `job cancel` cannot address it.

The only emphasis is that a nonzero delta renders bright and a zero delta renders dim. Strip it and the numbers are unchanged.

The counter block holds its position and never scrolls. The activity block scrolls and reports what it hid as `3 more`.

Pausing replaces `read_at` with `paused_at`. A failed refresh appends `refresh_error="<message>"` to the `read_at` line and leaves the timestamp at the last successful read.

## Piped

One snapshot, no delta column, no key line. One record per line as `name value unit`, whitespace separated, raw digits, so `grep`, `awk`, and `cut` work on it. A record with no unit ends after the value.

Every value is exactly one whitespace-free field, whatever it holds, so `awk '{print $2}'` and `cut -d' ' -f2` read a value by position and never split one across fields.

A value carrying whitespace, a quote, or a backslash is escaped with Go quoting, and its spaces become `\x20`, so nothing inside it is whitespace. `strconv.Unquote` and any Go-string decoder recover the original bytes exactly. Escaping also means a value holding a newline cannot end its record early and forge a second one, which matters because a codebase path is operator-supplied input that reaches this renderer.

```
version 202607270542-fe-6e0a44c
pid 7342
uptime_s 11539 s
socket /Users/agoodkind/.local/state/lm-semantic-search/sockets/lm-semantic-search-daemon.sock
read_at 2026-07-27T12:52:31Z
dependency_health.degraded false
dependency_health.mode ""
dependency_health.since null
dependency_health.last_healthy_at 2026-07-27T14:52:20Z
index_slots_in_use 2 slots
index_slots_total 4 slots
jobs_active 2 jobs
jobs_completed_total 28989 jobs
jobs_failed_total 61 jobs
jobs_cancelled_total 362 jobs
boot_resumes_total 0 jobs
sync_skipped_inflight_total 1524 requests
embed_inflight 1 batches
embed_batches_total 824 batches
embed_batches_failed 0 batches
embed_vectors_total 3946 vectors
embed_latency_ms_sum 2189323 ms
embed_chunks_reused_total 4025 chunks
converge_upsert_total 843 paths
converge_remove_total 5 paths
converge_copy_chunks_total 0 paths
sweep_runs_total 200 runs
sweep_changed_total 6 runs
num_goroutine 34 goroutines
heap_alloc_bytes 249278160 bytes
heap_inuse_bytes 289931264 bytes
num_gc 289 cycles
codebases_total 128 codebases
codebases.status=indexed 105 codebases
codebases.status=indexing 2 codebases
codebases.status=stale 0 codebases
codebases.status=failed 0 codebases
codebases.status=missing 21 codebases
codebases.status=discovered 0 codebases
codebases.status=quarantined 0 codebases
activity.running 2
activity.queued 3
activity.0.job_id job_1785156767_bf1c45076324
activity.0.codebase_id cb_1785149463_0213d6697d89
activity.0.operation index
activity.0.state running
activity.0.canonical_path /Users/agoodkind/Sites/clyde-dev/clyde/.claude/worktrees/cursor-ingest-fix
activity.0.phase embedding
activity.0.overall_percent 40.8 %
activity.0.files_processed 312 files
activity.0.files_total 764 files
activity.0.chunks_generated 4123 chunks
activity.0.chunks_embedded 98 chunks
activity.0.chunks_reused 4025 chunks
activity.0.chunks_dropped 0 chunks
activity.0.collection_rows_written 1032 rows
activity.1.job_id null
activity.1.source watcher
activity.1.codebase_id cb_1780707585_a95525d0b0db
activity.1.operation converge
activity.1.state running
activity.1.pending_paths 8 paths
```

Timestamps print as UTC, the same value the JSON carries. This form is parsed, so it takes the one unambiguous zone; the terminal is the surface that converts to the reader's own zone.

`--output single-line` prints the first line, `version <value>`.

## JSON

`--json` prints one compact line through `protojson`, the same path every other command uses. The sample below is indented to be read; the real output has no newlines.

Two verified `protojson` behaviors decide the shape.

A plain proto3 scalar at its zero value is omitted. `lm-semantic-search --json job list` today returns `"dependencyHealth": {"lastHealthyAt": "..."}` with no `degraded` and no `mode`, because `false` and `""` are dropped. A consumer cannot tell a healthy daemon from a missing field.

A field inside a `oneof` has explicit presence and is emitted at any value. Modelling the value as a `oneof` is therefore what keeps `0`, `false`, `""`, and absent as four distinguishable facts, which the terminal and piped forms already distinguish.

```proto
message Metric {
  string group = 1;
  string name = 2;
  string unit = 3;
  oneof value {
    int64 int_value = 4;
    double double_value = 5;
    bool bool_value = 6;
    string string_value = 7;
  }
}
```

The counter name is a string value, not a field name, so `protojson` casing never touches it. `embed_vectors_total` is byte-identical on the screen, in the piped text, in the JSON, and in the `daemon.perf_counters` log line.

The key that is present names the type. No value key at all means the value is absent, which the other two forms print as `null`.

```json
{
  "readAt": "2026-07-27T12:52:31.004Z",
  "daemon": {
    "version": "202607270542-fe-6e0a44c",
    "commit": "6e0a44c",
    "pid": 7342,
    "socketPath": "/Users/agoodkind/.local/state/lm-semantic-search/sockets/lm-semantic-search-daemon.sock",
    "startedAt": "2026-07-27T09:40:12.117Z"
  },
  "metrics": [
    {"group": "daemon", "name": "uptime_s", "intValue": "11539", "unit": "s"},

    {"group": "dependency_health", "name": "dependency_health.degraded", "boolValue": false},
    {"group": "dependency_health", "name": "dependency_health.mode", "stringValue": ""},
    {"group": "dependency_health", "name": "dependency_health.since"},
    {"group": "dependency_health", "name": "dependency_health.last_healthy_at",
     "stringValue": "2026-07-27T12:52:20.881Z"},

    {"group": "jobs", "name": "index_slots_in_use", "intValue": "2", "unit": "slots"},
    {"group": "jobs", "name": "index_slots_total", "intValue": "4", "unit": "slots"},
    {"group": "jobs", "name": "jobs_active", "intValue": "2", "unit": "jobs"},
    {"group": "jobs", "name": "jobs_completed_total", "intValue": "28989", "unit": "jobs"},
    {"group": "jobs", "name": "jobs_failed_total", "intValue": "61", "unit": "jobs"},
    {"group": "jobs", "name": "jobs_cancelled_total", "intValue": "362", "unit": "jobs"},
    {"group": "jobs", "name": "boot_resumes_total", "intValue": "0", "unit": "jobs"},
    {"group": "jobs", "name": "sync_skipped_inflight_total", "intValue": "1524", "unit": "requests"},

    {"group": "embed", "name": "embed_inflight", "intValue": "1", "unit": "batches"},
    {"group": "embed", "name": "embed_batches_total", "intValue": "824", "unit": "batches"},
    {"group": "embed", "name": "embed_batches_failed", "intValue": "0", "unit": "batches"},
    {"group": "embed", "name": "embed_vectors_total", "intValue": "3946", "unit": "vectors"},
    {"group": "embed", "name": "embed_latency_ms_sum", "intValue": "2189323", "unit": "ms"},
    {"group": "embed", "name": "embed_chunks_reused_total", "intValue": "4025", "unit": "chunks"},

    {"group": "converge", "name": "converge_upsert_total", "intValue": "843", "unit": "paths"},
    {"group": "converge", "name": "converge_remove_total", "intValue": "5", "unit": "paths"},
    {"group": "converge", "name": "converge_copy_chunks_total", "intValue": "0", "unit": "paths"},
    {"group": "converge", "name": "sweep_runs_total", "intValue": "200", "unit": "runs"},
    {"group": "converge", "name": "sweep_changed_total", "intValue": "6", "unit": "runs"},

    {"group": "runtime", "name": "num_goroutine", "intValue": "34", "unit": "goroutines"},
    {"group": "runtime", "name": "heap_alloc_bytes", "intValue": "249278160", "unit": "bytes"},
    {"group": "runtime", "name": "heap_inuse_bytes", "intValue": "289931264", "unit": "bytes"},
    {"group": "runtime", "name": "num_gc", "intValue": "289", "unit": "cycles"},

    {"group": "codebases", "name": "codebases_total", "intValue": "128", "unit": "codebases"},
    {"group": "codebases", "name": "codebases.status=indexed", "intValue": "105", "unit": "codebases"},
    {"group": "codebases", "name": "codebases.status=indexing", "intValue": "2", "unit": "codebases"},
    {"group": "codebases", "name": "codebases.status=stale", "intValue": "0", "unit": "codebases"},
    {"group": "codebases", "name": "codebases.status=failed", "intValue": "0", "unit": "codebases"},
    {"group": "codebases", "name": "codebases.status=missing", "intValue": "21", "unit": "codebases"},
    {"group": "codebases", "name": "codebases.status=discovered", "intValue": "0", "unit": "codebases"},
    {"group": "codebases", "name": "codebases.status=quarantined", "intValue": "0", "unit": "codebases"},

    {"group": "activity", "name": "activity.running", "intValue": "2"},
    {"group": "activity", "name": "activity.queued", "intValue": "3"}
  ],
  "activity": [
    {
      "metrics": [
        {"name": "job_id", "stringValue": "job_1785156767_bf1c45076324"},
        {"name": "codebase_id", "stringValue": "cb_1785149463_0213d6697d89"},
        {"name": "operation", "stringValue": "index"},
        {"name": "state", "stringValue": "running"},
        {"name": "source", "stringValue": "job"},
        {"name": "canonical_path", "stringValue":
         "/Users/agoodkind/Sites/clyde-dev/clyde/.claude/worktrees/cursor-ingest-fix"},
        {"name": "trigger", "stringValue": "changed_files"},
        {"name": "forced", "boolValue": false},
        {"name": "phase", "stringValue": "embedding"},
        {"name": "overall_percent", "doubleValue": 40.8, "unit": "%"},
        {"name": "phase_percent", "doubleValue": 61.2, "unit": "%"},
        {"name": "files_processed", "intValue": "312", "unit": "files"},
        {"name": "files_total", "intValue": "764", "unit": "files"},
        {"name": "chunks_generated", "intValue": "4123", "unit": "chunks"},
        {"name": "chunks_embedded", "intValue": "98", "unit": "chunks"},
        {"name": "chunks_reused", "intValue": "4025", "unit": "chunks"},
        {"name": "chunks_dropped", "intValue": "0", "unit": "chunks"},
        {"name": "reuse_vectors_loaded", "intValue": "4025", "unit": "vectors"},
        {"name": "collection_rows_written", "intValue": "1032", "unit": "rows"},
        {"name": "embedding_batches_completed", "intValue": "812", "unit": "batches"},
        {"name": "embedding_batches_total", "intValue": "824", "unit": "batches"},
        {"name": "started_at", "stringValue": "2026-07-27T12:48:02.441Z"},
        {"name": "last_event_at", "stringValue": "2026-07-27T12:52:27.019Z"},
        {"name": "heartbeat_at", "stringValue": "2026-07-27T12:52:30.772Z"}
      ]
    },
    {
      "metrics": [
        {"name": "job_id"},
        {"name": "source", "stringValue": "watcher"},
        {"name": "codebase_id", "stringValue": "cb_1780707585_a95525d0b0db"},
        {"name": "operation", "stringValue": "converge"},
        {"name": "state", "stringValue": "running"},
        {"name": "canonical_path", "stringValue": "/Users/agoodkind/Sites/lm-semantic-search"},
        {"name": "pending_paths", "intValue": "8", "unit": "paths"},
        {"name": "started_at", "stringValue": "2026-07-27T12:52:22.310Z"}
      ]
    },
    {
      "metrics": [
        {"name": "job_id"},
        {"name": "source", "stringValue": "watcher"},
        {"name": "codebase_id", "stringValue": "cb_1785151182_893b769460bc"},
        {"name": "operation", "stringValue": "converge"},
        {"name": "state", "stringValue": "queued"},
        {"name": "canonical_path", "stringValue": "/Users/agoodkind/Sites/configs"},
        {"name": "pending_paths", "intValue": "12", "unit": "paths"}
      ]
    }
  ]
}
```

The activity row index is the array position, so no index field is carried and none can be dropped at zero.

`{"name": "job_id"}` with no value key is how the file-watcher rows say they carry no job record. The terminal and piped forms print that as `job_id=null`.

An `intValue` is a JSON string because `protojson` encodes 64-bit integers as strings to keep precision in consumers whose numbers are doubles. `doubleValue` is a JSON number. This matches every other command in this CLI: `totalBytes` on `codebase list` is `"904166"` and `totalChunks` is `26669`.

Every timestamp is UTC on every surface: the terminal, the piped text, and the JSON. This repository bans local-time conversion, which the `gosmopolitan` linter enforces, so one zone is the only available answer and no two surfaces can disagree about an instant.

`displayText` appears in the JSON reply, as it does on every other command in this service, because one shared marshaller serves them all. A JSON consumer reads `metrics` and `activity` and ignores it.

There is no delta in JSON. A delta is the difference between two reads, and a single call is one read. A consumer that wants a rate calls twice and subtracts, which is what the terminal does.

### Extracting one value

```
lm-semantic-search --json status \
  | jq -r '.metrics[] | select(.name=="embed_vectors_total") | .intValue'
```

Every activity field uses the same filter one level down:

```
lm-semantic-search --json status \
  | jq -r '.activity[] | .metrics[] | select(.name=="pending_paths") | .intValue'
```

### Size

The reply is bounded by the counter count plus the active and queued rows, so it stays a few kilobytes. It carries no job history. `lm-semantic-search --json job list` returns 43,745 job records on one line today; `status` must not grow that way, and reporting only non-terminal work is what keeps it from doing so.

## Terminal detection

The live screen runs when the output mode is human and stdout is a terminal, using the rule `newCodebaseListCmd` already applies. Every other combination prints once. `--once` forces one print on a terminal. `--interval` sets the refresh cadence.

## Values the display reads

A flat, ordered, grouped list of name, value, and unit triples, with the daemon supplying the names, the units, and the grouping. The names above are the identifiers in `internal/metrics`, the `Progress` and `Job` protobuf fields, and the runtime gauges the periodic `daemon.perf_counters` line already carries.

Units the daemon assigns, matched to what each counter's increment call site actually counts: `slots`, `jobs`, `requests`, `batches`, `vectors`, `chunks`, `paths`, `runs`, `files`, `rows`, `goroutines`, `cycles`, `bytes`, `ms`, `s`, `%`, `codebases`.

`embed_chunks_reused_total` does not exist in `internal/metrics` today. Reuse counts exist per job on `Progress.chunks_reused` and per write in the `semantic.chunks_written` log line, so the counter is an addition rather than a rename.
