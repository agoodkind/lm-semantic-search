# Status command implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `lm-semantic-search status`, printing the daemon's counters by their own names in three forms: a live-refreshing terminal screen, one snapshot when output is piped, and one compact JSON line under `--json`.

**Architecture:** One new remote call, `GetStatus`, returns a flat ordered list of name, value, and unit triples plus one list of activity rows. The counter name is a string value rather than a protobuf field name, so the name is byte-identical on every surface and the command never calls a named getter. The command formats and positions the triples and derives nothing.

**Tech Stack:** Go, protobuf through `buf` 1.72.0, gRPC over a unix socket, Bubble Tea and Lip Gloss for the screen, `golang.org/x/term` for terminal detection.

Design: [`docs/superpowers/specs/2026-07-27-status-command-design.md`](../specs/2026-07-27-status-command-design.md)

## Global Constraints

Every `make` invocation needs `GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile`.

Targeted `go test` needs `PKG_CONFIG_PATH=<worktree>/.make/cgo/darwin-arm64/lib/pkgconfig` and `CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path'` exported.

Protobuf codegen is `buf generate` from the repository root. The Makefile does not drive it. `buf.gen.yaml` sets `require_unimplemented_servers=false`, so adding an RPC without a handler is a compile error rather than a silent gap.

The counter name is data, never a protobuf field name. `embed_vectors_total` reads identically on the screen, in piped text, in JSON, and in the `daemon.perf_counters` log line.

The value is a protobuf `oneof`. A plain proto3 scalar at its zero value is omitted by `protojson`, which is why `lm-semantic-search --json job list` returns a `dependencyHealth` object today with no `degraded` and no `mode`. A `oneof` member has explicit presence and is emitted at any value, so `0`, `false`, `""`, and absent stay four distinguishable facts.

The command must not call `GetStatus()` or `GetState()` on a wire record. `cmd/lm-semantic-search/display_guard_test.go` parses every non-test file in that package and fails the build if it does.

Digits group in threes on the terminal only. Piped and JSON output print raw digits, because both are parsed.

Timestamps stay UTC in JSON through `protojson` and render local with an offset on human surfaces through `formatLocalTime`, matching the existing rule in `AGENTS.md`.

No rolling window and no rate live in the daemon. A delta is the difference between two reads, and the terminal has two reads.

Do not weaken, skip, or suppress any `make` gate.

## File structure

| File | Responsibility |
| --- | --- |
| `internal/metrics/metrics.go` | add the reuse counter beside the embed counters |
| `internal/metrics/expvar.go` | publish the reuse counter |
| `internal/metrics/reporter.go` | log the reuse counter |
| `internal/semantic/staging.go` | count a reuse hit where the embedder boundary already counts a batch |
| `internal/daemon/eventqueue.go` | report pending path counts per codebase |
| `internal/daemon/background_sync.go` | report converging and queued file-change work |
| `internal/daemon/manager.go` | hold the daemon start time, the activity reporter, and the slot occupancy |
| `proto/lmsemanticsearch/v1/service.proto` | the `GetStatus` contract |
| `internal/daemon/status_metrics.go` | assemble the metric list (new) |
| `internal/daemon/grpc_server_status.go` | the `GetStatus` handler (new) |
| `internal/render/status_metrics_text.go` | render the piped text from the metric list (new) |
| `cmd/lm-semantic-search/status.go` | the command, piped and JSON paths (new) |
| `cmd/lm-semantic-search/status_tui.go` | the terminal screen (new) |
| `cmd/lm-semantic-search/root.go` | register the command |

---

### Task 1: Count reused chunks

Reuse has no counter today. `embed_vectors_total` counts only vectors that reached the embedder, because `embedChunkBatch` passes only the misses to `embedMissedTexts`. Without a reuse counter, a screen cannot show what a build avoided.

**Files:**
- Modify: `internal/metrics/metrics.go`
- Modify: `internal/metrics/expvar.go`
- Modify: `internal/metrics/reporter.go`
- Modify: `internal/semantic/staging.go:270-300` (`embedChunkBatch`)
- Test: `internal/metrics/metrics_test.go`
- Test: `internal/semantic/reuse_test.go`

**Interfaces:**
- Produces: `metrics.ChunksReused(count int)`, and `metrics.Snapshot.EmbedChunksReusedTotal int64`.

- [ ] **Step 1: Write the failing counter test**

Append to `internal/metrics/metrics_test.go`:

```go
// TestChunksReusedAccumulates proves the reuse counter adds only positive
// counts, so a caller that computed a zero or negative reuse count cannot
// move it. It compares deltas because the counters are package-level atomics
// shared with every other test in this package.
func TestChunksReusedAccumulates(t *testing.T) {
	before := Read().EmbedChunksReusedTotal
	ChunksReused(3)
	ChunksReused(0)
	ChunksReused(-1)
	after := Read().EmbedChunksReusedTotal
	if after-before != 3 {
		t.Fatalf("EmbedChunksReusedTotal delta = %d, want 3", after-before)
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
cd <worktree>
PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig" \
CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path' \
go test ./internal/metrics/ -run TestChunksReusedAccumulates
```

Expected: compile failure, `undefined: ChunksReused`.

- [ ] **Step 3: Add the counter**

In `internal/metrics/metrics.go`, add to the embed group of the `var` block, immediately after `embedInflight`:

```go
	embedChunksReusedTotal atomic.Int64
```

Add to `Snapshot`, immediately after `EmbedInflight`:

```go
	EmbedChunksReusedTotal int64
```

Add the increment beside `EmbedBatchDone`:

```go
// ChunksReused counts chunks served from an already-stored vector. Those
// chunks never reach the embedder, so EmbedBatchDone never sees them; the two
// counters together report what a batch cost and what it avoided. A count of
// zero or less is ignored, so a caller that computed no reuse cannot move it.
func ChunksReused(count int) {
	if count <= 0 {
		return
	}
	embedChunksReusedTotal.Add(int64(count))
}
```

Add to the `Read` literal, immediately after `EmbedInflight`:

```go
		EmbedChunksReusedTotal:   embedChunksReusedTotal.Load(),
```

- [ ] **Step 4: Publish and log it**

In `internal/metrics/expvar.go`, add inside `publish`, after the `embed_inflight` line:

```go
	expvar.Publish(expvarPrefix+"embed_chunks_reused_total", counterVar{value: &embedChunksReusedTotal})
```

In `internal/metrics/reporter.go`, add inside the `slog.LogAttrs` call, after the `embed_inflight` attribute:

```go
		slog.Int64("embed_chunks_reused_total", snapshot.EmbedChunksReusedTotal),
```

- [ ] **Step 5: Run the counter test and confirm it passes**

```bash
PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig" \
CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path' \
go test ./internal/metrics/ -run TestChunksReusedAccumulates
```

Expected: PASS.

- [ ] **Step 6: Write the failing call-site test**

Append to `internal/semantic/reuse_test.go`:

```go
// TestEmbedChunkBatchCountsReuseInMetrics proves a reuse hit reaches the
// process counter, so a status read reports what the run avoided rather than
// only what it spent. It asserts a delta because the counter is a
// package-level atomic shared across tests.
func TestEmbedChunkBatchCountsReuseInMetrics(t *testing.T) {
	service := &Service{embedder: &countingEmbedder{}}
	chunks := []model.StoredChunk{{Content: "reused-A"}, {Content: "fresh-B"}}
	reuse := map[string][]float32{contentVectorKey("reused-A"): {7, 7}}

	before := metrics.Read().EmbedChunksReusedTotal
	_, reused, err := service.embedChunkBatch(context.Background(), chunks, reuse)
	if err != nil {
		t.Fatalf("embedChunkBatch returned error: %v", err)
	}
	if reused != 1 {
		t.Fatalf("reused = %d, want 1", reused)
	}
	after := metrics.Read().EmbedChunksReusedTotal
	if after-before != 1 {
		t.Fatalf("EmbedChunksReusedTotal delta = %d, want 1", after-before)
	}
}
```

Add `"goodkind.io/lm-semantic-search/internal/metrics"` to that file's imports if it is absent.

- [ ] **Step 7: Run it and confirm it fails**

```bash
PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig" \
CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path' \
go test ./internal/semantic/ -run TestEmbedChunkBatchCountsReuseInMetrics
```

Expected: FAIL with `EmbedChunksReusedTotal delta = 0, want 1`.

- [ ] **Step 8: Count the hit at the embedder boundary**

In `internal/semantic/staging.go`, inside `embedChunkBatch`, immediately after the line `reused := len(chunkBatch) - len(missTexts)`:

```go
	metrics.ChunksReused(reused)
```

Add `"goodkind.io/lm-semantic-search/internal/metrics"` to that file's imports if it is absent.

This is the chokepoint rather than `insertChunksBatched`, because a reuse hit is decided here and a caller that never inserts still avoided the embed.

- [ ] **Step 9: Run both tests and confirm they pass**

```bash
PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig" \
CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path' \
go test ./internal/metrics/ ./internal/semantic/
```

Expected: `ok` for both packages.

- [ ] **Step 10: Commit**

```bash
git add internal/metrics internal/semantic
git commit -S -m "Count chunks served from a stored vector

embed_vectors_total counts only vectors that reached the embedder, because
embedChunkBatch passes the misses alone to embedMissedTexts. Nothing counted
the hits, so no surface could report what a run avoided.

Count a reuse hit at the same boundary EmbedBatchDone counts a batch, so the
two together report what a batch cost and what it skipped.

Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 2: Report file-change work

`lm-semantic-search job list` reports zero active while the daemon indexes a saved file, because a converge registers no job. The daemon holds the state in `BackgroundSync.converging` and `EventQueue.pending`; nothing reads it out.

**Files:**
- Modify: `internal/daemon/eventqueue.go`
- Modify: `internal/daemon/background_sync.go`
- Modify: `internal/daemon/manager.go`
- Test: `internal/daemon/eventqueue_test.go`
- Test: `internal/daemon/background_sync_activity_test.go` (create)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `daemon.WatcherActivity` struct with fields `CodebaseID string`, `State string`, `PendingPaths int`, `StartedAt time.Time`
  - constants `WatcherStateRunning = "running"` and `WatcherStateQueued = "queued"`
  - `(*EventQueue).PendingCounts() map[string]int`
  - `(*BackgroundSync).WatcherActivity() []WatcherActivity`
  - `(*Manager).SetWatcherActivityReporter(reporter WatcherActivityReporter)`
  - `(*Manager).WatcherActivity() []WatcherActivity`
  - `(*Manager).IndexSlots() (inUse int, total int)`
  - `(*Manager).StartedAt() time.Time`

- [ ] **Step 1: Write the failing queue test**

Append to `internal/daemon/eventqueue_test.go`:

```go
// TestEventQueuePendingCountsReportsWaitingPaths proves the queue can report
// the coalesced path count per codebase while the debounce timer is still
// running, which is the only record that file-change work is waiting.
func TestEventQueuePendingCountsReportsWaitingPaths(t *testing.T) {
	t.Parallel()

	queue := NewEventQueue(time.Hour, func(string, []string) {})
	queue.Enqueue("cb1", "a.go")
	queue.Enqueue("cb1", "a.go")
	queue.Enqueue("cb1", "b.go")
	queue.Enqueue("cb2", "x.go")

	counts := queue.PendingCounts()
	if counts["cb1"] != 2 {
		t.Fatalf("cb1 pending = %d, want 2 (a.go and b.go, coalesced)", counts["cb1"])
	}
	if counts["cb2"] != 1 {
		t.Fatalf("cb2 pending = %d, want 1", counts["cb2"])
	}
	if len(counts) != 2 {
		t.Fatalf("codebases with pending paths = %d, want 2", len(counts))
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig" \
CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path' \
go test ./internal/daemon/ -run TestEventQueuePendingCountsReportsWaitingPaths
```

Expected: compile failure, `queue.PendingCounts undefined`.

- [ ] **Step 3: Add the queue accessor**

Append to `internal/daemon/eventqueue.go`:

```go
// PendingCounts reports how many coalesced paths each codebase has waiting for
// its debounce timer. A status read uses it to report file-change work that is
// queued, which registers no job and so cannot be found in the job store. The
// returned map is a copy, so the caller may hold it past the lock.
func (queue *EventQueue) PendingCounts() map[string]int {
	queue.mu.Lock()
	defer queue.mu.Unlock()

	counts := make(map[string]int, len(queue.pending))
	for codebaseID, paths := range queue.pending {
		counts[codebaseID] = len(paths)
	}
	return counts
}
```

- [ ] **Step 4: Run it and confirm it passes**

Same command as Step 2. Expected: PASS.

- [ ] **Step 5: Read the converge bookkeeping before changing it**

Read `beginConverge` and `endConverge` in `internal/daemon/background_sync.go`. `converging` is `map[string]struct{}` today and carries no start time. The next step widens it to carry one, so a status row can report how long a converge has been running.

- [ ] **Step 6: Write the failing activity test**

Create `internal/daemon/background_sync_activity_test.go`:

```go
package daemon

import (
	"testing"
	"time"
)

// TestWatcherActivityReportsConvergingAndQueued proves the background syncer
// reports both the converge it is running and the paths still waiting on a
// debounce timer. Neither has a job record, so this is the only surface that
// can report them at all.
func TestWatcherActivityReportsConvergingAndQueued(t *testing.T) {
	t.Parallel()

	syncer := NewBackgroundSync(config.Config{}, nil)
	syncer.queue = NewEventQueue(time.Hour, func(string, []string) {})
	syncer.queue.Enqueue("cb-queued", "a.go")
	syncer.queue.Enqueue("cb-queued", "b.go")

	if !syncer.beginConverge("cb-running") {
		t.Fatal("beginConverge returned false on an idle codebase")
	}
	defer syncer.endConverge("cb-running")

	activity := syncer.WatcherActivity()
	byID := make(map[string]WatcherActivity, len(activity))
	for _, entry := range activity {
		byID[entry.CodebaseID] = entry
	}

	running, found := byID["cb-running"]
	if !found {
		t.Fatalf("converging codebase absent from activity: %+v", activity)
	}
	if running.State != WatcherStateRunning {
		t.Fatalf("running state = %q, want %q", running.State, WatcherStateRunning)
	}
	if running.StartedAt.IsZero() {
		t.Fatal("running entry carries no start time, so a surface cannot report its age")
	}

	queued, found := byID["cb-queued"]
	if !found {
		t.Fatalf("queued codebase absent from activity: %+v", activity)
	}
	if queued.State != WatcherStateQueued {
		t.Fatalf("queued state = %q, want %q", queued.State, WatcherStateQueued)
	}
	if queued.PendingPaths != 2 {
		t.Fatalf("queued pending paths = %d, want 2", queued.PendingPaths)
	}
}
```

Add `"goodkind.io/lm-semantic-search/internal/config"` to the imports.

- [ ] **Step 7: Run it and confirm it fails**

```bash
PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig" \
CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path' \
go test ./internal/daemon/ -run TestWatcherActivityReportsConvergingAndQueued
```

Expected: compile failure, `undefined: WatcherActivity`.

- [ ] **Step 8: Widen the converge record and add the reporter**

In `internal/daemon/background_sync.go`, change the field declaration:

```go
	convergeMu sync.Mutex
	converging map[string]time.Time
```

Change the constructor literal in `NewBackgroundSync`:

```go
		converging:           make(map[string]time.Time),
```

In `beginConverge`, store `clock.Now()` instead of `struct{}{}`. In `endConverge`, keep the `delete` unchanged.

Append to the same file:

```go
// Watcher activity states. They are the same words the status surface prints,
// and they are deliberately not the job lifecycle states: file-change work has
// no job record, so it has no job state.
const (
	WatcherStateRunning = "running"
	WatcherStateQueued  = "queued"
)

// WatcherActivity is one unit of file-change work the background syncer owns.
// It carries no job id because a converge registers no job, which is exactly
// why the job store cannot report it and this type exists.
type WatcherActivity struct {
	CodebaseID   string
	State        string
	PendingPaths int
	StartedAt    time.Time
}

// WatcherActivityReporter is the seam the manager holds so a status read can
// reach file-change work without the manager depending on the syncer.
type WatcherActivityReporter interface {
	WatcherActivity() []WatcherActivity
}

// WatcherActivity reports every converge running now and every codebase whose
// changed paths are waiting on a debounce timer, sorted by codebase id so two
// reads of an unchanged daemon render identically.
func (syncer *BackgroundSync) WatcherActivity() []WatcherActivity {
	syncer.convergeMu.Lock()
	running := make(map[string]time.Time, len(syncer.converging))
	for codebaseID, startedAt := range syncer.converging {
		running[codebaseID] = startedAt
	}
	syncer.convergeMu.Unlock()

	pending := map[string]int{}
	if syncer.queue != nil {
		pending = syncer.queue.PendingCounts()
	}

	activity := make([]WatcherActivity, 0, len(running)+len(pending))
	for codebaseID, startedAt := range running {
		activity = append(activity, WatcherActivity{
			CodebaseID:   codebaseID,
			State:        WatcherStateRunning,
			PendingPaths: pending[codebaseID],
			StartedAt:    startedAt,
		})
	}
	for codebaseID, count := range pending {
		if _, alreadyRunning := running[codebaseID]; alreadyRunning {
			continue
		}
		activity = append(activity, WatcherActivity{
			CodebaseID:   codebaseID,
			State:        WatcherStateQueued,
			PendingPaths: count,
			StartedAt:    time.Time{},
		})
	}
	sort.Slice(activity, func(first int, second int) bool {
		return activity[first].CodebaseID < activity[second].CodebaseID
	})
	return activity
}
```

Add `"sort"` to that file's imports if it is absent.

- [ ] **Step 9: Register the reporter on the manager**

In `internal/daemon/background_sync.go`, inside `Start`, immediately after `syncer.manager.SetCodebaseLifecycleHook(syncer)`:

```go
			syncer.manager.SetWatcherActivityReporter(syncer)
```

- [ ] **Step 10: Add the manager accessors**

In `internal/daemon/manager.go`, add to the `Manager` struct beside `lifecycleHook`:

```go
	// startedAt is when this daemon process built its manager. A status read
	// reports uptime from it, so the number comes from the process rather than
	// from a caller's clock.
	startedAt time.Time
	// watcherActivity reports file-change work, which registers no job. The
	// background syncer installs itself here at start; a manager with no
	// syncer reports no watcher activity rather than failing.
	watcherActivity      WatcherActivityReporter
	watcherActivityMutex sync.Mutex
```

Set `startedAt: clock.Now(),` in the `Manager` literal inside `NewManager`, and `watcherActivity: nil,` plus `watcherActivityMutex: sync.Mutex{},` so the literal stays exhaustive.

Append to the same file:

```go
// SetWatcherActivityReporter installs the reporter a status read uses to reach
// file-change work. The background syncer calls it at start.
func (manager *Manager) SetWatcherActivityReporter(reporter WatcherActivityReporter) {
	manager.watcherActivityMutex.Lock()
	defer manager.watcherActivityMutex.Unlock()
	manager.watcherActivity = reporter
}

// WatcherActivity reports file-change work, or nothing when no syncer is
// installed, which is the case in tests and in a daemon with file watching off.
func (manager *Manager) WatcherActivity() []WatcherActivity {
	manager.watcherActivityMutex.Lock()
	reporter := manager.watcherActivity
	manager.watcherActivityMutex.Unlock()
	if reporter == nil {
		return nil
	}
	return reporter.WatcherActivity()
}

// IndexSlots reports how many concurrent index jobs hold a slot and how many
// slots exist. A job that cannot take one stays queued, so the pair explains a
// queued job that is not waiting on any dependency.
func (manager *Manager) IndexSlots() (int, int) {
	return len(manager.indexSlots), cap(manager.indexSlots)
}

// StartedAt reports when this daemon process built its manager.
func (manager *Manager) StartedAt() time.Time {
	return manager.startedAt
}
```

- [ ] **Step 11: Run the daemon tests and confirm they pass**

```bash
PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig" \
CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path' \
go test ./internal/daemon/ -run 'TestEventQueue|TestWatcherActivity'
```

Expected: PASS. Then run the whole package to catch a broken `converging` caller:

```bash
PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig" \
CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path' \
go test ./internal/daemon/
```

- [ ] **Step 12: Commit**

```bash
git add internal/daemon
git commit -S -m "Report the file-change work that registers no job

A converge started by the file watcher writes no job record, so job list
reports zero active while the daemon indexes a saved file. The state exists
in the syncer's converging set and the event queue's pending paths; nothing
read it out.

Report both through one accessor, and carry a start time on a running
converge so a surface can report its age.

Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 3: The GetStatus contract and handler

**Files:**
- Modify: `proto/lmsemanticsearch/v1/service.proto`
- Create: `internal/daemon/status_metrics.go`
- Create: `internal/daemon/grpc_server_status.go`
- Test: `internal/daemon/status_metrics_test.go` (create)

**Interfaces:**
- Consumes: `metrics.Snapshot.EmbedChunksReusedTotal` from Task 1; `Manager.WatcherActivity`, `Manager.IndexSlots`, `Manager.StartedAt` from Task 2.
- Produces:
  - protobuf `Metric`, `ActivityRow`, `DaemonIdentity`, `GetStatusRequest`, `GetStatusResponse`
  - `daemon.buildStatusMetrics(manager *Manager, snapshot metrics.Snapshot, now time.Time) []*pb.Metric`
  - `daemon.buildStatusActivity(manager *Manager) []*pb.ActivityRow`

- [ ] **Step 1: Add the contract**

In `proto/lmsemanticsearch/v1/service.proto`, add to the `service` block after `rpc Doctor`:

```proto
  rpc GetStatus(GetStatusRequest) returns (GetStatusResponse);
```

Append these messages at the end of the file:

```proto
// Metric is one named observation. name is a string value rather than a
// protobuf field name, so protojson casing never touches it and the name reads
// identically on the terminal, in piped text, in JSON, and in the
// daemon.perf_counters log line. The value is a oneof because a oneof member
// has explicit presence: a plain proto3 scalar at its zero value is omitted by
// protojson, which would make false and absent indistinguishable.
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

// ActivityRow is one unit of work the daemon is doing or has queued, carried as
// metrics so every field uses the same naming as the counters. A row whose work
// the file watcher started has a job_id metric with no value set, because that
// work registers no job and the job commands cannot address it.
message ActivityRow {
  repeated Metric metrics = 1;
}

// DaemonIdentity names the running process.
message DaemonIdentity {
  string version = 1;
  string commit = 2;
  int32 pid = 3;
  string socket_path = 4;
  google.protobuf.Timestamp started_at = 5;
}

message GetStatusRequest {}

message GetStatusResponse {
  google.protobuf.Timestamp read_at = 1;
  DaemonIdentity daemon = 2;
  repeated Metric metrics = 3;
  repeated ActivityRow activity = 4;
  string display_text = 5;
}
```

- [ ] **Step 2: Generate and confirm the build breaks**

```bash
cd <worktree>
buf generate
PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig" \
CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path' \
go build ./...
```

Expected: `*GRPCServer does not implement SemanticSearchDaemonServiceServer (missing method GetStatus)`. `buf.gen.yaml` sets `require_unimplemented_servers=false`, so the missing handler is a compile error rather than a silent gap.

- [ ] **Step 3: Write the failing assembly test**

Create `internal/daemon/status_metrics_test.go`:

```go
package daemon

import (
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/metrics"
)

// metricByName finds one assembled metric so a test asserts on the name the
// operator sees rather than on a slice position.
func metricByName(list []*pb.Metric, name string) *pb.Metric {
	for _, metric := range list {
		if metric.GetName() == name {
			return metric
		}
	}
	return nil
}

// TestStatusMetricsCarryCounterNamesAndUnits proves the assembled list uses the
// counter identifiers verbatim, so a name on screen greps the daemon log, and
// that each carries the unit of the thing its increment call site counts.
func TestStatusMetricsCarryCounterNamesAndUnits(t *testing.T) {
	t.Parallel()

	snapshot := metrics.Snapshot{EmbedVectorsTotal: 3946, EmbedChunksReusedTotal: 4025}
	list := buildStatusMetrics(nil, snapshot, time.Unix(1785156767, 0))

	vectors := metricByName(list, "embed_vectors_total")
	if vectors == nil {
		t.Fatalf("embed_vectors_total absent from %d metrics", len(list))
	}
	if vectors.GetIntValue() != 3946 {
		t.Fatalf("embed_vectors_total = %d, want 3946", vectors.GetIntValue())
	}
	if vectors.GetUnit() != "vectors" {
		t.Fatalf("embed_vectors_total unit = %q, want \"vectors\"", vectors.GetUnit())
	}

	reused := metricByName(list, "embed_chunks_reused_total")
	if reused == nil || reused.GetIntValue() != 4025 || reused.GetUnit() != "chunks" {
		t.Fatalf("embed_chunks_reused_total = %+v, want 4025 chunks", reused)
	}
}

// TestStatusMetricsSetZeroValuedOneofs proves a zero counter still carries a
// value. protojson omits a plain proto3 scalar at its zero value, so a metric
// whose oneof is unset would be indistinguishable from an absent measurement.
func TestStatusMetricsSetZeroValuedOneofs(t *testing.T) {
	t.Parallel()

	list := buildStatusMetrics(nil, metrics.Snapshot{}, time.Unix(1785156767, 0))
	zero := metricByName(list, "embed_batches_failed")
	if zero == nil {
		t.Fatal("embed_batches_failed absent")
	}
	if zero.GetValue() == nil {
		t.Fatal("embed_batches_failed carries no value; zero and absent must differ")
	}
	if zero.GetIntValue() != 0 {
		t.Fatalf("embed_batches_failed = %d, want 0", zero.GetIntValue())
	}
}
```

- [ ] **Step 4: Run it and confirm it fails**

```bash
PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig" \
CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path' \
go test ./internal/daemon/ -run TestStatusMetrics
```

Expected: compile failure, `undefined: buildStatusMetrics`.

- [ ] **Step 5: Assemble the metric list**

Create `internal/daemon/status_metrics.go`. It holds the one table that names every counter and its unit, and the two builders that project the daemon's state through it.

```go
package daemon

import (
	"sort"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/metrics"
	"goodkind.io/lm-semantic-search/internal/model"
)

// Metric groups. They order the screen and nothing else; a consumer selects by
// name, which is unique across every group.
const (
	statusGroupDaemon     = "daemon"
	statusGroupDependency = "dependency_health"
	statusGroupJobs       = "jobs"
	statusGroupEmbed      = "embed"
	statusGroupConverge   = "converge"
	statusGroupRuntime    = "runtime"
	statusGroupCodebases  = "codebases"
	statusGroupActivity   = "activity"
)

// Units, named for what each counter's increment call site actually counts. A
// converge upsert is one path, and a skipped sync is one request, so neither
// borrows the other's noun.
const (
	unitSeconds    = "s"
	unitSlots      = "slots"
	unitJobs       = "jobs"
	unitRequests   = "requests"
	unitBatches    = "batches"
	unitVectors    = "vectors"
	unitChunks     = "chunks"
	unitPaths      = "paths"
	unitRuns       = "runs"
	unitFiles      = "files"
	unitRows       = "rows"
	unitGoroutines = "goroutines"
	unitCycles     = "cycles"
	unitBytes      = "bytes"
	unitMillis     = "ms"
	unitPercent    = "%"
	unitCodebases  = "codebases"
)

// intMetric builds one integer observation. Every builder goes through it so a
// zero value always sets the oneof, which is what keeps zero and absent apart
// once protojson drops unset plain scalars.
func intMetric(group string, name string, value int64, unit string) *pb.Metric {
	return &pb.Metric{
		Group: group,
		Name:  name,
		Unit:  unit,
		Value: &pb.Metric_IntValue{IntValue: value},
	}
}

func doubleMetric(group string, name string, value float64, unit string) *pb.Metric {
	return &pb.Metric{
		Group: group,
		Name:  name,
		Unit:  unit,
		Value: &pb.Metric_DoubleValue{DoubleValue: value},
	}
}

func boolMetric(group string, name string, value bool) *pb.Metric {
	return &pb.Metric{
		Group: group,
		Name:  name,
		Unit:  "",
		Value: &pb.Metric_BoolValue{BoolValue: value},
	}
}

func stringMetric(group string, name string, value string) *pb.Metric {
	return &pb.Metric{
		Group: group,
		Name:  name,
		Unit:  "",
		Value: &pb.Metric_StringValue{StringValue: value},
	}
}

// absentMetric builds an observation with no value, which every surface renders
// as null. A file-change row uses it for job_id, because that work has no job.
func absentMetric(group string, name string) *pb.Metric {
	return &pb.Metric{Group: group, Name: name, Unit: "", Value: nil}
}

// timeMetric renders a timestamp as UTC RFC3339, matching what protojson does
// with a Timestamp elsewhere in this service, or as absent when zero.
func timeMetric(group string, name string, value time.Time) *pb.Metric {
	if value.IsZero() {
		return absentMetric(group, name)
	}
	return stringMetric(group, name, value.UTC().Format(time.RFC3339Nano))
}

// buildStatusMetrics projects the process counters and the manager's own state
// into the ordered metric list. manager may be nil, which a unit test uses to
// exercise the counter projection alone; the manager-derived groups are then
// omitted rather than defaulted.
func buildStatusMetrics(manager *Manager, snapshot metrics.Snapshot, now time.Time) []*pb.Metric {
	list := make([]*pb.Metric, 0, 48)

	if manager != nil {
		list = append(list, intMetric(statusGroupDaemon, "uptime_s",
			int64(now.Sub(manager.StartedAt())/time.Second), unitSeconds))

		health := manager.DependencyHealth()
		list = append(list,
			boolMetric(statusGroupDependency, "dependency_health.degraded", health.Degraded()),
			stringMetric(statusGroupDependency, "dependency_health.mode", string(health.Mode)),
			timeMetric(statusGroupDependency, "dependency_health.since", health.Since),
			timeMetric(statusGroupDependency, "dependency_health.last_healthy_at", health.LastHealthyAt),
		)

		inUse, total := manager.IndexSlots()
		list = append(list,
			intMetric(statusGroupJobs, "index_slots_in_use", int64(inUse), unitSlots),
			intMetric(statusGroupJobs, "index_slots_total", int64(total), unitSlots),
		)
	}

	list = append(list,
		intMetric(statusGroupJobs, "jobs_active", snapshot.JobsActive, unitJobs),
		intMetric(statusGroupJobs, "jobs_completed_total", snapshot.JobsCompletedTotal, unitJobs),
		intMetric(statusGroupJobs, "jobs_failed_total", snapshot.JobsFailedTotal, unitJobs),
		intMetric(statusGroupJobs, "jobs_cancelled_total", snapshot.JobsCancelledTotal, unitJobs),
		intMetric(statusGroupJobs, "boot_resumes_total", snapshot.BootResumesTotal, unitJobs),
		intMetric(statusGroupJobs, "sync_skipped_inflight_total", snapshot.SyncSkippedInflightTotal, unitRequests),

		intMetric(statusGroupEmbed, "embed_inflight", snapshot.EmbedInflight, unitBatches),
		intMetric(statusGroupEmbed, "embed_batches_total", snapshot.EmbedBatchesTotal, unitBatches),
		intMetric(statusGroupEmbed, "embed_batches_failed", snapshot.EmbedBatchesFailed, unitBatches),
		intMetric(statusGroupEmbed, "embed_vectors_total", snapshot.EmbedVectorsTotal, unitVectors),
		intMetric(statusGroupEmbed, "embed_latency_ms_sum", snapshot.EmbedLatencyMSSum, unitMillis),
		intMetric(statusGroupEmbed, "embed_chunks_reused_total", snapshot.EmbedChunksReusedTotal, unitChunks),

		intMetric(statusGroupConverge, "converge_upsert_total", snapshot.ConvergeUpsertTotal, unitPaths),
		intMetric(statusGroupConverge, "converge_remove_total", snapshot.ConvergeRemoveTotal, unitPaths),
		intMetric(statusGroupConverge, "converge_copy_chunks_total", snapshot.ConvergeCopyChunksTotal, unitPaths),
		intMetric(statusGroupConverge, "sweep_runs_total", snapshot.SweepRunsTotal, unitRuns),
		intMetric(statusGroupConverge, "sweep_changed_total", snapshot.SweepChangedTotal, unitRuns),
	)

	list = append(list, runtimeMetrics()...)

	if manager != nil {
		list = append(list, codebaseMetrics(manager)...)
	}
	return list
}
```

Add the runtime and codebase helpers to the same file:

```go
// runtimeMetrics samples the Go runtime gauges the periodic
// daemon.perf_counters line already carries, from one ReadMemStats so the heap
// numbers are internally consistent.
func runtimeMetrics() []*pb.Metric {
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	return []*pb.Metric{
		intMetric(statusGroupRuntime, "num_goroutine", int64(runtime.NumGoroutine()), unitGoroutines),
		intMetric(statusGroupRuntime, "heap_alloc_bytes", int64(mem.HeapAlloc), unitBytes),
		intMetric(statusGroupRuntime, "heap_inuse_bytes", int64(mem.HeapInuse), unitBytes),
		intMetric(statusGroupRuntime, "num_gc", int64(mem.NumGC), unitCycles),
	}
}

// codebaseMetrics counts tracked codebases by their resolved display status.
// Every status the vocabulary defines gets a line even at zero, so the set of
// lines does not change as codebases move between states.
func codebaseMetrics(manager *Manager) []*pb.Metric {
	codebases := manager.ListIndexes(context.Background())
	counts := make(map[string]int, len(codebases))
	for _, codebase := range codebases {
		counts[manager.displayStatusFor(codebase)]++
	}

	names := make([]string, 0, len(counts))
	for name := range counts {
		names = append(names, name)
	}
	sort.Strings(names)

	list := make([]*pb.Metric, 0, len(names)+1)
	list = append(list, intMetric(statusGroupCodebases, "codebases_total", int64(len(codebases)), unitCodebases))
	for _, name := range names {
		list = append(list, intMetric(statusGroupCodebases, "codebases.status="+name, int64(counts[name]), unitCodebases))
	}
	return list
}
```

Add `"context"` and `"runtime"` to the imports.

`manager.displayStatusFor` may not exist under that name. Find the daemon-side resolver that already produces `Codebase.display_status` for `ListIndexes` and call it. Do not compute a status here; the whole point of the design is that this file never resolves one.

- [ ] **Step 6: Run the assembly tests and confirm they pass**

```bash
PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig" \
CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path' \
go test ./internal/daemon/ -run TestStatusMetrics
```

Expected: PASS.

- [ ] **Step 7: Build the activity rows**

Append to `internal/daemon/status_metrics.go`:

```go
// activitySourceJob and activitySourceWatcher name what started a unit of work.
// A watcher row has no job id, which is why the two are worth telling apart.
const (
	activitySourceJob     = "job"
	activitySourceWatcher = "watcher"
)

// buildStatusActivity reports every unit of work the daemon is doing or has
// queued, from both sources. Registered jobs come from the job store; the
// file-change converges come from the syncer, because they register no job and
// the job store therefore cannot report them.
func buildStatusActivity(manager *Manager) []*pb.ActivityRow {
	rows := make([]*pb.ActivityRow, 0, 8)
	for _, job := range manager.ActiveJobs() {
		rows = append(rows, jobActivityRow(job))
	}
	for _, entry := range manager.WatcherActivity() {
		rows = append(rows, watcherActivityRow(manager, entry))
	}
	return rows
}

func jobActivityRow(job model.Job) *pb.ActivityRow {
	list := []*pb.Metric{
		stringMetric(statusGroupActivity, "job_id", job.ID),
		stringMetric(statusGroupActivity, "source", activitySourceJob),
		stringMetric(statusGroupActivity, "codebase_id", job.CodebaseID),
		stringMetric(statusGroupActivity, "canonical_path", job.CanonicalPath),
		stringMetric(statusGroupActivity, "operation", job.Operation),
		stringMetric(statusGroupActivity, "state", string(job.State)),
		stringMetric(statusGroupActivity, "trigger", job.Trigger),
		boolMetric(statusGroupActivity, "forced", job.Forced),
		stringMetric(statusGroupActivity, "phase", job.Progress.Phase),
		doubleMetric(statusGroupActivity, "overall_percent", job.Progress.OverallPercent, unitPercent),
		doubleMetric(statusGroupActivity, "phase_percent", job.Progress.PhasePercent, unitPercent),
		intMetric(statusGroupActivity, "files_processed", int64(job.Progress.FilesProcessed), unitFiles),
		intMetric(statusGroupActivity, "files_total", int64(job.Progress.FilesTotal), unitFiles),
		intMetric(statusGroupActivity, "chunks_generated", int64(job.Progress.ChunksGenerated), unitChunks),
		intMetric(statusGroupActivity, "chunks_embedded", int64(job.Progress.ChunksEmbedded), unitChunks),
		intMetric(statusGroupActivity, "chunks_reused", int64(job.Progress.ChunksReused), unitChunks),
		intMetric(statusGroupActivity, "chunks_dropped", int64(job.Progress.ChunksDropped), unitChunks),
		intMetric(statusGroupActivity, "reuse_vectors_loaded", int64(job.Progress.ReuseVectorsLoaded), unitVectors),
		intMetric(statusGroupActivity, "collection_rows_written", int64(job.Progress.CollectionRowsWritten), unitRows),
		intMetric(statusGroupActivity, "embedding_batches_completed", int64(job.Progress.EmbeddingBatchesCompleted), unitBatches),
		intMetric(statusGroupActivity, "embedding_batches_total", int64(job.Progress.EmbeddingBatchesTotal), unitBatches),
		timeMetric(statusGroupActivity, "started_at", job.StartedAt),
		timeMetric(statusGroupActivity, "last_event_at", job.Progress.LastEventAt),
		timeMetric(statusGroupActivity, "heartbeat_at", job.Progress.HeartbeatAt),
	}
	return &pb.ActivityRow{Metrics: list}
}

// watcherActivityRow carries an absent job_id rather than an empty string. The
// work has no job record, and an empty id would read as one the job commands
// could accept.
func watcherActivityRow(manager *Manager, entry WatcherActivity) *pb.ActivityRow {
	list := []*pb.Metric{
		absentMetric(statusGroupActivity, "job_id"),
		stringMetric(statusGroupActivity, "source", activitySourceWatcher),
		stringMetric(statusGroupActivity, "codebase_id", entry.CodebaseID),
		stringMetric(statusGroupActivity, "canonical_path", manager.CanonicalPathFor(entry.CodebaseID)),
		stringMetric(statusGroupActivity, "operation", "converge"),
		stringMetric(statusGroupActivity, "state", entry.State),
		intMetric(statusGroupActivity, "pending_paths", int64(entry.PendingPaths), unitPaths),
		timeMetric(statusGroupActivity, "started_at", entry.StartedAt),
	}
	return &pb.ActivityRow{Metrics: list}
}
```

`manager.ActiveJobs()` and `manager.CanonicalPathFor(id)` may not exist under those names. Find the existing manager accessors that `ListJobs` and `GetIndex` already use for the same two facts and call those instead of adding new ones.

- [ ] **Step 8: Write the failing handler test**

Append to `internal/daemon/status_metrics_test.go`:

```go
// TestWatcherActivityRowCarriesAbsentJobID proves a file-change row states it
// has no job, rather than carrying an empty id a reader could mistake for one
// the job commands accept.
func TestWatcherActivityRowCarriesAbsentJobID(t *testing.T) {
	t.Parallel()

	row := watcherActivityRow(nil, WatcherActivity{
		CodebaseID:   "cb-1",
		State:        WatcherStateQueued,
		PendingPaths: 12,
	})
	jobID := metricByName(row.GetMetrics(), "job_id")
	if jobID == nil {
		t.Fatal("job_id absent from the row entirely; it must be present with no value")
	}
	if jobID.GetValue() != nil {
		t.Fatalf("job_id carries a value %+v, want none", jobID.GetValue())
	}
	pending := metricByName(row.GetMetrics(), "pending_paths")
	if pending.GetIntValue() != 12 || pending.GetUnit() != unitPaths {
		t.Fatalf("pending_paths = %+v, want 12 paths", pending)
	}
}
```

If `watcherActivityRow` dereferences the manager for the canonical path, guard that lookup so a nil manager yields an empty path, or build the row with a real manager in the test. Do not change the production behavior to suit the test.

- [ ] **Step 9: Add the handler**

Create `internal/daemon/grpc_server_status.go`:

```go
package daemon

import (
	"context"
	"os"

	"goodkind.io/gklog/version"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/metrics"
	render "goodkind.io/lm-semantic-search/internal/render"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// GetStatus reports what the daemon is doing and what its counters read, as one
// flat list of named observations. It reports only non-terminal work, so the
// reply stays a few kilobytes rather than growing with job history the way
// ListJobs does.
func (server *GRPCServer) GetStatus(ctx context.Context, request *pb.GetStatusRequest) (resp *pb.GetStatusResponse, err error) {
	ctx, done := beginRPC(ctx, "GetStatus")
	defer done(&err)
	_ = request

	now := clock.Now()
	snapshot := metrics.Read()
	statusMetrics := buildStatusMetrics(server.manager, snapshot, now)
	activity := buildStatusActivity(server.manager)

	response := &pb.GetStatusResponse{
		ReadAt: timestamppb.New(now),
		Daemon: &pb.DaemonIdentity{
			Version:    version.String(),
			Commit:     version.Commit,
			Pid:        int32(os.Getpid()),
			SocketPath: server.manager.config.SocketPath,
			StartedAt:  timestamppb.New(server.manager.StartedAt()),
		},
		Metrics:     statusMetrics,
		Activity:    activity,
		DisplayText: "",
	}
	health := server.manager.DependencyHealth()
	response.DisplayText = server.envelopeText(ctx, health, render.StatusMetrics(response))
	return response, nil
}
```

`os.Getpid()` returns an `int`; convert it the way `currentClientInfo` in `cmd/lm-semantic-search/rpc.go` does, rejecting a value that does not fit `int32` rather than truncating silently.

`render.StatusMetrics` arrives in Task 4. Until then, pass the empty string so this task compiles and its tests run.

- [ ] **Step 10: Build and run the package**

```bash
PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig" \
CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path' \
go build ./... && \
PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig" \
CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path' \
go test ./internal/daemon/
```

Expected: build succeeds and the package tests pass.

- [ ] **Step 11: Commit**

```bash
git add proto gen internal/daemon
git commit -S -m "Add GetStatus, one call for what the daemon is doing

Report the counters and the active work as a flat list of named observations.
The name is a string value rather than a protobuf field name, so protojson
casing never touches it and embed_vectors_total reads identically on every
surface and in the perf counter log line.

Carry the value in a oneof. protojson omits a plain proto3 scalar at its zero
value, which is why job list returns a dependencyHealth object today with no
degraded and no mode; a oneof member has explicit presence, so zero, false,
empty string, and absent stay four distinguishable facts.

Report file-change converges alongside registered jobs, with an absent job id,
so the one surface that can see that work says plainly why the job commands
cannot address it.

Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 4: The piped renderer and the command

**Files:**
- Create: `internal/render/status_metrics_text.go`
- Create: `cmd/lm-semantic-search/status.go`
- Modify: `cmd/lm-semantic-search/grpc_server_status.go` call site from Task 3, Step 9
- Modify: `cmd/lm-semantic-search/root.go`
- Test: `internal/render/status_metrics_text_test.go` (create)

**Interfaces:**
- Consumes: `pb.GetStatusResponse` from Task 3.
- Produces: `render.StatusMetrics(response *pb.GetStatusResponse) string`, and the `status` command.

- [ ] **Step 1: Write the failing renderer test**

Create `internal/render/status_metrics_text_test.go`:

```go
package render

import (
	"strings"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

// TestStatusMetricsPrintsOneRecordPerLine proves the piped form is one
// whitespace-separated record per line, so grep, awk, and cut work on it, and
// that a value carries its unit as a third field.
func TestStatusMetricsPrintsOneRecordPerLine(t *testing.T) {
	t.Parallel()

	response := &pb.GetStatusResponse{
		Metrics: []*pb.Metric{
			{Group: "embed", Name: "embed_vectors_total", Unit: "vectors",
				Value: &pb.Metric_IntValue{IntValue: 3946}},
			{Group: "dependency_health", Name: "dependency_health.degraded",
				Value: &pb.Metric_BoolValue{BoolValue: false}},
			{Group: "dependency_health", Name: "dependency_health.since"},
		},
	}

	lines := strings.Split(strings.TrimSpace(StatusMetrics(response)), "\n")
	got := map[string]string{}
	for _, line := range lines {
		fields := strings.SplitN(line, " ", 2)
		got[fields[0]] = fields[1]
	}

	if got["embed_vectors_total"] != "3946 vectors" {
		t.Fatalf("embed_vectors_total line = %q, want \"3946 vectors\"", got["embed_vectors_total"])
	}
	if got["dependency_health.degraded"] != "false" {
		t.Fatalf("degraded line = %q, want \"false\"", got["dependency_health.degraded"])
	}
	if got["dependency_health.since"] != "null" {
		t.Fatalf("absent value line = %q, want \"null\"", got["dependency_health.since"])
	}
}

// TestStatusMetricsPrintsRawDigits proves the piped form does not group digits.
// That output is parsed, so a separator would break every consumer.
func TestStatusMetricsPrintsRawDigits(t *testing.T) {
	t.Parallel()

	response := &pb.GetStatusResponse{
		Metrics: []*pb.Metric{
			{Group: "runtime", Name: "heap_alloc_bytes", Unit: "bytes",
				Value: &pb.Metric_IntValue{IntValue: 249278160}},
		},
	}
	if !strings.Contains(StatusMetrics(response), "heap_alloc_bytes 249278160 bytes") {
		t.Fatalf("piped output grouped digits:\n%s", StatusMetrics(response))
	}
}

// TestStatusMetricsIndexesActivityRows proves an activity field is addressable
// by its row position, so the flat form keeps the rows apart.
func TestStatusMetricsIndexesActivityRows(t *testing.T) {
	t.Parallel()

	response := &pb.GetStatusResponse{
		Activity: []*pb.ActivityRow{
			{Metrics: []*pb.Metric{{Name: "job_id", Value: &pb.Metric_StringValue{StringValue: "job_a"}}}},
			{Metrics: []*pb.Metric{{Name: "job_id"}, {Name: "pending_paths", Unit: "paths",
				Value: &pb.Metric_IntValue{IntValue: 8}}}},
		},
	}
	text := StatusMetrics(response)
	for _, want := range []string{
		"activity.0.job_id job_a",
		"activity.1.job_id null",
		"activity.1.pending_paths 8 paths",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("missing %q in:\n%s", want, text)
		}
	}
}
```

- [ ] **Step 2: Run them and confirm they fail**

```bash
PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig" \
CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path' \
go test ./internal/render/ -run TestStatusMetrics
```

Expected: compile failure, `undefined: StatusMetrics`.

- [ ] **Step 3: Write the renderer**

Create `internal/render/status_metrics_text.go`:

```go
package render

import (
	"strconv"
	"strings"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

// StatusMetrics renders one status reply as one record per line, formatted
// "name value unit" with single spaces. The output is parsed by consumers, so
// digits are raw and a record with no unit ends after the value.
func StatusMetrics(response *pb.GetStatusResponse) string {
	var builder strings.Builder
	for _, metric := range response.GetMetrics() {
		writeMetricLine(&builder, metric.GetName(), metric)
	}
	for index, row := range response.GetActivity() {
		prefix := "activity." + strconv.Itoa(index) + "."
		for _, metric := range row.GetMetrics() {
			writeMetricLine(&builder, prefix+metric.GetName(), metric)
		}
	}
	return strings.TrimRight(builder.String(), "\n")
}

// writeMetricLine emits one record. An unset value prints null, which is how
// every surface says a fact is absent rather than zero or empty.
func writeMetricLine(builder *strings.Builder, name string, metric *pb.Metric) {
	builder.WriteString(name)
	builder.WriteString(" ")
	builder.WriteString(metricValueText(metric))
	if unit := metric.GetUnit(); unit != "" {
		builder.WriteString(" ")
		builder.WriteString(unit)
	}
	builder.WriteString("\n")
}

// metricValueText renders the set oneof member. A string renders bare except
// when it is empty, which prints as a quoted empty string so it is not confused
// with an absent value.
func metricValueText(metric *pb.Metric) string {
	switch value := metric.GetValue().(type) {
	case *pb.Metric_IntValue:
		return strconv.FormatInt(value.IntValue, 10)
	case *pb.Metric_DoubleValue:
		return strconv.FormatFloat(value.DoubleValue, 'f', -1, 64)
	case *pb.Metric_BoolValue:
		return strconv.FormatBool(value.BoolValue)
	case *pb.Metric_StringValue:
		if value.StringValue == "" {
			return `""`
		}
		return value.StringValue
	default:
		return "null"
	}
}
```

- [ ] **Step 4: Wire the renderer into the handler**

In `internal/daemon/grpc_server_status.go`, replace the empty display text with the real call:

```go
	response.DisplayText = server.envelopeText(ctx, health, render.StatusMetrics(response))
```

- [ ] **Step 5: Run the renderer tests and confirm they pass**

```bash
PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig" \
CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path' \
go test ./internal/render/ ./internal/daemon/
```

Expected: `ok` for both.

- [ ] **Step 6: Add the command**

Create `cmd/lm-semantic-search/status.go`:

```go
package main

import (
	"context"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"golang.org/x/term"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/response"
)

// defaultStatusInterval is how often the live screen re-reads the daemon.
const defaultStatusInterval = 2 * time.Second

// minimumStatusInterval floors the refresh cadence so several open screens
// cannot turn into a busy loop against the daemon socket.
const minimumStatusInterval = 500 * time.Millisecond

func newStatusCmd(options *rootOptions) *cobra.Command {
	var once bool
	var interval time.Duration

	cmd := &cobra.Command{
		Use:   "status",
		Short: "Show what the daemon is doing and what its counters read",
		Long: strings.Join([]string{
			"Show what the daemon is doing and what its counters read.",
			"",
			"On a terminal this refreshes in place and shows the change since the",
			"previous read. Piped or under --json it prints one snapshot and exits.",
		}, "\n"),
		Args: requireNoArgs("status"),
		Example: strings.Join([]string{
			"  lm-semantic-search status",
			"  lm-semantic-search status --once",
			"  lm-semantic-search --json status",
			"  lm-semantic-search status | grep embed_vectors_total",
		}, "\n"),
		RunE: func(cmd *cobra.Command, args []string) error {
			cliOpts := options.cliOptions()
			if interval < minimumStatusInterval {
				interval = minimumStatusInterval
			}
			live := cliOpts.outputMode == response.ModeHuman &&
				!once &&
				term.IsTerminal(int(os.Stdout.Fd()))
			if live {
				return runStatusTUI(cliOpts, interval)
			}
			return callAndPrint(cliOpts, func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
				return client.GetStatus(ctx, &pb.GetStatusRequest{})
			})
		},
	}
	cmd.Flags().BoolVar(&once, "once", false, "print one snapshot even on a terminal")
	cmd.Flags().DurationVar(&interval, "interval", defaultStatusInterval, "refresh cadence for the live screen")
	return cmd
}
```

In `cmd/lm-semantic-search/root.go`, add to `newRoot` after `root.AddCommand(newCodebaseCmd(options))`:

```go
	root.AddCommand(newStatusCmd(options))
```

`runStatusTUI` arrives in Task 5. Until then, stub it in `status.go` so this task compiles:

```go
// runStatusTUI drives the live screen. Task 5 replaces this body.
func runStatusTUI(options cliOptions, interval time.Duration) error {
	_ = interval
	return callAndPrint(options, func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
		return client.GetStatus(ctx, &pb.GetStatusRequest{})
	})
}
```

- [ ] **Step 7: Confirm the display guard still passes**

```bash
PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig" \
CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path' \
go test ./cmd/lm-semantic-search/
```

Expected: PASS, including `TestCLIDisplayDoesNotReadRawStatusFields`. The command reads name and value pairs and calls no named status getter, so the guard has nothing to catch.

- [ ] **Step 8: Prove the three forms against the running daemon**

```bash
make build GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile
./bin/lm-semantic-search status | head -20
./bin/lm-semantic-search status | grep embed_vectors_total
./bin/lm-semantic-search --json status | jq -r '.metrics[] | select(.name=="embed_vectors_total") | .intValue'
./bin/lm-semantic-search --json status | jq '.metrics[] | select(.name=="dependency_health.degraded")'
```

The fourth command must print an object containing `"boolValue": false`. If the key is missing, the value is not in a `oneof` and the zero-versus-absent distinction is lost.

- [ ] **Step 9: Commit**

```bash
git add internal/render cmd/lm-semantic-search
git commit -S -m "Add the status command with its piped and JSON forms

Print one record per line as name, value, unit, with raw digits, so grep, awk,
and cut work on the output. An absent value prints null and an empty string
prints a quoted empty string, so the two do not collapse.

Take the JSON form from protojson unchanged, the same path every other command
uses, because the reply already carries the names as values.

Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 5: The live screen

**Files:**
- Create: `cmd/lm-semantic-search/status_tui.go`
- Modify: `cmd/lm-semantic-search/status.go` (remove the Task 4 stub)
- Test: `cmd/lm-semantic-search/status_tui_test.go` (create)

**Interfaces:**
- Consumes: `pb.GetStatusResponse` and the `status` command from Task 4.
- Produces: `runStatusTUI(options cliOptions, interval time.Duration) error`.

- [ ] **Step 1: Write the failing formatting tests**

Create `cmd/lm-semantic-search/status_tui_test.go`:

```go
package main

import (
	"strings"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

// TestGroupDigitsSeparatesThousands proves the terminal groups digits, which is
// what makes a value crossing a digit boundary visible without reading it.
func TestGroupDigitsSeparatesThousands(t *testing.T) {
	t.Parallel()

	cases := map[string]string{
		"0":         "0",
		"824":       "824",
		"1524":      "1,524",
		"249278160": "249,278,160",
		"-1":        "-1",
		"-12345":    "-12,345",
	}
	for input, want := range cases {
		if got := groupDigits(input); got != want {
			t.Fatalf("groupDigits(%q) = %q, want %q", input, got, want)
		}
	}
}

// TestDeltaTextSignsTheChange proves a delta always carries its sign, so a
// reader never has to infer direction, and that an unchanged constant carries
// no delta at all.
func TestDeltaTextSignsTheChange(t *testing.T) {
	t.Parallel()

	if got := deltaText(72, true); got != "+72" {
		t.Fatalf("deltaText(72) = %q, want \"+72\"", got)
	}
	if got := deltaText(-1, true); got != "-1" {
		t.Fatalf("deltaText(-1) = %q, want \"-1\"", got)
	}
	if got := deltaText(16104, true); got != "+16,104" {
		t.Fatalf("deltaText(16104) = %q, want \"+16,104\"", got)
	}
	if got := deltaText(0, false); got != "" {
		t.Fatalf("deltaText with no prior read = %q, want empty", got)
	}
}

// TestStatusRowsCarryNullForAbsentValues proves the screen prints null for a
// metric with no value, so an absent fact never renders as a zero.
func TestStatusRowsCarryNullForAbsentValues(t *testing.T) {
	t.Parallel()

	response := &pb.GetStatusResponse{
		Metrics: []*pb.Metric{
			{Group: "dependency_health", Name: "dependency_health.since"},
			{Group: "embed", Name: "embed_vectors_total", Unit: "vectors",
				Value: &pb.Metric_IntValue{IntValue: 3946}},
		},
	}
	body := statusCounterBlock(response, nil, 100)
	if !strings.Contains(body, "null") {
		t.Fatalf("absent value did not render as null:\n%s", body)
	}
	if !strings.Contains(body, "3,946") {
		t.Fatalf("value was not digit-grouped:\n%s", body)
	}
}
```

- [ ] **Step 2: Run them and confirm they fail**

```bash
PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig" \
CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path' \
go test ./cmd/lm-semantic-search/ -run 'TestGroupDigits|TestDeltaText|TestStatusRows'
```

Expected: compile failure, `undefined: groupDigits`.

- [ ] **Step 3: Write the screen**

Create `cmd/lm-semantic-search/status_tui.go` modelled on `codebase_list_tui.go`, reusing its `padTo`, `padLeftTo`, `fitHead`, `fitTail`, `clampInt`, and `keyMatches` helpers plus its Lip Gloss styles rather than declaring new ones.

The model holds the current response, the previous response keyed by metric name for the delta column, the terminal size, a paused flag, the last successful read time, and the last refresh error.

`statusCounterBlock(response *pb.GetStatusResponse, previous map[string]int64, width int) string` lays out four columns: the name left-aligned, the value right-aligned and digit-grouped, the unit left-aligned, and the delta right-aligned. Column widths come from one pass over the metrics, so they are stable for a given terminal width. A blank line separates each group change.

`statusActivityBlock` renders each row as an indented `key=value` block, digit-grouped, with `null` for an absent value. It scrolls; the counter block does not.

The header line carries `version`, `pid`, `uptime_s`, the socket path, `read_at`, and `interval`. When paused it reads `paused_at` instead. When a refresh fails it keeps the last successful `read_at` and appends `refresh_error="<message>"`.

Refresh polls `GetStatus`; the daemon's `WatchJobs` sends one message per requested job id and returns, so it is a snapshot rather than a subscription and cannot drive a live screen.

Add these two helpers:

```go
// groupDigits inserts a comma every three digits from the right, preserving a
// leading sign. The terminal groups digits so a value crossing a digit boundary
// is visible without reading it; the piped and JSON forms keep raw digits
// because both are parsed.
func groupDigits(digits string) string {
	sign := ""
	if strings.HasPrefix(digits, "-") {
		sign = "-"
		digits = digits[1:]
	}
	if len(digits) <= 3 {
		return sign + digits
	}
	var parts []string
	for len(digits) > 3 {
		parts = append([]string{digits[len(digits)-3:]}, parts...)
		digits = digits[:len(digits)-3]
	}
	parts = append([]string{digits}, parts...)
	return sign + strings.Join(parts, ",")
}

// deltaText renders the change since the previous read. It always carries a
// sign so direction never has to be inferred, and it is empty when there is no
// previous read to compare against.
func deltaText(delta int64, hasPrevious bool) string {
	if !hasPrevious {
		return ""
	}
	if delta < 0 {
		return groupDigits(strconv.FormatInt(delta, 10))
	}
	return "+" + groupDigits(strconv.FormatInt(delta, 10))
}
```

- [ ] **Step 4: Remove the Task 4 stub**

Delete the stub `runStatusTUI` from `cmd/lm-semantic-search/status.go`; the real one now lives in `status_tui.go`.

- [ ] **Step 5: Run the tests and confirm they pass**

```bash
PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig" \
CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path' \
go test ./cmd/lm-semantic-search/
```

Expected: PASS, including the display guard.

- [ ] **Step 6: Drive the screen against the running daemon**

```bash
make build GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile
./bin/lm-semantic-search status
```

Confirm by watching, then report what you observed rather than that it worked:

- the counter block holds its position while numbers change
- a counter that moves shows a signed delta and one that does not shows `+0`
- `index_slots_total` shows no delta, because it is a constant
- `p` pauses and the header reads `paused_at`
- `q` exits cleanly and leaves the terminal usable

Then confirm the reporting gap is closed. In a second terminal, edit one file under a tracked codebase, and check that a row appears with `source=watcher` and `job_id=null` while `lm-semantic-search job list` reports `Active: 0 queued, 0 running, 0 canceling`.

- [ ] **Step 7: Run every gate**

```bash
make fmt GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile
make build GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile
make test GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile
```

All three must pass, including the vet, lint-golangci, lint-format, lint-gocyclo, lint-deadcode, staticcheck-extra, and govulncheck gates. Fix any failure in the code rather than weakening the check.

- [ ] **Step 8: Commit**

```bash
git add cmd/lm-semantic-search
git commit -S -m "Add the live status screen

Poll the daemon and show the change since the previous read as a fourth
column. Two observations, not a derived rate: embed_latency_ms_sum and
embed_batches_total are both on screen and their quotient is the reader's.

Hold the counter block in place and give the activity block the scrolling, so
a two-second refresh does not reflow the screen.

Keep the last successful read time and append the error when a refresh fails,
so a dead connection never reads as a quiet system.

Co-authored-by: Claude <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage.** Rules covered by Tasks 3 through 5. Numbers and units by Tasks 3 and 5. Busy, idle, degraded, and narrow screens by Task 5. Vertical space by Task 5. Detail on enter is deferred and named below. Liveness and keys by Task 5. Piped output by Task 4. JSON by Tasks 3 and 4. Terminal detection by Task 4. Values the display reads by Task 3.

**Deferred from the design, deliberately.** The detail view on `enter` is not in any task. The list already carries every field the detail view would show except the full path, which the list truncates. Ship the list first and add the detail view once the column widths are settled against real terminals.

**Two names to resolve during execution.** `manager.displayStatusFor`, `manager.ActiveJobs`, and `manager.CanonicalPathFor` are placeholders for accessors that already exist under other names, since `ListIndexes`, `ListJobs`, and `GetIndex` all read those same facts today. Find and call the existing ones. Do not add a parallel accessor, and do not resolve a display status inside `status_metrics.go`.

**Type consistency.** `WatcherActivity` carries `State string` and the constants `WatcherStateRunning` and `WatcherStateQueued`, used identically in Tasks 2 and 3. `Metric` is built only through `intMetric`, `doubleMetric`, `boolMetric`, `stringMetric`, `absentMetric`, and `timeMetric`, so no call site can leave a zero value with an unset `oneof`. `groupDigits` takes and returns a string in both its definition and its test.
