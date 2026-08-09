package semantic

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/metrics"
)

type residencyLogHandler struct {
	mutex   sync.Mutex
	records []slog.Record
}

func (handler *residencyLogHandler) Enabled(context.Context, slog.Level) bool {
	return true
}

func (handler *residencyLogHandler) Handle(_ context.Context, record slog.Record) error {
	handler.mutex.Lock()
	defer handler.mutex.Unlock()
	handler.records = append(handler.records, record.Clone())
	return nil
}

func (handler *residencyLogHandler) WithAttrs([]slog.Attr) slog.Handler {
	return handler
}

func (handler *residencyLogHandler) WithGroup(string) slog.Handler {
	return handler
}

func (handler *residencyLogHandler) record(message string) (slog.Record, bool) {
	handler.mutex.Lock()
	defer handler.mutex.Unlock()
	for _, record := range handler.records {
		if record.Message == message {
			return record, true
		}
	}
	return slog.Record{}, false
}

func captureResidencyLogs(t *testing.T) *residencyLogHandler {
	t.Helper()
	handler := &residencyLogHandler{}
	previous := slog.Default()
	slog.SetDefault(slog.New(handler))
	t.Cleanup(func() {
		slog.SetDefault(previous)
	})
	return handler
}

func requireResidencyEventFields(t *testing.T, record slog.Record) {
	t.Helper()
	want := map[string]bool{
		"collection":  false,
		"state":       false,
		"progress":    false,
		"elapsed_ms":  false,
		"lease_count": false,
		"idle_ms":     false,
		"error_class": false,
	}
	record.Attrs(func(attribute slog.Attr) bool {
		if _, found := want[attribute.Key]; found {
			want[attribute.Key] = true
		}
		return true
	})
	for name, found := range want {
		if !found {
			t.Errorf("%s missing %q", record.Message, name)
		}
	}
}

func requireResidencyEventValues(
	t *testing.T,
	record slog.Record,
	progress int64,
	elapsedMS int64,
	leaseCount int64,
	idleMS int64,
) {
	t.Helper()
	want := map[string]int64{
		"progress":    progress,
		"elapsed_ms":  elapsedMS,
		"lease_count": leaseCount,
		"idle_ms":     idleMS,
	}
	record.Attrs(func(attribute slog.Attr) bool {
		wantValue, found := want[attribute.Key]
		if found && attribute.Value.Int64() != wantValue {
			t.Errorf(
				"%s %s = %d want %d",
				record.Message,
				attribute.Key,
				attribute.Value.Int64(),
				wantValue,
			)
		}
		return true
	})
}

func TestResidencyEmitsLoadAndUnloadMetricsAndEvents(t *testing.T) {
	handler := captureResidencyLogs(t)
	before := metrics.Read()
	clock := newTestResidencyClock()
	unloaded := make(chan struct{}, 1)
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: 15 * time.Second,
		idleTimeout: time.Minute,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			return nil
		},
		unload: func(context.Context, string) error {
			clock.Advance(250 * time.Millisecond)
			unloaded <- struct{}{}
			return nil
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	lease, err := controller.Acquire(context.Background(), "live")
	if err != nil {
		t.Fatalf("Acquire returned error: %v", err)
	}
	lease.Release()
	clock.Advance(75 * time.Second)
	select {
	case <-unloaded:
	case <-time.After(time.Second):
		t.Fatal("idle unload did not run")
	}

	deadline := time.Now().Add(time.Second)
	for {
		if _, found := handler.record("semantic.collection_unloaded"); found {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("unload completion event was not emitted")
		}
		time.Sleep(time.Millisecond)
	}

	after := metrics.Read()
	if after.MilvusCollectionLoadsTotal-before.MilvusCollectionLoadsTotal != 1 {
		t.Fatalf("load delta = %d, want 1", after.MilvusCollectionLoadsTotal-before.MilvusCollectionLoadsTotal)
	}
	if after.MilvusCollectionUnloadsTotal-before.MilvusCollectionUnloadsTotal != 1 {
		t.Fatalf("unload delta = %d, want 1", after.MilvusCollectionUnloadsTotal-before.MilvusCollectionUnloadsTotal)
	}
	if after.MilvusCollectionLeasesActive != before.MilvusCollectionLeasesActive {
		t.Fatalf("active leases = %d, want %d", after.MilvusCollectionLeasesActive, before.MilvusCollectionLeasesActive)
	}
	wantEvents := map[string][4]int64{
		"semantic.collection_load_started":   {0, 0, 1, 0},
		"semantic.collection_load_ready":     {100, 0, 1, 0},
		"semantic.collection_unload_started": {0, 0, 0, 75000},
		"semantic.collection_unloaded":       {100, 250, 0, 75000},
	}
	for message, want := range wantEvents {
		record, found := handler.record(message)
		if !found {
			t.Errorf("missing event %q", message)
			continue
		}
		requireResidencyEventFields(t, record)
		requireResidencyEventValues(t, record, want[0], want[1], want[2], want[3])
	}
}

func TestResidencyReportsLoadWaitTimeout(t *testing.T) {
	handler := captureResidencyLogs(t)
	before := metrics.Read()
	clock := newTestResidencyClock()
	finishLoad := make(chan struct{})
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: 15 * time.Second,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			<-finishLoad
			return nil
		},
	})
	t.Cleanup(func() {
		close(finishLoad)
		_ = controller.Close(context.Background())
	})

	result := make(chan error, 1)
	go func() {
		_, err := controller.Acquire(context.Background(), "live")
		result <- err
	}()
	waitForLeaseCount(t, controller, "live", 1)
	clock.Advance(15 * time.Second)
	if err := <-result; !errors.Is(err, ErrCollectionLoadWaitTimeout) {
		t.Fatalf("Acquire error = %v, want ErrCollectionLoadWaitTimeout", err)
	}
	after := metrics.Read()
	if after.MilvusCollectionLoadWaitTimeoutsTotal-before.MilvusCollectionLoadWaitTimeoutsTotal != 1 {
		t.Fatalf("wait timeout delta = %d, want 1", after.MilvusCollectionLoadWaitTimeoutsTotal-before.MilvusCollectionLoadWaitTimeoutsTotal)
	}
	record, found := handler.record("semantic.collection_load_wait_timeout")
	if !found {
		t.Fatal("load wait timeout event was not emitted")
	}
	requireResidencyEventFields(t, record)
	requireResidencyEventValues(t, record, 0, 15000, 1, 0)
}

func TestResidencyCountsTimerLeaseRaceAsSkippedInUse(t *testing.T) {
	clock := newTestResidencyClock()
	unloaded := make(chan struct{}, 1)
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: time.Second,
		idleTimeout: time.Minute,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			return nil
		},
		unload: func(context.Context, string) error {
			unloaded <- struct{}{}
			return nil
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	firstLease, err := controller.Acquire(context.Background(), "live")
	if err != nil {
		t.Fatalf("first Acquire returned error: %v", err)
	}
	firstLease.Release()
	staleTimer := clock.LastTimer()
	activeLease, err := controller.Acquire(context.Background(), "live")
	if err != nil {
		t.Fatalf("second Acquire returned error: %v", err)
	}
	defer activeLease.Release()

	before := metrics.Read()
	staleTimer.FireStale()
	after := metrics.Read()
	if delta := after.MilvusCollectionUnloadSkippedInUseTotal -
		before.MilvusCollectionUnloadSkippedInUseTotal; delta != 1 {
		t.Fatalf("skipped unload delta = %d want 1", delta)
	}
	select {
	case <-unloaded:
		t.Fatal("timer and lease race unloaded an active collection")
	default:
	}
}

func TestResidencyDoesNotCountUnrelatedStaleTimerAsSkippedInUse(t *testing.T) {
	clock := newTestResidencyClock()
	unloaded := make(chan struct{}, 1)
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: time.Second,
		idleTimeout: time.Minute,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			return nil
		},
		unload: func(context.Context, string) error {
			unloaded <- struct{}{}
			return nil
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	firstLease, err := controller.Acquire(context.Background(), "live")
	if err != nil {
		t.Fatalf("first Acquire returned error: %v", err)
	}
	firstLease.Release()
	staleTimer := clock.LastTimer()
	secondLease, err := controller.Acquire(context.Background(), "live")
	if err != nil {
		t.Fatalf("second Acquire returned error: %v", err)
	}
	secondLease.Release()

	before := metrics.Read()
	staleTimer.FireStale()
	after := metrics.Read()
	if delta := after.MilvusCollectionUnloadSkippedInUseTotal -
		before.MilvusCollectionUnloadSkippedInUseTotal; delta != 0 {
		t.Fatalf("skipped unload delta = %d want 0", delta)
	}
	select {
	case <-unloaded:
		t.Fatal("unrelated stale timer unloaded a collection")
	default:
	}
}

func TestResidencyStateGaugesExcludeStagingCollections(t *testing.T) {
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       newTestResidencyClock(),
		waitTimeout: time.Second,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			return nil
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	liveLease, err := controller.Acquire(context.Background(), "live")
	if err != nil {
		t.Fatalf("acquire live: %v", err)
	}
	stagingLease, err := controller.Acquire(
		context.Background(),
		"live"+stagingCollectionSuffix,
	)
	if err != nil {
		t.Fatalf("acquire staging: %v", err)
	}
	liveLease.Release()
	stagingLease.Release()

	snapshot := metrics.Read()
	if snapshot.MilvusCollectionsReady != 1 {
		t.Fatalf("MilvusCollectionsReady = %d, want 1", snapshot.MilvusCollectionsReady)
	}
}

func TestResidencyReportsLoadAndUnloadFailures(t *testing.T) {
	handler := captureResidencyLogs(t)
	before := metrics.Read()
	loadController := newCollectionResidencyController(residencyControllerConfig{
		clock:       newTestResidencyClock(),
		waitTimeout: time.Second,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			return errors.New("load failed")
		},
	})
	_, loadErr := loadController.Acquire(context.Background(), "load-failure")
	if loadErr == nil {
		t.Fatal("Acquire returned nil error for failed load")
	}
	if err := loadController.Close(context.Background()); err != nil {
		t.Fatalf("close load controller: %v", err)
	}

	clock := newTestResidencyClock()
	unloadAttempted := make(chan struct{}, 1)
	unloadController := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: time.Second,
		idleTimeout: time.Minute,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			return nil
		},
		unload: func(context.Context, string) error {
			clock.Advance(500 * time.Millisecond)
			unloadAttempted <- struct{}{}
			return ErrCollectionNotReady
		},
	})
	t.Cleanup(func() {
		_ = unloadController.Close(context.Background())
	})
	lease, err := unloadController.Acquire(context.Background(), "unload-failure")
	if err != nil {
		t.Fatalf("Acquire returned error: %v", err)
	}
	lease.Release()
	clock.Advance(75 * time.Second)
	select {
	case <-unloadAttempted:
	case <-time.After(time.Second):
		t.Fatal("unload was not attempted")
	}
	deadline := time.Now().Add(time.Second)
	for {
		if _, found := handler.record("semantic.collection_unload_failed"); found {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("unload failure event was not emitted")
		}
		time.Sleep(time.Millisecond)
	}

	after := metrics.Read()
	if after.MilvusCollectionLoadFailuresTotal-before.MilvusCollectionLoadFailuresTotal != 1 {
		t.Fatalf("load failure delta = %d, want 1", after.MilvusCollectionLoadFailuresTotal-before.MilvusCollectionLoadFailuresTotal)
	}
	if after.MilvusCollectionUnloadFailuresTotal-before.MilvusCollectionUnloadFailuresTotal != 1 {
		t.Fatalf("unload failure delta = %d, want 1", after.MilvusCollectionUnloadFailuresTotal-before.MilvusCollectionUnloadFailuresTotal)
	}
	record, found := handler.record("semantic.collection_unload_failed")
	if !found {
		t.Fatal("unload failure event missing")
	}
	requireResidencyEventFields(t, record)
	requireResidencyEventValues(t, record, 0, 500, 0, 75000)
	errorClass := ""
	record.Attrs(func(attribute slog.Attr) bool {
		if attribute.Key == "error_class" {
			errorClass = attribute.Value.String()
		}
		return true
	})
	if errorClass != "collection_not_ready" {
		t.Fatalf("error_class = %q, want collection_not_ready", errorClass)
	}
}
