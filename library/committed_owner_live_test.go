//go:build live

package library_test

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/embedding"
	"goodkind.io/lm-semantic-search/library/milvus"
	"goodkind.io/lm-semantic-search/library/observation"
)

//go:embed committed_owner_checkpoint.sql
var committedReaderCheckpoint string

type committedReadEvents struct {
	mutex  sync.Mutex
	events []observation.Event
	output *os.File
	err    error
}

func (events *committedReadEvents) Observe(event observation.Event) {
	events.mutex.Lock()
	defer events.mutex.Unlock()
	events.events = append(events.events, event)
	if events.err == nil {
		events.err = json.NewEncoder(events.output).Encode(event)
		if events.err == nil {
			events.err = events.output.Sync()
		}
	}
}

func newCommittedReadEvents(t *testing.T) *committedReadEvents {
	t.Helper()
	root := os.Getenv("LMS_COMMITTED_READER_EVIDENCE_ROOT")
	if root == "" {
		var err error
		root, err = os.MkdirTemp("", "lms-committed-reader-evidence-")
		if err != nil {
			t.Fatalf("create retained observation directory: %v", err)
		}
	}
	if !filepath.IsAbs(root) {
		t.Fatal("committed reader evidence root must be absolute")
	}
	path := filepath.Join(root, "operation-events.jsonl")
	output, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		t.Fatalf("create retained observation file: %v", err)
	}
	events := &committedReadEvents{output: output}
	t.Logf("retained actual operation observations %s", path)
	t.Cleanup(func() {
		events.mutex.Lock()
		defer events.mutex.Unlock()
		if err := errors.Join(events.err, output.Close()); err != nil {
			t.Errorf("persist actual operation observations: %v", err)
		}
	})
	return events
}

func committedReaderDatabase(t *testing.T) string {
	t.Helper()
	database := os.Getenv("LMS_COMMITTED_READER_DATABASE")
	if database == "" {
		return "lms_committed_reader_" + hex.EncodeToString(randomCommittedReaderBytes(t))
	}
	const prefix = "clyde_frozen_fixture_"
	if !strings.HasPrefix(database, prefix) || len(database) == len(prefix) {
		t.Fatal("supplied committed reader database requires clyde_frozen_fixture_ followed by digits")
	}
	for _, digit := range strings.TrimPrefix(database, prefix) {
		if digit < '0' || digit > '9' {
			t.Fatal("supplied committed reader database suffix must contain only ASCII digits")
		}
	}
	return database
}

func committedReadContext(ctx context.Context) context.Context {
	return observation.WithScope(ctx, observation.Scope{RunID: "committed-reader", Generation: 0, ProcessID: os.Getpid(), OperationID: 0, ParentOperationID: 0, Purpose: observation.Recovery})
}

func openCommittedReaderFixture(t *testing.T) (*library.Library, context.Context, *committedReadEvents, string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 180*time.Second)
	t.Cleanup(cancel)
	settings, err := config.Default()
	if err != nil {
		t.Fatalf("load live settings: %v", err)
	}
	endpoint, err := url.Parse(settings.OpenAIBaseURL)
	if err != nil || endpoint.Port() != "5400" || !slices.Contains([]string{"localhost", "127.0.0.1", "::1"}, endpoint.Hostname()) {
		t.Fatal("committed reader fixture requires the isolated local embedding endpoint on port 5400")
	}
	if settings.MilvusAddress != "localhost:39530" && settings.MilvusAddress != "127.0.0.1:39530" {
		t.Fatal("committed reader fixture requires isolated Milvus on localhost:39530")
	}
	if settings.EmbeddingModel != "nvidia/NV-EmbedCode-7b-v1" {
		t.Fatal("committed reader fixture requires the verified local NV-EmbedCode-7b-v1 model")
	}
	admin, err := milvusclient.New(ctx, &milvusclient.ClientConfig{Address: settings.MilvusAddress})
	if err != nil {
		t.Fatalf("connect isolated Milvus: %v", err)
	}
	t.Cleanup(func() {
		cleanup, stop := context.WithTimeout(context.Background(), 30*time.Second)
		defer stop()
		if err := admin.Close(cleanup); err != nil {
			t.Errorf("close isolated admin: %v", err)
		}
	})
	database := committedReaderDatabase(t)
	existing, err := admin.ListDatabase(ctx, milvusclient.NewListDatabaseOption())
	if err != nil || slices.Contains(existing, database) {
		t.Fatalf("verify isolated database prior absence: present=%t error=%v", slices.Contains(existing, database), err)
	}
	t.Cleanup(func() { deleteCommittedReaderDatabase(t, admin, database) })
	if err := admin.CreateDatabase(ctx, milvusclient.NewCreateDatabaseOption(database)); err != nil {
		t.Fatalf("create isolated database: %v", err)
	}
	t.Logf("created isolated database %s", database)
	client, err := milvusclient.New(ctx, &milvusclient.ClientConfig{Address: settings.MilvusAddress, DBName: database})
	if err != nil {
		t.Fatalf("connect isolated database: %v", err)
	}
	t.Cleanup(func() {
		cleanup, stop := context.WithTimeout(context.Background(), 30*time.Second)
		defer stop()
		if err := client.DropCollection(cleanup, milvusclient.NewDropCollectionOption("committed_reader")); err != nil {
			t.Errorf("drop isolated collection: %v", err)
		}
		if err := client.Close(cleanup); err != nil {
			t.Errorf("close isolated client: %v", err)
		}
	})
	events := newCommittedReadEvents(t)
	vectors, err := milvus.New(client, milvus.Config{Observer: events, Database: database, Collection: "committed_reader"})
	if err != nil {
		t.Fatalf("create actual vector adapter: %v", err)
	}
	embedder, err := embedding.NewOpenAI(ctx, embedding.OpenAIConfig{Observer: events, BaseURL: settings.OpenAIBaseURL, APIKey: settings.OpenAIAPIKey, Model: settings.EmbeddingModel, Dimension: 4096, RequestTimeout: 30 * time.Second, MaxAttempts: 1, BackoffBase: 0})
	if err != nil {
		t.Fatalf("create actual embedding adapter: %v", err)
	}
	root := t.TempDir()
	opened, err := library.Open(ctx, library.Config{Observer: events, Store: library.StoreDescriptor{CatalogPath: filepath.Join(root, "catalog.sqlite"), LockPath: filepath.Join(root, "catalog.lock"), PoolID: "committed_reader", EmbeddingModel: settings.EmbeddingModel, EmbeddingRevision: "2a97ba03aee57d4c2b146fbd74ee84b3a219ddfac13e5d4ea32f373638d2d97f", Dimension: 4096, Normalization: "unit-l2"}, Vectors: vectors, Embedder: embedder})
	if err != nil {
		t.Fatalf("open actual library: %v", err)
	}
	t.Cleanup(func() {
		if err := opened.Close(); err != nil {
			t.Errorf("close library: %v", err)
		}
	})
	return opened, ctx, events, filepath.Join(root, "catalog.sqlite")
}

func randomCommittedReaderBytes(t *testing.T) []byte {
	t.Helper()
	value := make([]byte, 12)
	if _, err := rand.Read(value); err != nil {
		t.Fatalf("create unique database identity: %v", err)
	}
	return value
}

func deleteCommittedReaderDatabase(t *testing.T, admin *milvusclient.Client, database string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	existing, err := admin.ListDatabase(ctx, milvusclient.NewListDatabaseOption())
	if err != nil {
		t.Errorf("verify isolated database before cleanup: %v", err)
		return
	}
	if !slices.Contains(existing, database) {
		t.Logf("verified absent isolated database %s", database)
		return
	}
	if err := admin.DropDatabase(ctx, milvusclient.NewDropDatabaseOption(database)); err != nil {
		t.Errorf("delete isolated database %s: %v", database, err)
		return
	}
	remaining, err := admin.ListDatabase(ctx, milvusclient.NewListDatabaseOption())
	if err != nil || slices.Contains(remaining, database) {
		t.Errorf("verify deletion of %s: present=%t error=%v", database, slices.Contains(remaining, database), err)
		return
	}
	t.Logf("verified deleted isolated database %s", database)
}

func committedReaderSpec() library.NamespaceSpec {
	return library.NamespaceSpec{ID: "committed", Policy: library.AppendOnly, Scalars: []library.ScalarColumn{
		{Name: "role", Type: library.String, Nullable: false, Mutable: false, MaxLength: 32},
		{Name: "workspace", Type: library.String, Nullable: true, Mutable: true, MaxLength: 128},
		{Name: "archived", Type: library.Bool, Nullable: false, Mutable: true, MaxLength: 0},
		{Name: "timestamp", Type: library.Int64, Nullable: false, Mutable: false, MaxLength: 0},
	}}
}

func committedReaderRow(key string) library.Occurrence {
	sortKeys := map[string]string{"a": "z", "b": "y", "c": "x", "d": "w"}
	return library.Occurrence{RowKey: key, SortKey: sortKeys[key], SourceText: "Original source " + key + "\x00 retained", SearchText: "lexical " + key, EmbeddingInput: "Document input " + key, Scalars: map[string]library.ScalarValue{
		"role":      {Type: library.String, String: "assistant"},
		"workspace": {Type: library.String, String: "accepted"},
		"archived":  {Type: library.Bool, Bool: false},
		"timestamp": {Type: library.Int64, Int64: 0},
	}}
}

func stageCommittedReaderRows(ctx context.Context, opened *library.Library, order uint64, rows []library.Occurrence) (library.GenerationKey, library.GenerationSeal, error) {
	key := library.GenerationKey{Namespace: "committed", OwnerID: "owner", GenerationOrder: order, IdempotencyToken: fmt.Sprintf("generation-%d", order)}
	seal, err := library.SealRows(rows)
	if err != nil {
		slog.WarnContext(ctx, "seal live reader rows failed", "err", err)
		return key, seal, fmt.Errorf("seal live reader rows: %w", err)
	}
	if err := opened.Stage(ctx, library.StageBatch{Key: key, Mode: library.Append, Rows: rows}); err != nil {
		slog.WarnContext(ctx, "stage live reader rows failed", "err", err)
		return key, seal, fmt.Errorf("stage live reader rows: %w", err)
	}
	return key, seal, nil
}

func projectCommittedReaderWorkspace(ctx context.Context, opened *library.Library, order uint64, value library.ScalarValue) error {
	_, err := opened.ReprojectScalars(ctx, library.ScalarProjection{Namespace: "committed", OwnerID: "owner", ProjectionOrder: order, IdempotencyToken: fmt.Sprintf("projection-%d", order), Rows: map[string]map[string]library.ScalarValue{"a": {"workspace": value, "archived": {Type: library.Bool, Bool: true}}}})
	if err != nil {
		slog.WarnContext(ctx, "project live reader metadata failed", "err", err)
		return fmt.Errorf("project live reader metadata: %w", err)
	}
	return nil
}

func TestReadCommittedOwnerLiveSnapshot(t *testing.T) {
	opened, ctx, events, catalogPath := openCommittedReaderFixture(t)
	if err := opened.RegisterNamespace(ctx, committedReaderSpec()); err != nil {
		t.Fatalf("register namespace: %v", err)
	}
	initial := []library.Occurrence{committedReaderRow("b"), committedReaderRow("a")}
	key, seal, err := stageCommittedReaderRows(ctx, opened, 1, initial)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := opened.CommitGeneration(ctx, key, seal); err != nil {
		t.Fatalf("commit initial generation: %v", err)
	}
	if err := projectCommittedReaderWorkspace(ctx, opened, 1, library.ScalarValue{Type: library.String, Null: true}); err != nil {
		t.Fatal(err)
	}
	staged := []library.Occurrence{committedReaderRow("c")}
	key, seal, err = stageCommittedReaderRows(ctx, opened, 2, staged)
	if err != nil {
		t.Fatal(err)
	}
	panicKey, panicSeal, err := stageCommittedReaderRows(ctx, opened, 3, []library.Occurrence{committedReaderRow("d")})
	if err != nil {
		t.Fatal(err)
	}
	state, err := opened.GetOwnerState(ctx, "committed", "owner")
	if err != nil {
		t.Fatal(err)
	}
	events.mutex.Lock()
	sdkBoundary := len(events.events)
	events.mutex.Unlock()
	var visited []library.CommittedOccurrence
	snapshot, err := opened.ReadCommittedOwner(committedReadContext(ctx), "committed", "owner", func(row library.CommittedOccurrence) error {
		visited = append(visited, row)
		if len(visited) != 1 {
			return nil
		}
		return commitDuringOwnerRead(ctx, opened, key, seal)
	})
	if err != nil || snapshot.State != state || snapshot.ProjectionOrder != 1 || snapshot.RowCount != 2 {
		t.Fatalf("original snapshot = %+v, error=%v", snapshot, err)
	}
	assertCommittedReaderRows(t, visited, []library.Occurrence{initial[1], initial[0]}, []uint64{1, 1}, library.ScalarValue{Type: library.String, Null: true})
	visited = nil
	snapshot, err = opened.ReadCommittedOwner(committedReadContext(ctx), "committed", "owner", func(row library.CommittedOccurrence) error { visited = append(visited, row); return nil })
	if err != nil || snapshot.State.GenerationOrder != 2 || snapshot.ProjectionOrder != 2 || snapshot.RowCount != 3 {
		t.Fatalf("later snapshot = %+v, error=%v", snapshot, err)
	}
	assertCommittedReaderRows(t, visited, []library.Occurrence{initial[1], initial[0], staged[0]}, []uint64{1, 1, 2}, library.ScalarValue{Type: library.String, String: "later"})
	assertCommittedReaderFailures(t, ctx, opened)
	assertCommittedReaderPanicCleanup(t, ctx, opened, catalogPath, panicKey, panicSeal)
	assertCommittedReaderNoSDK(t, events, sdkBoundary)
}

func commitDuringOwnerRead(ctx context.Context, opened *library.Library, key library.GenerationKey, seal library.GenerationSeal) error {
	done := make(chan error, 1)
	go func() {
		if _, err := opened.CommitGeneration(ctx, key, seal); err != nil {
			slog.WarnContext(ctx, "concurrent reader fixture commit failed", "err", err)
			done <- fmt.Errorf("concurrent reader fixture commit: %w", err)
			return
		}
		done <- projectCommittedReaderWorkspace(ctx, opened, 2, library.ScalarValue{Type: library.String, String: "later"})
	}()
	return <-done
}

func committedReaderHash(value string) string {
	digest := sha256.Sum256([]byte(value))
	return hex.EncodeToString(digest[:])
}

func assertCommittedReaderPanicCleanup(t *testing.T, ctx context.Context, opened *library.Library, catalogPath string, key library.GenerationKey, seal library.GenerationSeal) {
	t.Helper()
	const panicValue = "committed callback panic"
	func() {
		defer func() {
			if recovered := recover(); recovered != panicValue {
				t.Fatalf("callback panic = %v, want original panic", recovered)
			}
		}()
		_, err := opened.ReadCommittedOwner(committedReadContext(ctx), "committed", "owner", func(library.CommittedOccurrence) error {
			if err := commitDuringOwnerRead(ctx, opened, key, seal); err != nil {
				return err
			}
			panic(panicValue)
		})
		t.Fatalf("callback panic did not propagate: %v", err)
	}()
	checkpoint, err := sql.Open("sqlite3", "file:"+catalogPath+"?mode=rw&_busy_timeout=1000")
	if err != nil {
		t.Fatalf("open actual SQLite checkpoint connection: %v", err)
	}
	defer func() {
		if err := checkpoint.Close(); err != nil {
			t.Errorf("close checkpoint connection: %v", err)
		}
	}()
	var busy, frames, copied int
	if err := checkpoint.QueryRowContext(ctx, committedReaderCheckpoint).Scan(&busy, &frames, &copied); err != nil {
		t.Fatalf("checkpoint after callback panic: %v", err)
	}
	if busy != 0 || frames != 0 || copied != 0 {
		t.Fatalf("callback retained a read transaction: checkpoint busy=%d frames=%d copied=%d", busy, frames, copied)
	}
	snapshot, err := opened.ReadCommittedOwner(committedReadContext(ctx), "committed", "owner", func(library.CommittedOccurrence) error { return nil })
	if err != nil || snapshot.State.GenerationOrder != 3 || snapshot.RowCount != 4 {
		t.Fatalf("post-panic committed snapshot = %+v, %v", snapshot, err)
	}
}

func assertCommittedReaderRows(t *testing.T, got []library.CommittedOccurrence, rows []library.Occurrence, generations []uint64, workspace library.ScalarValue) {
	t.Helper()
	if len(got) != len(rows) {
		t.Fatalf("visited %d rows, want %d", len(got), len(rows))
	}
	for index, original := range rows {
		scalars := make(map[string]library.ScalarValue, len(original.Scalars))
		for name, value := range original.Scalars {
			scalars[name] = value
		}
		if original.RowKey == "a" {
			scalars["workspace"] = workspace
			scalars["archived"] = library.ScalarValue{Type: library.Bool, Bool: true}
		}
		want := library.CommittedOccurrence{ID: library.OccurrenceID{Namespace: "committed", OwnerID: "owner", RowKey: original.RowKey}, GenerationOrder: generations[index], SortKey: original.SortKey, SourceSHA256: committedReaderHash(original.SourceText), SearchSHA256: committedReaderHash(original.SearchText), EmbeddingInputSHA256: committedReaderHash(original.EmbeddingInput), Scalars: scalars}
		if !reflect.DeepEqual(got[index], want) {
			t.Fatalf("committed identity/scalars differ at row %s", original.RowKey)
		}
	}
}

func assertCommittedReaderFailures(t *testing.T, ctx context.Context, opened *library.Library) {
	t.Helper()
	ctx = committedReadContext(ctx)
	callbackError := errors.New("caller stopped iteration")
	got, err := opened.ReadCommittedOwner(ctx, "committed", "owner", func(library.CommittedOccurrence) error { return callbackError })
	if !errors.Is(err, callbackError) || got != (library.CommittedOwnerSnapshot{}) {
		t.Fatalf("callback failure returned %+v, %v", got, err)
	}
	cancelled, cancel := context.WithCancel(ctx)
	defer cancel()
	count := 0
	got, err = opened.ReadCommittedOwner(cancelled, "committed", "owner", func(library.CommittedOccurrence) error { count++; cancel(); return nil })
	if !errors.Is(err, context.Canceled) || got != (library.CommittedOwnerSnapshot{}) || count != 1 {
		t.Fatalf("canceled callback returned %+v, %v, count=%d", got, err, count)
	}
	got, err = opened.ReadCommittedOwner(ctx, "committed", "missing", func(library.CommittedOccurrence) error { t.Error("unknown owner visited a row"); return nil })
	if err != nil || got != (library.CommittedOwnerSnapshot{}) {
		t.Fatalf("unknown owner returned %+v, %v", got, err)
	}
	for _, request := range []struct {
		namespace, owner string
		visit            func(library.CommittedOccurrence) error
	}{
		{namespace: "", owner: "owner", visit: func(library.CommittedOccurrence) error { return nil }},
		{namespace: "missing", owner: "owner", visit: func(library.CommittedOccurrence) error { return nil }},
		{namespace: "committed", owner: "", visit: func(library.CommittedOccurrence) error { return nil }},
		{namespace: "committed", owner: "owner", visit: nil},
	} {
		got, err := opened.ReadCommittedOwner(ctx, request.namespace, request.owner, request.visit)
		if !errors.Is(err, library.ErrInvalidRequest) || got != (library.CommittedOwnerSnapshot{}) {
			t.Fatalf("invalid reader request returned %+v, %v", got, err)
		}
	}
}

func assertCommittedReaderNoSDK(t *testing.T, events *committedReadEvents, boundary int) {
	t.Helper()
	events.mutex.Lock()
	defer events.mutex.Unlock()
	if events.err != nil {
		t.Fatalf("persist actual reader observations: %v", events.err)
	}
	embedded, written := false, false
	for index, event := range events.events {
		if event.Scope.RunID == "committed-reader" && (event.Operation == observation.WriterAdmission || event.Operation == observation.CatalogTransaction || event.Operation == observation.StrongVerification) {
			t.Fatalf("committed reader performed %s", event.Operation)
		}
		if event.Operation != observation.EmbeddingAttempt && event.Operation != observation.UpsertCall {
			continue
		}
		if index >= boundary {
			t.Fatalf("committed reader performed %s", event.Operation)
		}
		if event.Phase == observation.Completed && event.Outcome == observation.Success {
			embedded = embedded || event.Operation == observation.EmbeddingAttempt
			written = written || event.Operation == observation.UpsertCall
		}
	}
	if !embedded || !written {
		t.Fatal("actual fixture SDK observations are missing")
	}
}
