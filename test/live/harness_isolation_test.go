//go:build live

package live

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"goodkind.io/lm-semantic-search/internal/config"
)

func TestResolveLiveConfigMakesEveryLaterDefaultUseSandboxDatabase(t *testing.T) {
	const harnessID = "child_config"
	databaseName := liveDatabasePrefix + harnessID
	stateRoot := t.TempDir()
	t.Setenv("MILVUS_DATABASE", "")

	resolveLiveConfig(
		t,
		stateRoot,
		filepath.Join(stateRoot, "daemon.sock"),
		"http://127.0.0.1:1",
		"127.0.0.1:1",
		"",
		databaseName,
		harnessID,
		0,
		false,
	)

	for childIndex := range 3 {
		childConfig, err := config.Default()
		if err != nil {
			t.Fatalf("child config %d: %v", childIndex, err)
		}
		if childConfig.MilvusDatabase != databaseName {
			t.Fatalf(
				"child config %d MilvusDatabase = %q, want %q",
				childIndex,
				childConfig.MilvusDatabase,
				databaseName,
			)
		}
	}
}

func TestMilvusCallRecorderExtractsDatabaseTargets(t *testing.T) {
	recorder := &milvusCallRecorder{}
	recorder.observe(
		"/milvus.proto.milvus.MilvusService/ReleaseCollection",
		"live_sandbox",
		&milvuspb.ReleaseCollectionRequest{CollectionName: "temporary_collection"},
	)
	recorder.observe(
		"/milvus.proto.milvus.MilvusService/CreateDatabase",
		"live_sandbox",
		&milvuspb.CreateDatabaseRequest{DbName: "live_sandbox"},
	)
	recorder.observe(
		"/milvus.proto.milvus.MilvusService/RenameCollection",
		"live_sandbox",
		&milvuspb.RenameCollectionRequest{
			OldName:   "temporary_collection",
			NewName:   "temporary_collection_next",
			NewDBName: defaultMilvusDatabase,
		},
	)

	calls := recorder.snapshot()
	if len(calls) != 3 {
		t.Fatalf("recorded calls = %d, want 3", len(calls))
	}
	if calls[0].databaseName != "live_sandbox" ||
		calls[0].method != "ReleaseCollection" ||
		len(calls[0].collectionNames) != 1 ||
		calls[0].collectionNames[0] != "temporary_collection" {
		t.Fatalf("collection call = %+v, want selected database and collection", calls[0])
	}
	if calls[1].databaseName != "live_sandbox" || calls[1].method != "CreateDatabase" {
		t.Fatalf("database call = %+v, want request database target", calls[1])
	}
	if calls[2].destinationDatabaseName != defaultMilvusDatabase {
		t.Fatalf(
			"rename destination database = %q, want %q",
			calls[2].destinationDatabaseName,
			defaultMilvusDatabase,
		)
	}
}

func TestOperatorStateAuditAllowsConcurrentUntrackedAddition(t *testing.T) {
	beforeDatabases := []string{"default"}
	beforeInventory := milvusInventory{
		"operator_collection": {
			"field:dense_vector": "true",
			"load_state":         "3",
		},
	}
	afterInventory := milvusInventory{
		"operator_collection": {
			"field:dense_vector": "true",
			"load_state":         "3",
		},
		"concurrent_operator_collection": {
			"field:dense_vector": "true",
			"load_state":         "2",
		},
	}
	audit := auditOperatorState(
		"live_sandbox",
		beforeDatabases,
		[]string{"default"},
		beforeInventory,
		afterInventory,
		map[string]struct{}{"temporary_collection": {}},
		[]milvusCall{{
			databaseName:    "live_sandbox",
			method:          "Insert",
			collectionNames: []string{"temporary_collection"},
		}},
	)
	if len(audit.violations) != 0 {
		t.Fatalf("concurrent addition violations = %v, want none", audit.violations)
	}
	if len(audit.concurrentAdditions) != 1 ||
		audit.concurrentAdditions[0] != "concurrent_operator_collection" {
		t.Fatalf(
			"concurrent additions = %v, want concurrent_operator_collection",
			audit.concurrentAdditions,
		)
	}
}

func TestOperatorStateAuditRejectsEveryProtectedDifference(t *testing.T) {
	baseline := milvusInventory{
		"operator_collection": {"load_state": "3", "field:vector": "true"},
	}
	tests := []struct {
		name           string
		afterDatabases []string
		afterInventory milvusInventory
		temporaryNames map[string]struct{}
		calls          []milvusCall
		want           string
	}{
		{
			name:           "database list changed",
			afterDatabases: []string{"default", "unexpected"},
			afterInventory: baseline,
			want:           "Milvus database inventory changed",
		},
		{
			name:           "baseline collection removed",
			afterDatabases: []string{"default"},
			afterInventory: milvusInventory{},
			want:           `baseline collection "operator_collection" was removed`,
		},
		{
			name:           "baseline collection changed",
			afterDatabases: []string{"default"},
			afterInventory: milvusInventory{
				"operator_collection": {"load_state": "2", "field:vector": "true"},
			},
			want: `baseline collection "operator_collection" changed`,
		},
		{
			name:           "tracked temporary collection added",
			afterDatabases: []string{"default"},
			afterInventory: milvusInventory{
				"operator_collection":  {"load_state": "3", "field:vector": "true"},
				"temporary_collection": {"load_state": "2", "field:vector": "true"},
			},
			temporaryNames: map[string]struct{}{"temporary_collection": {}},
			want:           `tracked temporary collection "temporary_collection" appeared`,
		},
		{
			name:           "default database mutation recorded",
			afterDatabases: []string{"default"},
			afterInventory: baseline,
			calls: []milvusCall{{
				databaseName:    defaultMilvusDatabase,
				method:          "Insert",
				collectionNames: []string{"operator_collection"},
			}},
			want: `protected Milvus call Insert targeted database "default"`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			audit := auditOperatorState(
				"live_sandbox",
				[]string{"default"},
				test.afterDatabases,
				baseline,
				test.afterInventory,
				test.temporaryNames,
				test.calls,
			)
			joined := strings.Join(audit.violations, "\n")
			if !strings.Contains(joined, test.want) {
				t.Fatalf("violations = %v, want %q", audit.violations, test.want)
			}
		})
	}
}

func TestOperatorStateAuditTreatsAbsentAndDestinationDatabaseAsDefault(t *testing.T) {
	baseline := milvusInventory{"operator_collection": {"load_state": "3"}}
	audit := auditOperatorState(
		"live_sandbox",
		[]string{"default"},
		[]string{"default"},
		baseline,
		baseline,
		nil,
		[]milvusCall{
			{
				databaseName:    "",
				method:          "ReleaseCollection",
				collectionNames: []string{"operator_collection"},
			},
			{
				databaseName:            "live_sandbox",
				destinationDatabaseName: defaultMilvusDatabase,
				method:                  "RenameCollection",
				collectionNames:         []string{"temporary_collection", "operator_collection"},
			},
		},
	)
	if len(audit.violations) != 2 {
		t.Fatalf("default mutation violations = %v, want 2", audit.violations)
	}
}

func TestOperatorStateAuditRejectsAdditionWithHarnessMutationEvidence(t *testing.T) {
	baseline := milvusInventory{"operator_collection": {"load_state": "3"}}
	after := milvusInventory{
		"operator_collection":     {"load_state": "3"},
		"new_operator_collection": {"load_state": "2"},
	}
	audit := auditOperatorState(
		"live_sandbox",
		[]string{"default"},
		[]string{"default"},
		baseline,
		after,
		nil,
		[]milvusCall{{
			databaseName:    defaultMilvusDatabase,
			method:          "CreateCollection",
			collectionNames: []string{"new_operator_collection"},
		}},
	)
	if len(audit.concurrentAdditions) != 0 {
		t.Fatalf("concurrent additions = %v, want none", audit.concurrentAdditions)
	}
	joined := strings.Join(audit.violations, "\n")
	if !strings.Contains(joined, `untracked operator collection "new_operator_collection" appeared with harness mutation evidence`) {
		t.Fatalf("violations = %v, want mutation-attributed addition", audit.violations)
	}
}

func TestMilvusIsolationRejectsEveryDefaultDatabaseMutation(t *testing.T) {
	protectedMethods := []string{
		"AlterCollection",
		"CreateAlias",
		"CreateCollection",
		"CreateIndex",
		"CreatePartition",
		"Delete",
		"DropAlias",
		"DropCollection",
		"DropIndex",
		"DropPartition",
		"Flush",
		"FlushAll",
		"Import",
		"Insert",
		"LoadCollection",
		"ReleaseCollection",
		"ReplicateMessage",
		"RenameCollection",
		"TruncateCollection",
		"Upsert",
	}
	for _, method := range protectedMethods {
		t.Run(method, func(t *testing.T) {
			violations := milvusIsolationViolations(
				"live_sandbox",
				map[string]struct{}{"temporary_collection": {}},
				[]milvusCall{{
					databaseName:    defaultMilvusDatabase,
					method:          method,
					collectionNames: []string{"operator_collection"},
				}},
			)
			if len(violations) == 0 {
				t.Fatalf("default database %s produced no isolation violation", method)
			}
		})
	}

	for _, method := range []string{"CreateDatabase", "DropDatabase"} {
		t.Run(method, func(t *testing.T) {
			violations := milvusIsolationViolations(
				"live_sandbox",
				nil,
				[]milvusCall{{databaseName: defaultMilvusDatabase, method: method}},
			)
			if len(violations) == 0 {
				t.Fatalf("default database %s produced no isolation violation", method)
			}
		})
	}
}

func TestMilvusIsolationRejectsAbsentDatabaseAndRenameDestination(t *testing.T) {
	temporaryNames := map[string]struct{}{
		"temporary_collection":      {},
		"temporary_collection_next": {},
	}
	violations := milvusIsolationViolations(
		"live_sandbox",
		temporaryNames,
		[]milvusCall{
			{
				databaseName:    "",
				method:          "ReleaseCollection",
				collectionNames: []string{"temporary_collection"},
			},
			{
				databaseName:            "live_sandbox",
				destinationDatabaseName: defaultMilvusDatabase,
				method:                  "RenameCollection",
				collectionNames: []string{
					"temporary_collection",
					"temporary_collection_next",
				},
			},
		},
	)
	joined := strings.Join(violations, "\n")
	for _, expected := range []string{
		"targeted database \"\"",
		"destination database \"default\"",
	} {
		if !strings.Contains(joined, expected) {
			t.Fatalf("violations = %v, want %q", violations, expected)
		}
	}
}
