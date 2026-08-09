//go:build live

package live

import (
	"strings"
	"testing"

	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
)

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

func TestOperatorMilvusStateRequiresExactBeforeAfterEquality(t *testing.T) {
	beforeDatabases := []string{"default"}
	beforeInventory := milvusInventory{
		"operator_collection": {
			"field:dense_vector": "true",
			"load_state":         "3",
		},
	}
	equalInventory := milvusInventory{
		"operator_collection": {
			"field:dense_vector": "true",
			"load_state":         "3",
		},
	}
	if violations := operatorStateViolations(
		beforeDatabases,
		[]string{"default"},
		beforeInventory,
		equalInventory,
	); len(violations) != 0 {
		t.Fatalf("equal operator state violations = %v, want none", violations)
	}

	changedInventory := milvusInventory{
		"operator_collection": {
			"field:dense_vector": "true",
			"load_state":         "1",
		},
	}
	violations := operatorStateViolations(
		beforeDatabases,
		[]string{"default", "unexpected"},
		beforeInventory,
		changedInventory,
	)
	joined := strings.Join(violations, "\n")
	for _, expected := range []string{
		"Milvus database inventory changed",
		"operator Milvus inventory changed",
	} {
		if !strings.Contains(joined, expected) {
			t.Fatalf("violations = %v, want %q", violations, expected)
		}
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
