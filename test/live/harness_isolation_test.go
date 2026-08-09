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
		"live_sandbox",
		"/milvus.proto.milvus.MilvusService/ReleaseCollection",
		&milvuspb.ReleaseCollectionRequest{CollectionName: "temporary_collection"},
	)
	recorder.observe(
		defaultMilvusDatabase,
		"/milvus.proto.milvus.MilvusService/CreateDatabase",
		&milvuspb.CreateDatabaseRequest{DbName: "live_sandbox"},
	)

	calls := recorder.snapshot()
	if len(calls) != 2 {
		t.Fatalf("recorded calls = %d, want 2", len(calls))
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
