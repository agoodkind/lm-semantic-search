//go:build live

package live

import (
	"context"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

type mmapRollbackTarget struct {
	kind  string
	name  string
	value string
}

func TestMmapPropertyRollbackPreservesCollection(t *testing.T) {
	harness := newResidencyHarness(t, 0)
	_, collectionName := indexResidencyCodebase(
		t,
		harness,
		"mmap-rollback",
		"mmap rollback preservation sentinel",
	)

	rowsBefore := queryRowSnapshots(t, harness, collectionName, `id != ""`)
	jobsBefore := publicJobIDs(t, harness)
	embeddingsBefore := harness.embeddingRecorder.snapshot()
	targets := snapshotMmapTargets(t, harness, collectionName)
	requireAllMmapTargets(t, targets, "true")
	requireLoadState(t, harness, collectionName, entity.LoadStateLoaded)

	rollbackMmapProperties(t, harness, collectionName, targets)
	requireLoadState(t, harness, collectionName, entity.LoadStateNotLoad)
	requireAllMmapTargets(t, snapshotMmapTargets(t, harness, collectionName), "")

	reapplyMmapProperties(t, harness, collectionName, targets)
	loadCollectionDirect(t, harness, collectionName)
	requireAllMmapTargets(t, snapshotMmapTargets(t, harness, collectionName), "true")

	rowsAfter := queryRowSnapshots(t, harness, collectionName, `id != ""`)
	if !reflect.DeepEqual(rowsAfter, rowsBefore) {
		t.Fatalf("rows changed across mmap rollback\nbefore: %+v\nafter: %+v", rowsBefore, rowsAfter)
	}
	if jobsAfter := publicJobIDs(t, harness); !slices.Equal(jobsAfter, jobsBefore) {
		t.Fatalf("jobs changed across mmap rollback\nbefore: %v\nafter: %v", jobsBefore, jobsAfter)
	}
	if embeddingsAfter := harness.embeddingRecorder.snapshot(); !reflect.DeepEqual(embeddingsAfter, embeddingsBefore) {
		t.Fatalf(
			"embedding calls changed across mmap rollback\nbefore: %v\nafter: %v",
			embeddingsBefore,
			embeddingsAfter,
		)
	}
	vectorChecksums := make([]string, 0, len(rowsAfter))
	for _, row := range rowsAfter {
		vectorChecksums = append(vectorChecksums, row.vectorChecksum)
	}
	t.Logf(
		"mmap rollback targets=%d rows=%d dense_vector_sha256=%v jobs=%d embeddings_unchanged=true",
		len(targets),
		len(rowsAfter),
		vectorChecksums,
		len(jobsBefore),
	)
}

func snapshotMmapTargets(
	t *testing.T,
	harness *harness,
	collectionName string,
) []mmapRollbackTarget {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	return snapshotMmapTargetsForClient(t, ctx, harness.milvus, collectionName)
}

func snapshotMmapTargetsForClient(
	t *testing.T,
	ctx context.Context,
	client *milvusclient.Client,
	collectionName string,
) []mmapRollbackTarget {
	t.Helper()
	description, err := client.DescribeCollection(
		ctx,
		milvusclient.NewDescribeCollectionOption(collectionName),
	)
	if err != nil {
		t.Fatalf("describe collection %s for mmap targets: %v", collectionName, err)
	}
	targets := make([]mmapRollbackTarget, 0)
	for _, field := range description.Schema.Fields {
		supported := !field.PrimaryKey && field.Name != "id" && field.Name != "sparse_vector"
		if supported && field.Name != "vector" && field.DataType.IsVectorType() {
			supported = false
		}
		if supported {
			targets = append(targets, mmapRollbackTarget{
				kind:  "field",
				name:  field.Name,
				value: field.TypeParams["mmap.enabled"],
			})
		}
	}
	for _, fieldName := range []string{"vector", "contentHash", "sparse_vector"} {
		indexNames, listErr := client.ListIndexes(
			ctx,
			milvusclient.NewListIndexOption(collectionName).WithFieldName(fieldName),
		)
		if listErr != nil {
			t.Fatalf("list %s indexes for mmap targets: %v", fieldName, listErr)
		}
		for _, indexName := range indexNames {
			indexDescription, describeErr := client.DescribeIndex(
				ctx,
				milvusclient.NewDescribeIndexOption(collectionName, indexName),
			)
			if describeErr != nil {
				t.Fatalf("describe index %s for mmap targets: %v", indexName, describeErr)
			}
			params := indexDescription.Params()
			supported := fieldName == "vector" ||
				(fieldName == "contentHash" && params["index_type"] == "INVERTED") ||
				(fieldName == "sparse_vector" && params["index_type"] == "SPARSE_INVERTED_INDEX")
			if supported {
				targets = append(targets, mmapRollbackTarget{
					kind:  "index",
					name:  indexName,
					value: params["mmap.enabled"],
				})
			}
		}
	}
	slices.SortFunc(targets, func(left mmapRollbackTarget, right mmapRollbackTarget) int {
		if left.kind != right.kind {
			return strings.Compare(left.kind, right.kind)
		}
		return strings.Compare(left.name, right.name)
	})
	return targets
}

func requireAllMmapTargets(t *testing.T, targets []mmapRollbackTarget, want string) {
	t.Helper()
	if len(targets) == 0 {
		t.Fatal("mmap rollback found no supported targets")
	}
	for _, target := range targets {
		if target.value != want {
			t.Fatalf(
				"%s %s mmap.enabled = %q, want %q",
				target.kind,
				target.name,
				target.value,
				want,
			)
		}
	}
}

func rollbackMmapProperties(
	t *testing.T,
	harness *harness,
	collectionName string,
	targets []mmapRollbackTarget,
) {
	t.Helper()
	releaseCollectionDirect(t, harness, collectionName)
	for _, target := range targets {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		switch target.kind {
		case "field":
			status, err := harness.milvus.GetService().AlterCollectionField(
				ctx,
				&milvuspb.AlterCollectionFieldRequest{
					DbName:         harness.databaseName,
					CollectionName: collectionName,
					FieldName:      target.name,
					DeleteKeys:     []string{"mmap.enabled"},
				},
			)
			if err != nil {
				cancel()
				t.Fatalf("remove mmap property from field %s: %v", target.name, err)
			}
			if status.GetErrorCode() != commonpb.ErrorCode_Success {
				cancel()
				t.Fatalf("remove mmap property from field %s: %s", target.name, status.GetReason())
			}
		case "index":
			if err := harness.milvus.DropIndexProperties(
				ctx,
				milvusclient.NewDropIndexPropertiesOption(
					collectionName,
					target.name,
					"mmap.enabled",
				),
			); err != nil {
				cancel()
				t.Fatalf("remove mmap property from index %s: %v", target.name, err)
			}
		}
		cancel()
	}
}

func reapplyMmapProperties(
	t *testing.T,
	harness *harness,
	collectionName string,
	targets []mmapRollbackTarget,
) {
	t.Helper()
	for _, target := range targets {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		var err error
		switch target.kind {
		case "field":
			err = harness.milvus.AlterCollectionFieldProperty(
				ctx,
				milvusclient.NewAlterCollectionFieldPropertiesOption(
					collectionName,
					target.name,
				).WithProperty("mmap.enabled", "true"),
			)
		case "index":
			err = harness.milvus.AlterIndexProperties(
				ctx,
				milvusclient.NewAlterIndexPropertiesOption(
					collectionName,
					target.name,
				).WithProperty("mmap.enabled", "true"),
			)
		}
		cancel()
		if err != nil {
			t.Fatalf("restore mmap property on %s %s: %v", target.kind, target.name, err)
		}
	}
}
