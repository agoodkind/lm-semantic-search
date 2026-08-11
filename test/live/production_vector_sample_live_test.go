//go:build live && production

package live

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/semantic/milvusgrpc"
)

type vectorSampleRow struct {
	PrimaryKey     string `json:"primary_key"`
	VectorChecksum string `json:"vector_checksum"`
}

type collectionVectorSample struct {
	Collection   string            `json:"collection"`
	PrimaryField string            `json:"primary_field"`
	Rows         []vectorSampleRow `json:"rows"`
}

func TestProductionVectorSamples(t *testing.T) {
	requireProductionOptIn(t)
	cfg, err := config.Default()
	if err != nil {
		t.Fatalf("resolve production configuration: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	client, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address:     cfg.MilvusAddress,
		APIKey:      cfg.MilvusToken,
		DBName:      productionDatabaseName,
		DialOptions: milvusgrpc.DialOptions(ctx, slog.Default(), milvusgrpc.DefaultCallTimeouts()),
	})
	if err != nil {
		t.Fatalf("connect production Milvus for vector sampling: %v", err)
	}
	t.Cleanup(func() { _ = client.Close(context.Background()) })

	expectedPath := requiredProductionEnvironment(t, "LMS_VECTOR_EXPECT")
	expectedBytes, readErr := os.ReadFile(expectedPath)
	if readErr != nil {
		t.Fatalf("read expected vector samples: %v", readErr)
	}
	var targets []collectionVectorSample
	if err := json.Unmarshal(expectedBytes, &targets); err != nil {
		t.Fatalf("decode expected vector samples: %v", err)
	}
	if len(targets) == 0 {
		t.Fatal("expected vector samples are empty")
	}

	actual := make([]collectionVectorSample, 0, len(targets))
	for targetIndex := range targets {
		target := &targets[targetIndex]
		if target.Rows == nil {
			target.Rows = make([]vectorSampleRow, 0)
		}
		slices.SortFunc(target.Rows, func(left vectorSampleRow, right vectorSampleRow) int {
			return strings.Compare(left.PrimaryKey, right.PrimaryKey)
		})
		state, stateErr := client.GetLoadState(
			ctx,
			milvusclient.NewGetLoadStateOption(target.Collection),
		)
		if stateErr != nil {
			t.Fatalf("get vector sample load state for %s: %v", target.Collection, stateErr)
		}
		if state.State == entity.LoadStateNotLoad {
			if _, err := client.LoadCollection(
				ctx,
				milvusclient.NewLoadCollectionOption(target.Collection),
			); err != nil {
				t.Fatalf("load vector sample collection %s: %v", target.Collection, err)
			}
			waitForClientLoadState(t, ctx, client, target.Collection, entity.LoadStateLoaded)
			defer func(collectionName string) {
				if err := client.ReleaseCollection(
					context.Background(),
					milvusclient.NewReleaseCollectionOption(collectionName),
				); err != nil {
					t.Fatalf("restore vector sample collection %s: %v", collectionName, err)
				}
				waitForClientLoadState(
					t,
					context.Background(),
					client,
					collectionName,
					entity.LoadStateNotLoad,
				)
			}(target.Collection)
		}
		filter := target.PrimaryField + ` != ""`
		limit := 16
		if len(target.Rows) > 0 {
			quotedKeys := make([]string, 0, len(target.Rows))
			for _, row := range target.Rows {
				encodedKey, marshalErr := json.Marshal(row.PrimaryKey)
				if marshalErr != nil {
					t.Fatalf("encode vector sample key: %v", marshalErr)
				}
				quotedKeys = append(quotedKeys, string(encodedKey))
			}
			filter = fmt.Sprintf("%s in [%s]", target.PrimaryField, strings.Join(quotedKeys, ","))
			limit = len(target.Rows)
		}
		result, queryErr := client.Query(
			ctx,
			milvusclient.NewQueryOption(target.Collection).
				WithFilter(filter).
				WithLimit(limit).
				WithOutputFields(target.PrimaryField, "vector").
				WithConsistencyLevel(entity.ClStrong),
		)
		if queryErr != nil {
			t.Fatalf("query vector samples from %s: %v", target.Collection, queryErr)
		}
		keyColumn := result.GetColumn(target.PrimaryField)
		vectorColumn := result.GetColumn("vector")
		if keyColumn == nil || vectorColumn == nil {
			t.Fatalf("vector sample query from %s omitted required columns", target.Collection)
		}
		rows := make([]vectorSampleRow, 0, result.ResultCount)
		for rowIndex := range result.ResultCount {
			key, keyErr := keyColumn.GetAsString(rowIndex)
			if keyErr != nil {
				t.Fatalf("read vector sample key from %s: %v", target.Collection, keyErr)
			}
			value, vectorErr := vectorColumn.Get(rowIndex)
			if vectorErr != nil {
				t.Fatalf("read vector sample from %s: %v", target.Collection, vectorErr)
			}
			vector, ok := value.(entity.FloatVector)
			if !ok {
				t.Fatalf("vector sample from %s has type %T", target.Collection, value)
			}
			rows = append(rows, vectorSampleRow{
				PrimaryKey:     key,
				VectorChecksum: checksumVector(vector),
			})
		}
		slices.SortFunc(rows, func(left vectorSampleRow, right vectorSampleRow) int {
			return strings.Compare(left.PrimaryKey, right.PrimaryKey)
		})
		actual = append(actual, collectionVectorSample{
			Collection:   target.Collection,
			PrimaryField: target.PrimaryField,
			Rows:         rows,
		})
	}
	encoded, err := json.Marshal(actual)
	if err != nil {
		t.Fatalf("encode production vector samples: %v", err)
	}
	if outputPath := os.Getenv("LMS_VECTOR_OUTPUT"); outputPath != "" {
		if err := os.WriteFile(outputPath, append(encoded, '\n'), 0o600); err != nil {
			t.Fatalf("write production vector samples: %v", err)
		}
	}
	t.Logf("PRODUCTION_VECTOR_SAMPLES_JSON=%s", encoded)
	if !reflect.DeepEqual(actual, targets) {
		t.Fatalf("production vector samples changed\nexpected: %+v\nactual: %+v", targets, actual)
	}
}
