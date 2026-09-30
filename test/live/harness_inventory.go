//go:build live

package live

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

func readMilvusInventory(client *milvusclient.Client) (milvusInventory, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	collectionNames, err := client.ListCollections(ctx, milvusclient.NewListCollectionOption())
	if err != nil {
		slog.Error("read Milvus collection inventory", "error", err)
		return nil, fmt.Errorf("list Milvus collections for inventory: %w", err)
	}
	slices.Sort(collectionNames)
	inventory := make(milvusInventory, len(collectionNames))
	for _, collectionName := range collectionNames {
		properties := make(map[string]string)
		loadState, loadErr := client.GetLoadState(
			ctx,
			milvusclient.NewGetLoadStateOption(collectionName),
		)
		if loadErr != nil {
			return nil, fmt.Errorf(
				"get Milvus load state for %s inventory: %w",
				collectionName,
				loadErr,
			)
		}
		properties["load_state"] = strconv.FormatInt(int64(loadState.State), 10)
		collection, describeErr := client.DescribeCollection(
			ctx,
			milvusclient.NewDescribeCollectionOption(collectionName),
		)
		if describeErr != nil {
			return nil, fmt.Errorf(
				"describe Milvus collection %s for inventory: %w",
				collectionName,
				describeErr,
			)
		}
		for key, value := range collection.Properties {
			properties["collection:"+key] = value
		}
		if collection.Schema != nil {
			for _, field := range collection.Schema.Fields {
				properties["field:"+field.Name] = field.TypeParams["mmap.enabled"]
			}
		}
		indexNames, listErr := client.ListIndexes(
			ctx,
			milvusclient.NewListIndexOption(collectionName),
		)
		if listErr != nil {
			return nil, fmt.Errorf(
				"list Milvus indexes for %s inventory: %w",
				collectionName,
				listErr,
			)
		}
		slices.Sort(indexNames)
		for _, indexName := range indexNames {
			description, indexErr := client.DescribeIndex(
				ctx,
				milvusclient.NewDescribeIndexOption(collectionName, indexName),
			)
			if indexErr != nil {
				return nil, fmt.Errorf(
					"describe Milvus index %s on %s: %w",
					indexName,
					collectionName,
					indexErr,
				)
			}
			properties["index:"+indexName] = description.Params()["mmap.enabled"]
		}
		inventory[collectionName] = properties
	}
	return inventory, nil
}

func listMilvusDatabases(client *milvusclient.Client) ([]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	databaseNames, err := client.ListDatabase(ctx, milvusclient.NewListDatabaseOption())
	if err != nil {
		slog.Error("read Milvus database inventory", "error", err)
		return nil, fmt.Errorf("list Milvus databases: %w", err)
	}
	slices.Sort(databaseNames)
	return databaseNames, nil
}

func dropCollectionIfPresent(
	client *milvusclient.Client,
	collectionName string,
) error {
	dropCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := client.DropCollection(
		dropCtx,
		milvusclient.NewDropCollectionOption(collectionName),
	); err != nil {
		if !strings.Contains(err.Error(), "not exist") && !strings.Contains(err.Error(), "not found") {
			slog.Error("drop test collection", "collection", collectionName, "error", err)
			return fmt.Errorf("DropCollection(%s) returned error: %w", collectionName, err)
		}
	}
	return nil
}

func dropEveryCollection(client *milvusclient.Client) []error {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	collectionNames, err := client.ListCollections(ctx, milvusclient.NewListCollectionOption())
	cancel()
	if err != nil {
		return []error{fmt.Errorf("list temporary Milvus collections for cleanup: %w", err)}
	}
	cleanupErrors := make([]error, 0)
	slices.Sort(collectionNames)
	for _, collectionName := range collectionNames {
		if err := dropCollectionIfPresent(client, collectionName); err != nil {
			cleanupErrors = append(cleanupErrors, err)
		}
	}
	return cleanupErrors
}
