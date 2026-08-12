//go:build restartacceptance

package restartacceptance

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/semantic/milvusgrpc"
	"google.golang.org/protobuf/proto"
)

const (
	productionDatabaseConfirmation = "LMS_PRODUCTION_CONFIRM_DATABASE"
	productionCensusSampleLimit    = 64
)

type productionMilvusSettings struct {
	Address  string
	Token    string
	Database string
}

type collectionFingerprint struct {
	ShardCount       int32              `json:"shard_count"`
	Consistency      int32              `json:"consistency"`
	Properties       map[string]string  `json:"properties"`
	PhysicalChannels []string           `json:"physical_channels"`
	VirtualChannels  []string           `json:"virtual_channels"`
	Indexes          []indexFingerprint `json:"indexes"`
	RowCount         int64              `json:"row_count"`
}

type rowFingerprint struct {
	Identity     string            `json:"identity"`
	VectorHashes map[string]string `json:"vector_hashes"`
}

type indexFingerprint struct {
	Name       string            `json:"name"`
	Parameters map[string]string `json:"parameters"`
}

type sampleFingerprint struct {
	RowCount   int64            `json:"row_count"`
	RowSamples []rowFingerprint `json:"row_samples"`
}

func configuredProductionMilvusSettings() (productionMilvusSettings, error) {
	database := os.Getenv(productionDatabaseConfirmation)
	configuration, err := config.Default()
	if err != nil {
		return productionMilvusSettings{}, fmt.Errorf("resolve production LMS configuration: %w", err)
	}
	if configuration.MilvusDatabase != "" && configuration.MilvusDatabase != database {
		return productionMilvusSettings{}, fmt.Errorf(
			"configured production Milvus database %q does not match confirmation %q",
			configuration.MilvusDatabase,
			database,
		)
	}
	settings := productionMilvusSettings{
		Address:  configuration.MilvusAddress,
		Token:    configuration.MilvusToken,
		Database: database,
	}
	if err := validateProductionMilvusSettings(settings); err != nil {
		return productionMilvusSettings{}, err
	}
	return settings, nil
}

func validateProductionMilvusSettings(settings productionMilvusSettings) error {
	if settings.Address == "" {
		return fmt.Errorf("production Milvus address is empty")
	}
	if settings.Database != "default" {
		return fmt.Errorf("%s must equal default", productionDatabaseConfirmation)
	}
	return nil
}

func configuredProductionMilvusCensus(ctx context.Context) (productionMilvusCensus, error) {
	settings, err := configuredProductionMilvusSettings()
	if err != nil {
		return productionMilvusCensus{}, err
	}
	return readProductionMilvusCensus(ctx, settings)
}

func readProductionMilvusCensus(ctx context.Context, settings productionMilvusSettings) (productionMilvusCensus, error) {
	if err := validateProductionMilvusSettings(settings); err != nil {
		return productionMilvusCensus{}, err
	}
	client, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address:     settings.Address,
		APIKey:      settings.Token,
		DBName:      settings.Database,
		DialOptions: milvusgrpc.DialOptions(ctx, slog.Default(), milvusgrpc.DefaultCallTimeouts()),
	})
	if err != nil {
		return productionMilvusCensus{}, fmt.Errorf("connect production Milvus for read-only census: %w", err)
	}
	defer func() {
		closeContext, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = client.Close(closeContext)
	}()

	databases, err := client.ListDatabase(ctx, milvusclient.NewListDatabaseOption())
	if err != nil {
		return productionMilvusCensus{}, fmt.Errorf("list production Milvus databases: %w", err)
	}
	slices.Sort(databases)
	collectionNames, err := client.ListCollections(ctx, milvusclient.NewListCollectionOption())
	if err != nil {
		return productionMilvusCensus{}, fmt.Errorf("list production Milvus collections: %w", err)
	}
	slices.Sort(collectionNames)
	collections := make(collectionCensus, len(collectionNames))
	samples := make(collectionCensus)
	for _, collectionName := range collectionNames {
		hash, sampleHash, hashErr := readCollectionFingerprints(ctx, client, collectionName)
		if hashErr != nil {
			return productionMilvusCensus{}, hashErr
		}
		identity := collectionIdentity{Database: settings.Database, Collection: collectionName}
		collections[identity] = hash
		if sampleHash != "" {
			samples[identity] = sampleHash
		}
	}
	return productionMilvusCensus{
		Databases:   slices.Clone(databases),
		Collections: collections,
		Samples:     samples,
	}, nil
}

func readCollectionFingerprints(ctx context.Context, client *milvusclient.Client, collectionName string) (string, string, error) {
	loadState, err := client.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(collectionName))
	if err != nil {
		return "", "", fmt.Errorf("get production collection %q load state: %w", collectionName, err)
	}
	collection, err := client.DescribeCollection(ctx, milvusclient.NewDescribeCollectionOption(collectionName))
	if err != nil {
		return "", "", fmt.Errorf("describe production collection %q: %w", collectionName, err)
	}
	if collection.Schema == nil {
		return "", "", fmt.Errorf("production collection %q has no schema", collectionName)
	}
	stats, err := client.GetCollectionStats(ctx, milvusclient.NewGetCollectionStatsOption(collectionName))
	if err != nil {
		return "", "", fmt.Errorf("get production collection %q statistics: %w", collectionName, err)
	}
	rowCount, err := strconv.ParseInt(stats["row_count"], 10, 64)
	if err != nil {
		return "", "", fmt.Errorf("parse production collection %q row count: %w", collectionName, err)
	}
	strongRowCount := rowCount
	if loadState.State == entity.LoadStateLoaded {
		strongRowCount, err = readStrongCollectionRowCount(ctx, client, collectionName)
		if err != nil {
			return "", "", err
		}
	}
	rowSamples, err := readLoadedCollectionSamples(
		ctx,
		client,
		collectionName,
		collection.Schema,
		loadState.State,
		strongRowCount,
	)
	if err != nil {
		return "", "", err
	}
	schemaBody, err := proto.MarshalOptions{Deterministic: true}.Marshal(collection.Schema.ProtoMessage())
	if err != nil {
		return "", "", fmt.Errorf("encode production collection %q schema: %w", collectionName, err)
	}
	indexNames, err := client.ListIndexes(ctx, milvusclient.NewListIndexOption(collectionName))
	if err != nil {
		return "", "", fmt.Errorf("list production collection %q indexes: %w", collectionName, err)
	}
	slices.Sort(indexNames)
	indexes := make([]indexFingerprint, 0, len(indexNames))
	for _, indexName := range indexNames {
		description, describeErr := client.DescribeIndex(
			ctx,
			milvusclient.NewDescribeIndexOption(collectionName, indexName),
		)
		if describeErr != nil {
			return "", "", fmt.Errorf("describe production index %q on %q: %w", indexName, collectionName, describeErr)
		}
		indexes = append(indexes, indexFingerprint{
			Name:       indexName,
			Parameters: description.Params(),
		})
	}
	fingerprint := collectionFingerprint{
		ShardCount:       collection.ShardNum,
		Consistency:      int32(collection.ConsistencyLevel),
		Properties:       collection.Properties,
		PhysicalChannels: slices.Clone(collection.PhysicalChannels),
		VirtualChannels:  slices.Clone(collection.VirtualChannels),
		Indexes:          indexes,
		RowCount:         rowCount,
	}
	slices.Sort(fingerprint.PhysicalChannels)
	slices.Sort(fingerprint.VirtualChannels)
	fingerprintBody, err := json.Marshal(fingerprint)
	if err != nil {
		return "", "", fmt.Errorf("encode production collection %q properties: %w", collectionName, err)
	}
	digest := sha256.New()
	_, _ = digest.Write(schemaBody)
	_, _ = digest.Write(fingerprintBody)
	durableHash := hex.EncodeToString(digest.Sum(nil))
	if loadState.State != entity.LoadStateLoaded {
		return durableHash, "", nil
	}
	sampleBody, err := json.Marshal(sampleFingerprint{RowCount: strongRowCount, RowSamples: rowSamples})
	if err != nil {
		return "", "", fmt.Errorf("encode production collection %q samples: %w", collectionName, err)
	}
	sampleDigest := sha256.Sum256(sampleBody)
	return durableHash, hex.EncodeToString(sampleDigest[:]), nil
}

func readStrongCollectionRowCount(
	ctx context.Context,
	client *milvusclient.Client,
	collectionName string,
) (int64, error) {
	result, err := client.Query(
		ctx,
		milvusclient.NewQueryOption(collectionName).
			WithOutputFields("count(*)").
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		return 0, fmt.Errorf("count production collection %q rows: %w", collectionName, err)
	}
	countColumn := result.GetColumn("count(*)")
	if countColumn == nil {
		return 0, fmt.Errorf("count production collection %q omitted count column", collectionName)
	}
	rowCount, err := countColumn.GetAsInt64(0)
	if err != nil {
		return 0, fmt.Errorf("read production collection %q row count: %w", collectionName, err)
	}
	return rowCount, nil
}

func readLoadedCollectionSamples(
	ctx context.Context,
	client *milvusclient.Client,
	collectionName string,
	schema *entity.Schema,
	loadState entity.LoadStateCode,
	rowCount int64,
) ([]rowFingerprint, error) {
	if loadState != entity.LoadStateLoaded || rowCount == 0 {
		return nil, nil
	}
	primaryField := schema.PKField()
	if primaryField == nil {
		return nil, nil
	}
	vectorFields := make([]string, 0)
	for _, field := range schema.Fields {
		if field.DataType.IsVectorType() {
			vectorFields = append(vectorFields, field.Name)
		}
	}
	if len(vectorFields) == 0 {
		return nil, nil
	}
	slices.Sort(vectorFields)
	outputFields := append([]string{primaryField.Name}, vectorFields...)
	iterator, err := client.QueryIterator(
		ctx,
		milvusclient.NewQueryIteratorOption(collectionName).
			WithBatchSize(productionCensusSampleLimit).
			WithIteratorLimit(productionCensusSampleLimit).
			WithOutputFields(outputFields...).
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		return nil, fmt.Errorf("open production collection %q row sample: %w", collectionName, err)
	}
	result, err := iterator.Next(ctx)
	if err != nil {
		return nil, fmt.Errorf("query production collection %q row sample: %w", collectionName, err)
	}
	identityColumn := result.GetColumn(primaryField.Name)
	if identityColumn == nil {
		return nil, fmt.Errorf("query production collection %q omitted primary field %q", collectionName, primaryField.Name)
	}
	rows := make([]rowFingerprint, 0, result.ResultCount)
	for rowIndex := range result.ResultCount {
		identity, identityErr := identityColumn.Get(rowIndex)
		if identityErr != nil {
			return nil, fmt.Errorf("read production collection %q row identity: %w", collectionName, identityErr)
		}
		identityBody, marshalErr := json.Marshal(identity)
		if marshalErr != nil {
			return nil, fmt.Errorf("encode production collection %q row identity: %w", collectionName, marshalErr)
		}
		vectorHashes := make(map[string]string, len(vectorFields))
		for _, vectorField := range vectorFields {
			vectorColumn := result.GetColumn(vectorField)
			if vectorColumn == nil {
				return nil, fmt.Errorf("query production collection %q omitted vector field %q", collectionName, vectorField)
			}
			vectorBody, vectorErr := proto.MarshalOptions{Deterministic: true}.Marshal(
				vectorColumn.Slice(rowIndex, rowIndex+1).FieldData(),
			)
			if vectorErr != nil {
				return nil, fmt.Errorf("encode production collection %q vector field %q: %w", collectionName, vectorField, vectorErr)
			}
			vectorDigest := sha256.Sum256(vectorBody)
			vectorHashes[vectorField] = hex.EncodeToString(vectorDigest[:])
		}
		rows = append(rows, rowFingerprint{Identity: string(identityBody), VectorHashes: vectorHashes})
	}
	slices.SortFunc(rows, func(left rowFingerprint, right rowFingerprint) int {
		return strings.Compare(left.Identity, right.Identity)
	})
	return rows, nil
}
