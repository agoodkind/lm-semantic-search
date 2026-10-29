//go:build offlinelive

package offlinelive

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/collection"
	"goodkind.io/lm-semantic-search/internal/localvec"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/store"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	documentCodebaseID       = "cb_1780000000_d0c0c0de0001"
	documentCollectionID     = "clyde-conversations"
	documentCanonicalPath    = "chat:///" + documentCollectionID
	documentItemIDColumn     = "conversationId"
	documentItemIDMaxLength  = 128
	documentConversationID   = "conv-guard-001"
	documentSeedRowCount     = 6
	documentQueryLimit       = 10_000
	documentStagingSuffix    = ".staging"
	documentLocalVectorDir   = "localvec"
	documentSeedEmbeddingTag = "offline-live-seed"
)

var documentDeclaration = collection.Declaration{
	ItemIDColumn: documentItemIDColumn,
	Scalars: []collection.ScalarColumn{
		{Name: documentItemIDColumn, Type: collection.ScalarTypeString, Nullable: false, MaxLength: documentItemIDMaxLength},
	},
}

// TestForceStartIndexLeavesDocumentCollectionRows checks that the offline daemon
// refuses StartIndex with Force true for a document codebase ID and leaves
// the 6 rows and their IDs unchanged in conv_chunks_09cfca5e.
func TestForceStartIndexLeavesDocumentCollectionRows(t *testing.T) {
	harness := newUnstartedHarness(t)
	options := harnessOptions{}
	cfg := resolveOfflineConfig(t, harness.stateRoot, harness.socketPath, options)
	prepareState(t, cfg)

	localVectorRoot := filepath.Join(cfg.StateRoot, documentLocalVectorDir)
	collectionName := collection.DocumentName(documentCollectionID)
	seedDocumentCollection(t, localVectorRoot, collectionName, int(cfg.EmbeddingDimension))
	registry := model.RegistryFile{Codebases: []model.Codebase{{
		ID:                documentCodebaseID,
		Kind:              model.CodebaseKindDocument,
		CanonicalPath:     documentCanonicalPath,
		Status:            model.CodebaseStatusIndexed,
		CollectionName:    collectionName,
		Declaration:       &documentDeclaration,
		LiveChunkTotal:    documentSeedRowCount,
		LastSuccessfulRun: &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: documentSeedRowCount, Status: string(model.JobStateCompleted)},
		UpdatedAt:         time.Now().Add(-time.Hour),
	}}}
	if err := store.WriteRegistry(cfg.RegistryPath, registry); err != nil {
		t.Fatalf("write document registry: %v", err)
	}

	beforeIDs := readDocumentRowIDs(t, localVectorRoot, collectionName)
	if len(beforeIDs) != documentSeedRowCount {
		t.Fatalf("seeded rows = %d, want %d", len(beforeIDs), documentSeedRowCount)
	}
	t.Logf("before StartIndex: collection %s has %d rows: %v", collectionName, len(beforeIDs), beforeIDs)

	harness.start(options, false)
	t.Cleanup(harness.teardown)

	response, startErr := harness.client.StartIndex(correlatedContext(), &pb.StartIndexRequest{
		Path:   documentCodebaseID,
		Force:  true,
		Client: harnessClientInfo(),
	})
	refused := startErr != nil
	if refused {
		t.Logf("StartIndex refused: code=%s message=%q", status.Code(startErr), status.Convert(startErr).Message())
	} else {
		t.Logf("StartIndex accepted: job=%s state=%s deduplicated=%v canonical=%s", response.GetJobId(), response.GetState(), response.GetDeduplicated(), response.GetCanonicalPath())
		if response.GetJobId() != "" {
			job := harness.waitForJob(response.GetJobId())
			jobError := ""
			if job.GetError() != nil {
				jobError = job.GetError().GetMessage()
			}
			t.Logf("job %s finished: state=%s error=%q", response.GetJobId(), job.GetState(), jobError)
		}
	}

	afterIDs, afterErr := tryReadDocumentRowIDs(localVectorRoot, collectionName)
	stagingPath := filepath.Join(localVectorRoot, collectionName+documentStagingSuffix)
	_, stagingStatErr := os.Stat(stagingPath)
	stagingPresent := stagingStatErr == nil
	if afterErr != nil {
		t.Logf("after StartIndex: read collection %s failed: %v; staging present=%v", collectionName, afterErr, stagingPresent)
	} else {
		t.Logf("after StartIndex: collection %s has %d rows: %v; staging present=%v", collectionName, len(afterIDs), afterIDs, stagingPresent)
	}

	if !refused {
		t.Errorf("StartIndex(%s, force) accepted a document codebase, want a refusal", documentCodebaseID)
	} else if status.Code(startErr) != codes.InvalidArgument {
		t.Errorf("StartIndex refusal code = %s, want %s", status.Code(startErr), codes.InvalidArgument)
	}
	if afterErr != nil {
		t.Fatalf("read document rows after StartIndex: %v", afterErr)
	}
	if !slices.Equal(beforeIDs, afterIDs) {
		t.Fatalf("document rows changed: before %d %v, after %d %v", len(beforeIDs), beforeIDs, len(afterIDs), afterIDs)
	}
}

func seedDocumentCollection(t *testing.T, root string, collectionName string, dimension int) {
	t.Helper()

	if dimension <= 0 {
		t.Fatalf("offline embedding dimension = %d, want a positive width", dimension)
	}
	generic, err := localvec.OpenGeneric(root, documentSeedEmbeddingTag)
	if err != nil {
		t.Fatalf("open local document store: %v", err)
	}
	defer generic.Close()
	ctx := context.Background()
	ensure := collection.EnsureRequest{Collection: collectionName, Declaration: documentDeclaration, Dimension: dimension}
	if err := generic.EnsureCollection(ctx, ensure); err != nil {
		t.Fatalf("ensure document collection: %v", err)
	}
	rows := make([]collection.Row, 0, documentSeedRowCount)
	for rowIndex := range documentSeedRowCount {
		vector := make([]float32, dimension)
		vector[rowIndex%dimension] = 1
		rows = append(rows, collection.Row{
			ID:            fmt.Sprintf("chunk_document_%02d", rowIndex),
			Content:       fmt.Sprintf("conversation message %d", rowIndex),
			RelativePath:  fmt.Sprintf("conv/%s/%02d", documentConversationID, rowIndex),
			StartLine:     int32(rowIndex + 1),
			EndLine:       int32(rowIndex + 1),
			FileExtension: ".md",
			Metadata:      "{}",
			Vector:        vector,
			Scalars:       map[string]collection.ScalarValue{documentItemIDColumn: collection.StringScalar(documentConversationID)},
		})
	}
	if err := generic.Upsert(ctx, collectionName, documentDeclaration, rows); err != nil {
		t.Fatalf("upsert document rows: %v", err)
	}
}

func readDocumentRowIDs(t *testing.T, root string, collectionName string) []string {
	t.Helper()

	identifiers, err := tryReadDocumentRowIDs(root, collectionName)
	if err != nil {
		t.Fatalf("read document rows: %v", err)
	}
	return identifiers
}

// tryReadDocumentRowIDs opens a new store on root and reads the collection
// files from disk instead of the daemon's loaded copy.
func tryReadDocumentRowIDs(root string, collectionName string) ([]string, error) {
	generic, err := localvec.OpenGeneric(root, documentSeedEmbeddingTag)
	if err != nil {
		return nil, fmt.Errorf("open local document store: %w", err)
	}
	defer generic.Close()
	hits, err := generic.Query(context.Background(), collection.QueryRequest{
		Collection: collectionName, Declaration: documentDeclaration, Filter: nil, Limit: documentQueryLimit,
	})
	if err != nil {
		if errors.Is(err, collection.ErrCollectionMissing) {
			return nil, fmt.Errorf("collection %s is missing: %w", collectionName, err)
		}
		return nil, fmt.Errorf("query collection %s: %w", collectionName, err)
	}
	identifiers := make([]string, 0, len(hits))
	for _, hit := range hits {
		identifiers = append(identifiers, hit.ID)
	}
	return identifiers, nil
}
