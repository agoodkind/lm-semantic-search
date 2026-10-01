//go:build live

package live

import (
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
)

func TestLibraryCodebaseResidencyAfterRestart(t *testing.T) {
	for _, storage := range []config.CodebaseStoreKind{config.CodebaseStoreLibrary, config.CodebaseStoreSemantic} {
		t.Run(string(storage), func(t *testing.T) {
			t.Setenv("CLAUDE_CONTEXT_MILVUS_COLLECTION_IDLE_TIMEOUT_MS", "1000")
			codebaseDaemon := newCodebaseLiveDaemon(t, storage)
			root := t.TempDir()
			writeCodebaseFile(t, root, "recovery.go", goFile(goFunction("OldVersion", "oldversionmarker")))
			codebaseDaemon.index(t, root)
			logResidencySchemas(t, codebaseDaemon)
			collectionName := "lms_library_codebase"
			if storage == config.CodebaseStoreSemantic {
				collections, err := codebaseDaemon.harness.milvus.ListCollections(t.Context(), milvusclient.NewListCollectionOption())
				if err != nil {
					t.Fatalf("list legacy collections: %v", err)
				}
				collectionName = ""
				for _, name := range collections {
					if strings.HasPrefix(name, "code_chunks_") || strings.HasPrefix(name, "hybrid_code_chunks_") {
						if collectionName != "" {
							t.Fatal("the isolated legacy codebase has multiple collections")
						}
						collectionName = name
					}
				}
				if collectionName == "" {
					t.Fatal("public indexing did not create the legacy collection")
				}
			}
			var vectorsBefore []string
			if storage == config.CodebaseStoreLibrary {
				vectorsBefore = codebaseDaemon.harness.backendVectorIDs(collectionName)
			}
			requireResidencySearch(t, codebaseDaemon, root, "oldversionmarker")
			loaded, err := codebaseDaemon.harness.milvus.GetLoadState(t.Context(), milvusclient.NewGetLoadStateOption(collectionName))
			if err != nil || loaded.State != entity.LoadStateLoaded {
				t.Fatalf("pre-restart collection state=%v, error=%v; want Loaded", loaded, err)
			}
			codebaseDaemon.stop()
			child := startCodebaseRestartChild(t, codebaseDaemon)
			reopened, err := codebaseDaemon.harness.milvus.GetLoadState(t.Context(), milvusclient.NewGetLoadStateOption(collectionName))
			if err != nil || reopened.State != entity.LoadStateLoaded {
				t.Fatalf("reopened collection state=%v, error=%v; want Loaded", reopened, err)
			}
			time.Sleep(3 * time.Second)
			state, err := codebaseDaemon.harness.milvus.GetLoadState(t.Context(), milvusclient.NewGetLoadStateOption(collectionName))
			if err != nil {
				t.Fatalf("read post-idle load state: %v", err)
			}
			wantedState := entity.LoadStateLoaded
			if storage == config.CodebaseStoreSemantic {
				wantedState = entity.LoadStateNotLoad
			}
			if state.State != wantedState {
				t.Fatalf("post-restart idle state=%v, want %v", state.State, wantedState)
			}
			if storage == config.CodebaseStoreLibrary && !slices.Equal(vectorsBefore, codebaseDaemon.harness.backendVectorIDs(collectionName)) {
				t.Fatal("restart and idle observation changed canonical vectors")
			}
			requireResidencySearch(t, codebaseDaemon, root, "oldversionmarker")
			writeCodebaseFile(t, root, "recovery.go", goFile(goFunction("NewVersion", "newversionmarker")))
			codebaseDaemon.sync(t, root)
			requireResidencySearch(t, codebaseDaemon, root, "newversionmarker")
			if storage == config.CodebaseStoreSemantic {
				time.Sleep(3 * time.Second)
				cold, coldErr := codebaseDaemon.harness.milvus.GetLoadState(t.Context(), milvusclient.NewGetLoadStateOption(collectionName))
				if coldErr != nil || cold.State != entity.LoadStateNotLoad {
					t.Fatalf("pre-restart cold state=%v, error=%v; want NotLoad", cold, coldErr)
				}
				killCodebaseRestartChild(t, child)
				child = startCodebaseRestartChild(t, codebaseDaemon)
				cold, coldErr = codebaseDaemon.harness.milvus.GetLoadState(t.Context(), milvusclient.NewGetLoadStateOption(collectionName))
				if coldErr != nil || cold.State != entity.LoadStateNotLoad {
					t.Fatalf("startup warmed cold collection: state=%v error=%v", cold, coldErr)
				}
				requireConcurrentResidencySearch(t, codebaseDaemon, root, "newversionmarker")
			}
			killCodebaseRestartChild(t, child)
		})
	}
}

func requireConcurrentResidencySearch(t *testing.T, codebaseDaemon *libraryCodebaseDaemon, root, marker string) {
	t.Helper()
	type searchResult struct {
		response *pb.SearchCodeResponse
		err      error
	}
	results := make(chan searchResult, 2)
	start := make(chan struct{})
	for range 2 {
		go func() {
			<-start
			response, err := codebaseDaemon.client.SearchCode(codebaseDaemon.harness.context(), &pb.SearchCodeRequest{Path: root, Query: marker, Limit: codebaseSearchLimit})
			results <- searchResult{response: response, err: err}
		}()
	}
	close(start)
	for range 2 {
		result := <-results
		if result.err != nil {
			t.Fatalf("concurrent public search after cold restart: %v", result.err)
		}
		if len(result.response.GetResults()) != 1 || !strings.Contains(result.response.GetResults()[0].GetContent(), marker) {
			t.Fatal("concurrent public search did not return the committed source")
		}
	}
}

func logResidencySchemas(t *testing.T, codebaseDaemon *libraryCodebaseDaemon) {
	t.Helper()
	collections, err := codebaseDaemon.harness.milvus.ListCollections(t.Context(), milvusclient.NewListCollectionOption())
	if err != nil {
		t.Fatalf("list physical schema evidence: %v", err)
	}
	for _, name := range collections {
		collection, describeErr := codebaseDaemon.harness.milvus.DescribeCollection(t.Context(), milvusclient.NewDescribeCollectionOption(name))
		if describeErr != nil {
			t.Fatalf("describe physical schema evidence: %v", describeErr)
		}
		for _, field := range collection.Schema.Fields {
			t.Logf("database=%s collection=%s field=%s type=%s primary=%t", codebaseDaemon.harness.database, name, field.Name, field.DataType, field.PrimaryKey)
		}
	}
}

func requireResidencySearch(t *testing.T, codebaseDaemon *libraryCodebaseDaemon, root, marker string) {
	t.Helper()
	response, err := codebaseDaemon.client.SearchCode(codebaseDaemon.harness.context(), &pb.SearchCodeRequest{Path: root, Query: marker, Limit: codebaseSearchLimit})
	if err != nil {
		t.Fatalf("public search after idle: %v", err)
	}
	if len(response.GetResults()) != 1 || response.GetResults()[0].GetRelativePath() != "recovery.go" || !strings.Contains(response.GetResults()[0].GetContent(), marker) {
		t.Fatalf("public search after idle did not return the committed %s source", marker)
	}
}
