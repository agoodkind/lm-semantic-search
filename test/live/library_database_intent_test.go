//go:build live

package live

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestLibraryDatabaseIntentChild(t *testing.T) {
	if os.Getenv("LMS_LIBRARY_INTENT_NEGATIVE_CHILD") == "" {
		return
	}
	newLibraryHarness(t)
}

func TestLibraryDatabaseIntentRejectsUnsafeRegistration(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("registered oracle admission requires the Mac isolated /private/tmp testbed")
	}
	runID := fmt.Sprintf("oracle_intent_%d", time.Now().UnixNano())
	if registeredID := os.Getenv("LMS_LIBRARY_INTENT_NEGATIVE_RUN_ID"); registeredID != "" {
		if !validLibraryRunID(registeredID) || len(libraryLiveDatabasePrefix+registeredID) > 128 {
			t.Fatal("negative database registration run ID is invalid")
		}
		runID = registeredID
	}
	root := filepath.Join("/private/tmp", "lms-oracle-rerun-"+runID)
	if err := os.Mkdir(root, 0o700); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(root); err != nil {
			t.Error(err)
		}
	})
	intent := libraryDatabaseIntent{SchemaVersion: 1, Database: libraryLiveDatabasePrefix + runID,
		MilvusAddress: "localhost:39630", RunRoot: root, RunID: runID,
		RegisteredAt: time.Now().UTC(), AbsenceVerified: true}
	for _, name := range []string{"database", "endpoint", "permission", "scope", "contention"} {
		t.Run(name, func(t *testing.T) {
			candidate := intent
			mode := os.FileMode(0o600)
			pattern := "^TestLibrarySearchCompletePagesMatchTheExhaustiveOracle$"
			reason := ""
			switch name {
			case "database":
				candidate.Database = "default"
				reason = "database intent does not bind its exact isolated root and database"
			case "endpoint":
				candidate.MilvusAddress = "localhost:19530"
				reason = "database intent requires prior absence, timestamp and exact isolated native endpoint"
			case "permission":
				mode = 0o644
				reason = "database intent must be a private owned regular file of at most 4096 bytes"
			case "scope":
				pattern = "^TestLibraryDatabaseIntentChild$"
				reason = "database intent requires the exact existing exhaustive oracle test"
			case "contention":
				reason = "claim exclusive database intent:"
			}
			data, err := json.Marshal(candidate)
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(root, "database-intent.json")
			if err = os.WriteFile(path, data, 0o600); err != nil {
				t.Fatal(err)
			}
			if err = os.Chmod(path, mode); err != nil {
				t.Fatal(err)
			}
			var registration *os.File
			if name == "contention" {
				_, registration, err = readLibraryDatabaseIntent("TestLibrarySearchCompletePagesMatchTheExhaustiveOracle", path, intent.MilvusAddress)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := registration.Close(); err != nil {
						t.Error(err)
					}
				})
			}
			ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
			defer cancel()
			executable, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			command := exec.CommandContext(ctx, executable, "-test.run="+pattern, "-test.count=1", "-test.v")
			command.Env = append(os.Environ(),
				"HOME="+root, "CLAUDE_CONTEXTD_CONFIG_ROOT="+root,
				"CLAUDE_CONTEXTD_STATE_ROOT="+root, "CLAUDE_CONTEXTD_CONTEXT_ROOT="+root,
				"MILVUS_ADDRESS=localhost:39630", "OPENAI_BASE_URL=http://[::1]:5400/v1",
				"EMBEDDING_MODEL=nvidia/NV-EmbedCode-7b-v1",
				"LMS_LIBRARY_LIVE_DATABASE_INTENT="+path, "LMS_LIBRARY_INTENT_NEGATIVE_CHILD=1")
			output, commandErr := command.CombinedOutput()
			if commandErr == nil || ctx.Err() != nil || !strings.Contains(string(output), "reject library database intent: "+reason) || strings.Contains(string(output), "created Milvus database") {
				t.Fatalf("unsafe registration did not fail before database creation: err=%v context=%v output=%s", commandErr, ctx.Err(), output)
			}
			if registration != nil {
				if err := registration.Close(); err != nil {
					t.Fatal(err)
				}
				_, registration, err = readLibraryDatabaseIntent("TestLibrarySearchCompletePagesMatchTheExhaustiveOracle", path, intent.MilvusAddress)
				if err != nil {
					t.Fatalf("reclaim database intent after close: %v", err)
				}
			}
		})
	}
}
