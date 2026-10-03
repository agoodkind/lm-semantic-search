//go:build live

package live

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

const (
	cliBinary              = "lm-semantic-search"
	heldSyncTimeout        = 30 * time.Second
	statusCommandTimeout   = 30 * time.Second
	missingNarrativeString = "semantic collection is missing"
)

// embedHold passes embed requests through the fake embedder's gate. While hold
// is set, each request waits until resume releases every held request.
type embedHold struct {
	gate       *embedGate
	holding    atomic.Bool
	heldCount  atomic.Int32
	resume     chan struct{}
	resumeOnce sync.Once
}

func newEmbedHold() *embedHold {
	return &embedHold{
		gate:   &embedGate{arrived: make(chan int, 64), release: make(chan struct{})},
		resume: make(chan struct{}),
	}
}

func (hold *embedHold) serve() {
	for range hold.gate.arrived {
		if hold.holding.Load() {
			hold.heldCount.Add(1)
			<-hold.resume
		}
		hold.gate.release <- struct{}{}
	}
}

func (hold *embedHold) release() {
	hold.holding.Store(false)
	hold.resumeOnce.Do(func() { close(hold.resume) })
}

// TestCodebaseStatusPrintsLiteralValuesWhileSyncRebuildsDroppedCollection runs
// the built daemon and CLI. It indexes a directory, drops that directory's
// collection in Milvus, then holds the rebuild sync inside the embedder. The
// status command must print the stored status, the collection probe, and the
// running sync as literal values instead of the old "collection is missing"
// narrative.
func TestCodebaseStatusPrintsLiteralValuesWhileSyncRebuildsDroppedCollection(t *testing.T) {
	hold := newEmbedHold()
	go hold.serve()

	harness, socketPath := newSandboxWatcherHarness(t, hold.gate)
	// Cleanups run last-in first-out, so registering the release after the
	// harness frees a held embed before the fake embedder waits for it to close.
	t.Cleanup(hold.release)
	root := t.TempDir()
	sourcePath := filepath.Join(root, "main.go")
	if err := os.WriteFile(sourcePath, []byte("package statusfixture\n\nfunc Existing() string { return \"indexed\" }\n"), 0o600); err != nil {
		t.Fatalf("write indexed source: %v", err)
	}

	initial := startPublicCodebaseIndex(t, harness, root)
	requirePublicCompleted(t, initial, "initial codebase index")
	snapshot := captureWatcherIndexSnapshot(t, harness, root)

	dropContext, cancelDrop := context.WithTimeout(harness.milvusContext, 15*time.Second)
	dropErr := harness.milvus.DropCollection(dropContext, milvusclient.NewDropCollectionOption(snapshot.collectionName))
	cancelDrop()
	if dropErr != nil {
		t.Fatalf("drop collection %s: %v", snapshot.collectionName, dropErr)
	}

	hold.holding.Store(true)
	if err := os.WriteFile(sourcePath, []byte("package statusfixture\n\nfunc Existing() string { return \"changed\" }\n"), 0o600); err != nil {
		t.Fatalf("write changed source: %v", err)
	}
	syncResponse, err := harness.client.SyncIndex(correlatedContext(), &pb.SyncIndexRequest{
		Path:   root,
		Client: &pb.ClientInfo{Name: "status-literal-live-harness"},
	})
	if err != nil {
		t.Fatalf("start sync: %v", err)
	}
	jobID := syncResponse.GetJobId()
	waitForHeldEmbed(t, hold, harness, jobID)

	output := runCodebaseStatus(t, socketPath, root)
	for _, want := range []string{
		"collection=absent",
		"job_id=" + jobID + " operation=sync state=running",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("codebase status lacks %q:\n%s", want, output)
		}
	}
	if strings.Contains(output, missingNarrativeString) {
		t.Fatalf("codebase status printed the mapped missing narrative:\n%s", output)
	}

	hold.release()
	requirePublicCompleted(t, waitForCodebasePublicJob(t, harness, jobID), "rebuild sync")
}

// waitForHeldEmbed waits until the sync job is running with one embed request
// held, so the status read observes a live job.
func waitForHeldEmbed(t *testing.T, hold *embedHold, harness *harness, jobID string) {
	t.Helper()
	deadline := time.Now().Add(heldSyncTimeout)
	for time.Now().Before(deadline) {
		response, err := harness.client.GetJob(correlatedContext(), &pb.GetJobRequest{JobId: jobID})
		if err != nil {
			t.Fatalf("get sync job %s: %v", jobID, err)
		}
		job := response.GetJob()
		if publicJobTerminal(job) {
			t.Fatalf("sync job %s ended before an embed was held: state=%s error=%s", jobID, job.GetState(), job.GetDisplayError())
		}
		if job.GetState() == "running" && hold.heldCount.Load() > 0 {
			return
		}
		time.Sleep(jobPollInterval)
	}
	t.Fatalf("sync job %s did not reach a held embed within %s", jobID, heldSyncTimeout)
}

// runCodebaseStatus runs the built CLI against the sandbox socket and returns
// its combined output.
func runCodebaseStatus(t *testing.T, socketPath string, root string) string {
	t.Helper()
	commandContext, cancel := context.WithTimeout(context.Background(), statusCommandTimeout)
	defer cancel()
	command := exec.CommandContext(commandContext, builtCLIPath(t), "--socket", socketPath, "codebase", "status", root)
	output := &bytes.Buffer{}
	command.Stdout = output
	command.Stderr = output
	if err := command.Run(); err != nil {
		t.Fatalf("run codebase status: %v\n%s", err, output.String())
	}
	return output.String()
}

func builtCLIPath(t *testing.T) string {
	t.Helper()
	_, sourcePath, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve status acceptance source path")
	}
	path := filepath.Join(filepath.Dir(sourcePath), "..", "..", "dist", cliBinary)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("built CLI %s is unavailable; run make build first: %v", path, err)
	}
	return path
}
