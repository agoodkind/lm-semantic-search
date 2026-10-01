//go:build live

package live

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/model"
	"google.golang.org/protobuf/proto"
)

func (h *harness) waitJob(jobID string) model.Job {
	h.t.Helper()
	deadline := time.Now().Add(jobPollTimeout)
	for time.Now().Before(deadline) {
		job, found := h.manager.GetJob(jobID)
		if found {
			switch job.State {
			case model.JobStateCompleted, model.JobStateFailed, model.JobStateCancelled:
				return job
			}
		}
		time.Sleep(jobPollInterval)
	}
	h.t.Fatalf("job %s did not reach a terminal state within %s", jobID, jobPollTimeout)
	return model.Job{}
}

func (h *harness) countRowsWithPrefix(prefix string) int64 {
	h.t.Helper()
	expression := fmt.Sprintf(`%s like "%s%%"`, relativePathField, prefix)
	resultSet, err := h.milvus.Query(correlatedContext(), milvusclient.NewQueryOption(h.collectionName).
		WithFilter(expression).
		WithOutputFields(countOutputField).
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		h.t.Fatalf("Milvus count query for %q returned error: %v", prefix, err)
	}
	column := resultSet.GetColumn(countOutputField)
	if column == nil {
		h.t.Fatalf("Milvus count query for %q returned no count column", prefix)
	}
	total, err := column.GetAsInt64(0)
	if err != nil {
		h.t.Fatalf("read count column for %q returned error: %v", prefix, err)
	}
	return total
}

func requireCompleted(t *testing.T, job model.Job, label string) {
	t.Helper()
	if job.State != model.JobStateCompleted {
		message := ""
		if job.Error != nil {
			message = job.Error.Message
		}
		t.Fatalf("%s job state = %q, want completed (error: %s)", label, job.State, message)
	}
}

func progressString(job model.Job) string {
	return fmt.Sprintf(
		"progress: state=%s filesTotal=%d filesModified=%d filesEmbedded=%d chunksProcessed=%d chunksReused=%d chunksEmbedded=%d",
		job.State, job.Progress.FilesTotal, job.Progress.FilesModified, job.Progress.FilesEmbedded,
		job.Progress.ChunksProcessed, job.Progress.ChunksReused, job.Progress.ChunksEmbedded,
	)
}

func liveCollectionDeclaration() model.CollectionDeclaration {
	return model.CollectionDeclaration{ItemIDColumn: "itemId", Scalars: []model.ScalarColumn{
		{Name: "itemId", Type: model.ScalarTypeString, MaxLength: 512},
	}}
}

func (h *harness) upsert(items map[string][]*pb.CollectionRow, reconcile pb.CollectionReconcileMode, backfill bool, force bool) model.Job {
	h.t.Helper()
	rows := make([]*pb.CollectionRow, 0)
	manifest := make(map[string]string, len(items))
	keys := make([]string, 0, len(items))
	for itemID := range items {
		keys = append(keys, itemID)
	}
	sort.Strings(keys)
	for _, itemID := range keys {
		hasher := sha256.New()
		for ordinal, row := range items[itemID] {
			copied := proto.Clone(row).(*pb.CollectionRow)
			copied.ItemId = itemID
			if copied.RowKey == "" {
				copied.RowKey = fmt.Sprintf("items/%s/%d", itemID, ordinal)
			}
			fmt.Fprintf(hasher, "%s\x00%s\x00", copied.RowKey, copied.Text)
			rows = append(rows, copied)
		}
		manifest[itemID] = hex.EncodeToString(hasher.Sum(nil))
	}
	response, err := h.sendGeneric(h.collectionID, rows, manifest, reconcile, backfill, force)
	if err != nil {
		h.t.Fatalf("UpsertCollectionItemsStream: %v", err)
	}
	return h.waitJob(response.GetJobId())
}

func seedItems() map[string][]*pb.CollectionRow {
	return map[string][]*pb.CollectionRow{
		"live-a": {{Text: "alpha item text"}},
		"live-b": {{Text: "bravo item text"}},
		"live-c": {{Text: "charlie item text"}},
	}
}

func (h *harness) countRowsContaining(content string) int64 {
	h.t.Helper()
	result, err := h.milvus.Query(correlatedContext(), milvusclient.NewQueryOption(h.collectionName).
		WithFilter("content like "+strconv.Quote("%"+content+"%")).
		WithOutputFields(countOutputField).
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		h.t.Fatalf("count stored content: %v", err)
	}
	count, err := result.GetColumn(countOutputField).GetAsInt64(0)
	if err != nil {
		h.t.Fatalf("read content count: %v", err)
	}
	return count
}
