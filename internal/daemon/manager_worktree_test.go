package daemon_test

import (
	"context"
	"strings"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/view"
)

const reuseReasonStatusField = "worktree_reuse_unavailable_reason="

func (scene *worktreeSourceScene) requireReuseReason(t *testing.T, path string, wantReason view.WorktreeReuseReason) {
	t.Helper()

	status := scene.status(t, path)
	gotReason := view.WorktreeReuseReason(status.GetCodebase().GetWorktreeReuseUnavailableReason())
	if gotReason != wantReason {
		t.Fatalf("GetIndex returned the reuse reason for path %s as %q instead of the expected %q.", path, gotReason, wantReason)
	}
	hasField := strings.Contains(status.GetDisplayText(), reuseReasonStatusField)
	if wantReason == view.WorktreeReuseAvailable && hasField {
		t.Fatalf("The status text for path %s must omit the reuse reason field from output %s.", path, status.GetDisplayText())
	}
	if wantReason != view.WorktreeReuseAvailable &&
		!strings.Contains(status.GetDisplayText(), reuseReasonStatusField+string(wantReason)) {
		t.Fatalf("The status text for path %s does not contain field %q in output %s.", path, wantReason, status.GetDisplayText())
	}

	listed, err := scene.server.ListIndexes(context.Background(), &pb.ListIndexesRequest{})
	if err != nil {
		t.Fatalf("ListIndexes returned error %v.", err)
	}
	for _, codebase := range listed.GetIndexes() {
		if codebase.GetCanonicalPath() != path {
			continue
		}
		listedReason := view.WorktreeReuseReason(codebase.GetWorktreeReuseUnavailableReason())
		if listedReason != wantReason {
			t.Fatalf("ListIndexes returned the reuse reason for path %s as %q instead of the expected %q.", path, listedReason, wantReason)
		}
		return
	}
	t.Fatalf("ListIndexes does not list path %s.", path)
}

func TestWorktreeStatusReportsUnregisteredAndUnindexedDeclaredSource(t *testing.T) {
	scene := newWorktreeSourceScene(t)

	scene.mustStartIndex(t, scene.worktreeA, scene.repository, "")
	scene.waitForRedirectingStatus(t, scene.worktreeA)
	scene.requireReuseReason(t, scene.worktreeA, view.WorktreeReuseSourceNotRegistered)

	scene.mustStartIndex(t, scene.repository, "", "")
	scene.waitForRedirectingStatus(t, scene.repository)
	scene.requireReuseReason(t, scene.worktreeA, view.WorktreeReuseSourceNotIndexed)
}

func TestWorktreeStatusOmitsReuseReasonWithoutDeclaredSource(t *testing.T) {
	scene := newWorktreeSourceScene(t)

	scene.mustStartIndex(t, scene.repository, "", "")
	scene.waitForRedirectingStatus(t, scene.repository)
	scene.mustStartIndex(t, scene.worktreeA, "", "")
	scene.waitForRedirectingStatus(t, scene.worktreeA)
	scene.requireReuseReason(t, scene.worktreeA, view.WorktreeReuseAvailable)
}
