package daemon

import (
	"errors"
	"fmt"
	"testing"

	"goodkind.io/lm-semantic-search/internal/adapterr"
)

// TestClassifyManagerErrorMapsPathGuardRefusals proves every path-guard
// refusal reaches the client as a typed invalid-path error instead of the
// opaque internal-error envelope, so the refusal reason is visible without
// reading the daemon log.
func TestClassifyManagerErrorMapsPathGuardRefusals(t *testing.T) {
	t.Parallel()

	cases := []error{
		fmt.Errorf("canonicalize path x: %w", errors.New(`path "chat:///x" looks like a URI; pass a filesystem directory instead`)),
		fmt.Errorf("canonicalize path x: %w", errors.New(`path "sub/dir" is relative; pass an absolute path or send caller_cwd`)),
		errors.New("refusing to index filesystem root /"),
		errors.New("codebase root /etc/hosts is not a directory"),
		errors.New("refusing to index /x because it covers daemon state root /y"),
	}
	for _, cause := range cases {
		classified := classifyManagerError("/x", cause)
		var adapterErr *adapterr.AdapterError
		if !errors.As(classified, &adapterErr) {
			t.Errorf("classifyManagerError(%q) stayed untyped; want invalid-path", cause)
			continue
		}
		if adapterErr.Class != adapterr.ClassInvalidPath {
			t.Errorf("classifyManagerError(%q) class = %q, want %q", cause, adapterErr.Class, adapterr.ClassInvalidPath)
		}
	}
}
