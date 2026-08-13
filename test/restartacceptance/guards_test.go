//go:build restartacceptance

package restartacceptance

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestValidateOptInRequiresExactConfirmation(t *testing.T) {
	t.Parallel()

	if err := validateOptIn(restartAcceptanceConfirmation); err != nil {
		t.Fatalf("valid confirmation: %v", err)
	}
	for _, value := range []string{"", "isolated", " isolated-clone"} {
		if err := validateOptIn(value); err == nil {
			t.Fatalf("validateOptIn(%q) returned no error", value)
		}
	}
}

func TestRequiredDirectoryFromEnvironmentRejectsMissingRelativeAndSymlinkPaths(t *testing.T) {
	realDirectory, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatalf("resolve real directory: %v", err)
	}
	if err := os.Mkdir(filepath.Join(realDirectory, "nested"), 0o700); err != nil {
		t.Fatalf("create nested directory: %v", err)
	}
	symlinkParent, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatalf("resolve symlink parent: %v", err)
	}
	symlink := filepath.Join(symlinkParent, "linked")
	if err := os.Symlink(realDirectory, symlink); err != nil {
		t.Fatalf("create symlink: %v", err)
	}
	for _, testCase := range []struct {
		name  string
		value string
	}{
		{name: "missing"},
		{name: "relative", value: "relative/path"},
		{name: "symlink", value: symlink},
		{name: "symlink_ancestor", value: filepath.Join(symlink, "nested")},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			variable := "LMS_RESTART_ACCEPTANCE_TEST_" + strings.ToUpper(testCase.name)
			t.Setenv(variable, testCase.value)
			if _, err := requiredDirectoryFromEnvironment(variable); err == nil {
				t.Fatalf("environment path %q was accepted", testCase.value)
			}
		})
	}

	variable := "LMS_RESTART_ACCEPTANCE_TEST_REAL"
	t.Setenv(variable, realDirectory)
	got, err := requiredDirectoryFromEnvironment(variable)
	if err != nil {
		t.Fatalf("resolve real directory: %v", err)
	}
	if got != realDirectory {
		t.Fatalf("directory = %q, want %q", got, realDirectory)
	}

	redundantVariable := "LMS_RESTART_ACCEPTANCE_TEST_REDUNDANT"
	redundantPath := realDirectory + string(os.PathSeparator) + "." + string(os.PathSeparator)
	t.Setenv(redundantVariable, redundantPath)
	got, err = requiredDirectoryFromEnvironment(redundantVariable)
	if err != nil {
		t.Fatalf("resolve directory with redundant segments: %v", err)
	}
	if got != realDirectory {
		t.Fatalf("redundant directory = %q, want %q", got, realDirectory)
	}
}

func TestNewRunIDUsesUTCAndEightLowercaseHexCharacters(t *testing.T) {
	t.Parallel()

	now := time.Date(2026, time.August, 11, 18, 2, 3, 0, time.FixedZone("PDT", -7*60*60))
	runID, err := newRunID(now, bytes.NewReader([]byte{0xab, 0xcd, 0xef, 0x01}))
	if err != nil {
		t.Fatalf("newRunID: %v", err)
	}
	if runID != "20260812T010203Z-abcdef01" {
		t.Fatalf("run ID = %q", runID)
	}
}

func TestValidateRunRootRejectsEscapeAndPreexistingPath(t *testing.T) {
	t.Parallel()

	parent := filepath.Join(t.TempDir(), "lms-restart-acceptance")
	valid := filepath.Join(parent, "20260812T010203Z-abcdef01")
	if err := validateRunRoot(parent, valid); err != nil {
		t.Fatalf("valid run root: %v", err)
	}
	for _, path := range []string{
		filepath.Join(filepath.Dir(parent), "20260812T010203Z-abcdef01"),
		filepath.Join(parent, "wrong"),
		parent,
	} {
		if err := validateRunRoot(parent, path); err == nil {
			t.Fatalf("validateRunRoot(%q) returned no error", path)
		}
	}
	if err := os.MkdirAll(valid, 0o755); err != nil {
		t.Fatalf("create existing root: %v", err)
	}
	if err := validateRunRoot(parent, valid); err == nil {
		t.Fatal("preexisting run root was accepted")
	}
}

func TestRequiredRestoreBytesRoundsUpToOneHundredTwentyFivePercent(t *testing.T) {
	t.Parallel()

	if got := requiredRestoreBytes([]int64{1, 2, 2, 2}); got != 9 {
		t.Fatalf("required bytes = %d, want 9", got)
	}
}

func TestValidatePortsRejectsBoundPort(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	t.Cleanup(func() { _ = listener.Close() })
	port := listener.Addr().(*net.TCPAddr).Port

	if err := validatePorts([]int{port}); err == nil {
		t.Fatalf("bound port %d was accepted", port)
	}
	if err := listener.Close(); err != nil {
		t.Fatalf("close listener: %v", err)
	}
	if err := validatePorts([]int{port}); err != nil {
		t.Fatalf("free port %d: %v", port, err)
	}
}

func TestValidateInstalledBinariesRequiresRegularExecutables(t *testing.T) {
	home := t.TempDir()
	binDir := filepath.Join(home, ".local", "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatalf("create bin dir: %v", err)
	}
	for _, name := range []string{"lm-semantic-search-daemon", "lm-semantic-search", "clyde"} {
		path := filepath.Join(binDir, name)
		if err := os.WriteFile(path, []byte("binary"), 0o755); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	paths, err := validateInstalledBinaries(home)
	if err != nil {
		t.Fatalf("validate binaries: %v", err)
	}
	if paths.Clyde != filepath.Join(binDir, "clyde") {
		t.Fatalf("Clyde path = %q", paths.Clyde)
	}
	if err := os.Chmod(paths.CLI, 0o644); err != nil {
		t.Fatalf("remove executable bit: %v", err)
	}
	if _, err := validateInstalledBinaries(home); err == nil {
		t.Fatal("non-executable CLI was accepted")
	}
}

func TestVerifyChecksumsValidatesEveryLiteralManifestEntry(t *testing.T) {
	root := t.TempDir()
	want := make(map[string]string)
	var manifest strings.Builder
	for _, fixture := range []struct {
		name string
		body string
	}{
		{name: "one.tar", body: "one"},
		{name: "two.yaml", body: "two"},
	} {
		if err := os.WriteFile(filepath.Join(root, fixture.name), []byte(fixture.body), 0o600); err != nil {
			t.Fatalf("write fixture: %v", err)
		}
		hash := fmt.Sprintf("%x", sha256.Sum256([]byte(fixture.body)))
		want[fixture.name] = hash
		fmt.Fprintf(&manifest, "%s  %s\n", hash, fixture.name)
	}
	if err := os.WriteFile(filepath.Join(root, checksumManifestName), []byte(manifest.String()), 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	if err := verifyChecksums(root, want); err != nil {
		t.Fatalf("verify checksums: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "two.yaml"), []byte("changed"), 0o600); err != nil {
		t.Fatalf("change fixture: %v", err)
	}
	if err := verifyChecksums(root, want); err == nil {
		t.Fatal("changed second manifest entry was accepted")
	}
}

func TestVerifyChecksumsRejectsManifestTraversal(t *testing.T) {
	root := t.TempDir()
	hash := fmt.Sprintf("%x", sha256.Sum256([]byte("outside")))
	manifest := fmt.Sprintf("%s  ../outside\n", hash)
	if err := os.WriteFile(filepath.Join(root, checksumManifestName), []byte(manifest), 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	if err := verifyChecksums(root, map[string]string{"../outside": hash}); err == nil {
		t.Fatal("manifest traversal was accepted")
	}
}
