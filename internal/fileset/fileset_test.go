package fileset

import (
	"os"
	"path/filepath"
	"testing"
)

func TestMaxFileBytesUsesDefault(t *testing.T) {
	t.Setenv("INDEX_MAX_FILE_BYTES", "")

	got := MaxFileBytes()
	want := int64(2 * 1024 * 1024)
	if got != want {
		t.Fatalf("MaxFileBytes() = %d, want %d", got, want)
	}
}

func TestMaxFileBytesUsesEnvOverride(t *testing.T) {
	t.Setenv("INDEX_MAX_FILE_BYTES", "17")

	got := MaxFileBytes()
	if got != 17 {
		t.Fatalf("MaxFileBytes() = %d, want 17", got)
	}
}

func TestMaxFileBytesDisable(t *testing.T) {
	t.Setenv("INDEX_MAX_FILE_BYTES", "0")

	got := MaxFileBytes()
	if got != 0 {
		t.Fatalf("MaxFileBytes() = %d, want 0", got)
	}
}

func TestMaxFileBytesInvalidFallsBackToDefault(t *testing.T) {
	t.Setenv("INDEX_MAX_FILE_BYTES", "not-a-number")

	got := MaxFileBytes()
	want := int64(2 * 1024 * 1024)
	if got != want {
		t.Fatalf("MaxFileBytes() = %d, want %d", got, want)
	}
}

func TestEligibleByStatKeepsRegularFile(t *testing.T) {
	t.Parallel()

	tempDirectory := t.TempDir()
	path := filepath.Join(tempDirectory, "small.go")
	if err := os.WriteFile(path, []byte("ok\n"), 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat returned error: %v", err)
	}

	keep, reason := EligibleByStat(info, 10)
	if !keep {
		t.Fatalf("EligibleByStat keep = false, want true with reason %q", reason)
	}
	if reason != SkipKeep {
		t.Fatalf("EligibleByStat reason = %q, want %q", reason, SkipKeep)
	}
}

func TestEligibleByStatSkipsDirectory(t *testing.T) {
	t.Parallel()

	tempDirectory := t.TempDir()
	info, err := os.Stat(tempDirectory)
	if err != nil {
		t.Fatalf("Stat returned error: %v", err)
	}

	keep, reason := EligibleByStat(info, 10)
	if keep {
		t.Fatal("EligibleByStat keep = true, want false for a directory")
	}
	if reason != SkipNotRegular {
		t.Fatalf("EligibleByStat reason = %q, want %q", reason, SkipNotRegular)
	}
}

func TestEligibleByStatSkipsOversize(t *testing.T) {
	t.Parallel()

	tempDirectory := t.TempDir()
	path := filepath.Join(tempDirectory, "large.go")
	if err := os.WriteFile(path, []byte("large\n"), 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat returned error: %v", err)
	}

	keep, reason := EligibleByStat(info, 3)
	if keep {
		t.Fatal("EligibleByStat keep = true, want false for an oversize file")
	}
	if reason != SkipOversize {
		t.Fatalf("EligibleByStat reason = %q, want %q", reason, SkipOversize)
	}
}

func TestEligibleByStatDisabledCapKeepsRegularFile(t *testing.T) {
	t.Parallel()

	tempDirectory := t.TempDir()
	path := filepath.Join(tempDirectory, "large.go")
	if err := os.WriteFile(path, []byte("large\n"), 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat returned error: %v", err)
	}

	keep, reason := EligibleByStat(info, 0)
	if !keep {
		t.Fatalf("EligibleByStat keep = false, want true with disabled cap and reason %q", reason)
	}
	if reason != SkipKeep {
		t.Fatalf("EligibleByStat reason = %q, want %q", reason, SkipKeep)
	}
}

func TestEligibleContentKeepsUTF8(t *testing.T) {
	t.Parallel()

	keep, reason := EligibleContent([]byte("package main\n"))
	if !keep {
		t.Fatalf("EligibleContent keep = false, want true with reason %q", reason)
	}
	if reason != SkipKeep {
		t.Fatalf("EligibleContent reason = %q, want %q", reason, SkipKeep)
	}
}

func TestEligibleContentSkipsNonUTF8(t *testing.T) {
	t.Parallel()

	keep, reason := EligibleContent([]byte{'g', 'o', 0xff})
	if keep {
		t.Fatal("EligibleContent keep = true, want false for non-UTF-8")
	}
	if reason != SkipNonUTF8 {
		t.Fatalf("EligibleContent reason = %q, want %q", reason, SkipNonUTF8)
	}
}
