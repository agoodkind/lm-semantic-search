package indexability

import (
	"os"
	"path/filepath"
	"testing"
)

func TestMaxFileBytesUsesDefault(t *testing.T) {
	t.Setenv("INDEX_MAX_FILE_BYTES", "")

	got := maxFileBytes()
	want := int64(2 * 1024 * 1024)
	if got != want {
		t.Fatalf("maxFileBytes() = %d, want %d", got, want)
	}
}

func TestMaxFileBytesUsesEnvOverride(t *testing.T) {
	t.Setenv("INDEX_MAX_FILE_BYTES", "17")

	got := maxFileBytes()
	if got != 17 {
		t.Fatalf("maxFileBytes() = %d, want 17", got)
	}
}

func TestMaxFileBytesDisable(t *testing.T) {
	t.Setenv("INDEX_MAX_FILE_BYTES", "0")

	got := maxFileBytes()
	if got != 0 {
		t.Fatalf("maxFileBytes() = %d, want 0", got)
	}
}

func TestMaxFileBytesInvalidFallsBackToDefault(t *testing.T) {
	t.Setenv("INDEX_MAX_FILE_BYTES", "not-a-number")

	got := maxFileBytes()
	want := int64(2 * 1024 * 1024)
	if got != want {
		t.Fatalf("maxFileBytes() = %d, want %d", got, want)
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

	keep, reason := eligibleByStat(info, 10)
	if !keep {
		t.Fatalf("eligibleByStat keep = false, want true with reason %q", reason)
	}
	if reason != SkipKeep {
		t.Fatalf("eligibleByStat reason = %q, want %q", reason, SkipKeep)
	}
}

func TestEligibleByStatSkipsDirectory(t *testing.T) {
	t.Parallel()

	tempDirectory := t.TempDir()
	info, err := os.Stat(tempDirectory)
	if err != nil {
		t.Fatalf("Stat returned error: %v", err)
	}

	keep, reason := eligibleByStat(info, 10)
	if keep {
		t.Fatal("eligibleByStat keep = true, want false for a directory")
	}
	if reason != SkipNotRegular {
		t.Fatalf("eligibleByStat reason = %q, want %q", reason, SkipNotRegular)
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

	keep, reason := eligibleByStat(info, 3)
	if keep {
		t.Fatal("eligibleByStat keep = true, want false for an oversize file")
	}
	if reason != SkipOversize {
		t.Fatalf("eligibleByStat reason = %q, want %q", reason, SkipOversize)
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

	keep, reason := eligibleByStat(info, 0)
	if !keep {
		t.Fatalf("eligibleByStat keep = false, want true with disabled cap and reason %q", reason)
	}
	if reason != SkipKeep {
		t.Fatalf("eligibleByStat reason = %q, want %q", reason, SkipKeep)
	}
}

func TestEligibleContentKeepsUTF8(t *testing.T) {
	t.Parallel()

	keep, reason := eligibleContent([]byte("package main\n"))
	if !keep {
		t.Fatalf("eligibleContent keep = false, want true with reason %q", reason)
	}
	if reason != SkipKeep {
		t.Fatalf("eligibleContent reason = %q, want %q", reason, SkipKeep)
	}
}

func TestEligibleContentSkipsNonUTF8(t *testing.T) {
	t.Parallel()

	keep, reason := eligibleContent([]byte{'g', 'o', 0xff})
	if keep {
		t.Fatal("eligibleContent keep = true, want false for non-UTF-8")
	}
	if reason != SkipNonUTF8 {
		t.Fatalf("eligibleContent reason = %q, want %q", reason, SkipNonUTF8)
	}
}
