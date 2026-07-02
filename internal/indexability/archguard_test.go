package indexability

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestArchguardSubmoduleDetectionStaysInIndexability(t *testing.T) {
	repoRoot := filepath.Clean(filepath.Join("..", ".."))
	allowedPathInsideSubmodule := map[string]bool{
		filepath.Join("internal", "gitworktree", "gitworktree.go"):      true,
		filepath.Join("internal", "gitworktree", "gitworktree_test.go"): true,
	}

	err := filepath.WalkDir(repoRoot, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			switch entry.Name() {
			case ".git", ".make", "gen", "third_party":
				return filepath.SkipDir
			}
			return nil
		}
		if filepath.Ext(path) != ".go" {
			return nil
		}
		relPath, relErr := filepath.Rel(repoRoot, path)
		if relErr != nil {
			return relErr
		}
		relPath = filepath.Clean(relPath)
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return readErr
		}
		content := string(data)
		if strings.Contains(content, "PathInsideSubmodule(") && !strings.HasPrefix(relPath, filepath.Join("internal", "indexability")+string(os.PathSeparator)) && !allowedPathInsideSubmodule[relPath] {
			t.Fatalf("%s calls PathInsideSubmodule; submodule admission must route through internal/indexability", relPath)
		}
		if strings.Contains(content, `".gitmodules"`) && !strings.HasPrefix(relPath, filepath.Join("internal", "indexability")+string(os.PathSeparator)) && !strings.HasSuffix(relPath, "_test.go") {
			t.Fatalf("%s reads .gitmodules; submodule admission must route through internal/indexability", relPath)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("WalkDir returned error: %v", err)
	}
}
