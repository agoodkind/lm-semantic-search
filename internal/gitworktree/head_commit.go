package gitworktree

import (
	"path/filepath"
	"slices"
	"strings"
)

const packedRefsFileName = "packed-refs"

const symbolicRefDepthLimit = 5

const (
	sha1ObjectNameLength   = 40
	sha256ObjectNameLength = 64
)

// HeadCommit returns the commit object name for HEAD of the worktree at worktreeRoot from on-disk files without starting a git process.
// HeadCommit reads a detached HEAD directly or resolves a branch by reading the loose ref file under the common directory, then packed-refs.
// The boolean is false when HEAD does not resolve to an object name.
func HeadCommit(worktreeRoot string) (string, bool) {
	gitDirPath, _, ok := gitDirForRoot(worktreeRoot)
	if !ok {
		return "", false
	}
	commonDir, ok := commonDirFor(gitDirPath)
	if !ok {
		return "", false
	}
	_, detached, head := parseHead(gitDirPath)
	if head == "" {
		return "", false
	}
	if detached {
		return head, isObjectName(head)
	}
	return resolveRef(commonDir, head)
}

func resolveRef(commonDir string, ref string) (string, bool) {
	current := ref
	for range symbolicRefDepthLimit {
		if !validRefName(current) {
			return "", false
		}
		content, found := readTrimmed(filepath.Join(commonDir, filepath.FromSlash(current)))
		if !found || content == "" {
			return packedRef(commonDir, current)
		}
		target, symbolic := strings.CutPrefix(content, refPrefix)
		if !symbolic {
			return content, isObjectName(content)
		}
		current = strings.TrimSpace(target)
	}
	return "", false
}

func packedRef(commonDir string, ref string) (string, bool) {
	content, found := readTrimmed(filepath.Join(commonDir, packedRefsFileName))
	if !found {
		return "", false
	}
	for line := range strings.SplitSeq(content, "\n") {
		objectName, refName, hasRef := strings.Cut(strings.TrimSpace(line), " ")
		if !hasRef || refName != ref {
			continue
		}
		return objectName, isObjectName(objectName)
	}
	return "", false
}

func validRefName(ref string) bool {
	if !strings.HasPrefix(ref, "refs/") {
		return false
	}
	return !slices.Contains(strings.Split(ref, "/"), "..")
}

func isObjectName(value string) bool {
	if len(value) != sha1ObjectNameLength && len(value) != sha256ObjectNameLength {
		return false
	}
	for _, character := range value {
		isDigit := character >= '0' && character <= '9'
		isLowerHex := character >= 'a' && character <= 'f'
		if !isDigit && !isLowerHex {
			return false
		}
	}
	return true
}
