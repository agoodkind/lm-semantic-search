// Command daemon-entitlements-signer scopes macOS signing entitlements to the daemon.
package main

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
)

type signerTool string

const (
	signerCodesign signerTool = "codesign"
	signerQuill    signerTool = "quill"

	daemonBinary    = "lm-semantic-search-daemon"
	entitlements    = "packaging/macos/lm-semantic-search.entitlements"
	errorPrefix     = "daemon-entitlements-signing-wrapper"
	realCodesignEnv = "LMS_REAL_CODESIGN"
	realQuillEnv    = "LMS_REAL_QUILL"
)

var optionValueFlags = map[signerTool]map[string]struct{}{
	signerCodesign: {
		"-i":                  {},
		"-o":                  {},
		"-r":                  {},
		"-s":                  {},
		"--entitlements":      {},
		"--file-list":         {},
		"--identifier":        {},
		"--options":           {},
		"--prefix":            {},
		"--preserve-metadata": {},
		"--requirements":      {},
		"--resource-rules":    {},
		"--sign":              {},
		"--timestamp":         {},
	},
	signerQuill: {
		"--apple-id":         {},
		"--entitlements":     {},
		"--notarize-timeout": {},
		"--notary-issuer":    {},
		"--notary-key":       {},
		"--notary-key-id":    {},
		"--output":           {},
		"--p12":              {},
		"--p12-password":     {},
		"--password":         {},
		"--team-id":          {},
	},
}

func main() {
	slog.Debug(
		"daemon entitlements signer starting",
		"signer",
		filepath.Base(os.Args[0]),
		"argument_count",
		len(os.Args)-1,
	)
	if err := run(os.Args[0], os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", errorPrefix, err)
		os.Exit(1)
	}
}

func run(executable string, arguments []string) error {
	signer := signerTool(filepath.Base(executable))
	wrapperExecutable, err := os.Executable()
	if err != nil {
		return wrapError("locate wrapper executable", err)
	}
	realSigner, err := resolveRealSigner(signer, wrapperExecutable)
	if err != nil {
		return err
	}
	rewritten, err := rewriteArguments(signer, arguments)
	if err != nil {
		return err
	}
	argv := make([]string, 0, len(rewritten)+1)
	argv = append(argv, realSigner)
	argv = append(argv, rewritten...)
	slog.Debug("execute real signer", "signer", signer, "argument_count", len(rewritten))
	// #nosec G702 -- realSigner is an absolute executable path checked against
	// the wrapper before syscall.Exec replaces this process.
	return wrapError("execute real signer", syscall.Exec(realSigner, argv, os.Environ()))
}

func resolveRealSigner(signer signerTool, wrapperExecutable string) (string, error) {
	var configured string
	var fallback string
	switch signer {
	case signerCodesign:
		configured = strings.TrimSpace(os.Getenv(realCodesignEnv))
		fallback = "/usr/bin/codesign"
	case signerQuill:
		configured = strings.TrimSpace(os.Getenv(realQuillEnv))
		fallback = string(signerQuill)
	default:
		return "", fmt.Errorf("unsupported tool %s", signer)
	}
	candidate := configured
	if candidate == "" {
		candidate = fallback
	}
	resolved, err := resolveExecutable(candidate, filepath.Dir(wrapperExecutable))
	if err != nil {
		return "", err
	}
	sameExecutable, err := executablePathsMatch(resolved, wrapperExecutable)
	if err != nil {
		return "", err
	}
	if sameExecutable {
		return "", fmt.Errorf(
			"resolved real %s signer %q is the wrapper executable",
			signer,
			resolved,
		)
	}
	return resolved, nil
}

func resolveExecutable(candidate string, wrapperDirectory string) (string, error) {
	if filepath.IsAbs(candidate) || strings.ContainsRune(candidate, os.PathSeparator) {
		absoluteCandidate, err := filepath.Abs(candidate)
		if err != nil {
			return "", wrapError("make configured signer path absolute", err)
		}
		resolved, err := exec.LookPath(absoluteCandidate)
		if err != nil {
			return "", wrapError("locate configured signer", err)
		}
		return resolved, nil
	}
	return findExecutableOnPath(candidate, os.Getenv("PATH"), wrapperDirectory)
}

func findExecutableOnPath(
	executableName string,
	pathValue string,
	wrapperDirectory string,
) (string, error) {
	for _, pathEntry := range filepath.SplitList(pathValue) {
		if pathEntry == "" {
			pathEntry = "."
		}
		sameDirectory, err := directoryPathsMatch(pathEntry, wrapperDirectory)
		if err != nil {
			return "", err
		}
		if sameDirectory {
			continue
		}
		absoluteEntry, err := filepath.Abs(pathEntry)
		if err != nil {
			return "", wrapError("make signer PATH entry absolute", err)
		}
		resolved, err := exec.LookPath(filepath.Join(absoluteEntry, executableName))
		if err == nil {
			return resolved, nil
		}
		if !errors.Is(err, exec.ErrNotFound) {
			return "", wrapError("locate signer on PATH", err)
		}
	}
	return "", fmt.Errorf("%s not found outside wrapper directory", executableName)
}

func directoryPathsMatch(first string, second string) (bool, error) {
	firstAbsolute, err := filepath.Abs(first)
	if err != nil {
		return false, wrapError("make first directory path absolute", err)
	}
	secondAbsolute, err := filepath.Abs(second)
	if err != nil {
		return false, wrapError("make second directory path absolute", err)
	}
	firstInfo, firstErr := os.Stat(firstAbsolute)
	secondInfo, secondErr := os.Stat(secondAbsolute)
	if firstErr == nil && secondErr == nil {
		return os.SameFile(firstInfo, secondInfo), nil
	}
	return filepath.Clean(firstAbsolute) == filepath.Clean(secondAbsolute), nil
}

func executablePathsMatch(first string, second string) (bool, error) {
	// #nosec G703 -- both paths are resolved executable candidates, and this
	// metadata check prevents the selected signer from recursing into the wrapper.
	firstInfo, err := os.Stat(first)
	if err != nil {
		return false, wrapError("inspect real signer executable", err)
	}
	secondInfo, err := os.Stat(second)
	if err != nil {
		return false, wrapError("inspect signer wrapper executable", err)
	}
	return os.SameFile(firstInfo, secondInfo), nil
}

func rewriteArguments(signer signerTool, arguments []string) ([]string, error) {
	signingInvocation, err := isSigningInvocation(signer, arguments)
	if err != nil {
		return nil, err
	}
	if !signingInvocation {
		return append([]string(nil), arguments...), nil
	}
	rewritten, err := stripEntitlements(arguments)
	if err != nil {
		return nil, err
	}
	target, targetIndex, err := signingTarget(signer, rewritten)
	if err != nil {
		return nil, err
	}
	if filepath.Base(target) != daemonBinary {
		return rewritten, nil
	}
	switch signer {
	case signerCodesign:
		withEntitlements := make([]string, 0, len(rewritten)+2)
		withEntitlements = append(withEntitlements, rewritten[:targetIndex]...)
		withEntitlements = append(withEntitlements, "--entitlements", entitlements)
		withEntitlements = append(withEntitlements, rewritten[targetIndex:]...)
		return withEntitlements, nil
	case signerQuill:
		return append(rewritten, "--entitlements", entitlements), nil
	default:
		return nil, fmt.Errorf("unsupported tool %s", signer)
	}
}

func isSigningInvocation(signer signerTool, arguments []string) (bool, error) {
	switch signer {
	case signerCodesign:
		for _, argument := range arguments {
			if argument == "--sign" || argument == "-s" ||
				strings.HasPrefix(argument, "--sign=") {
				return true, nil
			}
		}
		return false, nil
	case signerQuill:
		return len(arguments) > 0 && arguments[0] == "sign-and-notarize", nil
	default:
		return false, fmt.Errorf("unsupported tool %s", signer)
	}
}

func signingTarget(signer signerTool, arguments []string) (string, int, error) {
	valueFlags, found := optionValueFlags[signer]
	if !found {
		return "", -1, fmt.Errorf("unsupported tool %s", signer)
	}
	skipValue := false
	for argumentIndex, argument := range arguments {
		if skipValue {
			skipValue = false
			continue
		}
		if _, takesValue := valueFlags[argument]; takesValue {
			skipValue = true
			continue
		}
		if strings.HasPrefix(argument, "-") {
			continue
		}
		if signer == signerQuill && argumentIndex == 0 {
			continue
		}
		if signedBinaryName(filepath.Base(argument)) {
			return argument, argumentIndex, nil
		}
	}
	return "", -1, nil
}

func signedBinaryName(baseName string) bool {
	return baseName == daemonBinary ||
		baseName == "lm-semantic-search" ||
		baseName == "lm-semantic-search-mcp"
}

func stripEntitlements(arguments []string) ([]string, error) {
	filtered := make([]string, 0, len(arguments))
	skipValue := false
	for _, argument := range arguments {
		if skipValue {
			skipValue = false
			continue
		}
		if argument == "--entitlements" {
			skipValue = true
			continue
		}
		if strings.HasPrefix(argument, "--entitlements=") {
			continue
		}
		filtered = append(filtered, argument)
	}
	if skipValue {
		return nil, errors.New("--entitlements requires a value")
	}
	return filtered, nil
}

func wrapError(operation string, err error) error {
	slog.Error(operation+" failed", "err", err)
	return fmt.Errorf("%s: %w", operation, err)
}
