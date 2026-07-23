package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestRewriteArgumentsScopesEntitlementsToDaemon(t *testing.T) {
	testCases := []struct {
		name      string
		signer    signerTool
		arguments []string
		want      []string
	}{
		{
			name:   "codesign engine daemon",
			signer: signerCodesign,
			arguments: []string{
				"--force", "--sign", "identity", "--identifier", "io.goodkind.daemon",
				"--options", "runtime", "--timestamp=none",
				"dist/lm-semantic-search-daemon",
			},
			want: []string{
				"--force", "--sign", "identity", "--identifier", "io.goodkind.daemon",
				"--options", "runtime", "--timestamp=none",
				"--entitlements", entitlements, "dist/lm-semantic-search-daemon",
			},
		},
		{
			name:   "codesign daemon before trailing flags",
			signer: signerCodesign,
			arguments: []string{
				"--force", "-s", "identity", "dist/lm-semantic-search-daemon",
				"--options", "runtime", "--timestamp=none",
			},
			want: []string{
				"--force", "-s", "identity", "--entitlements", entitlements,
				"dist/lm-semantic-search-daemon", "--options", "runtime",
				"--timestamp=none",
			},
		},
		{
			name:   "codesign replaces existing entitlement after target",
			signer: signerCodesign,
			arguments: []string{
				"--sign", "identity", "--identifier", "io.goodkind.daemon",
				"dist/lm-semantic-search-daemon", "--entitlements", "old.plist",
				"--options", "runtime",
			},
			want: []string{
				"--sign", "identity", "--identifier", "io.goodkind.daemon",
				"--entitlements", entitlements, "dist/lm-semantic-search-daemon",
				"--options", "runtime",
			},
		},
		{
			name:   "codesign CLI engine shape",
			signer: signerCodesign,
			arguments: []string{
				"--force", "--sign", "identity", "--options", "runtime",
				"dist/lm-semantic-search",
			},
			want: []string{
				"--force", "--sign", "identity", "--options", "runtime",
				"dist/lm-semantic-search",
			},
		},
		{
			name:   "codesign MCP with target before trailing flag",
			signer: signerCodesign,
			arguments: []string{
				"--force", "-s", "identity", "dist/lm-semantic-search-mcp",
				"--timestamp=none",
			},
			want: []string{
				"--force", "-s", "identity", "dist/lm-semantic-search-mcp",
				"--timestamp=none",
			},
		},
		{
			name:   "quill release engine daemon",
			signer: signerQuill,
			arguments: []string{
				"sign-and-notarize",
				"dist/lm-semantic-search-daemon_darwin_arm64/lm-semantic-search-daemon",
				"-vv",
			},
			want: []string{
				"sign-and-notarize",
				"dist/lm-semantic-search-daemon_darwin_arm64/lm-semantic-search-daemon",
				"-vv", "--entitlements", entitlements,
			},
		},
		{
			name:   "quill daemon after flag with value",
			signer: signerQuill,
			arguments: []string{
				"sign-and-notarize", "--notarize-timeout", "10m",
				"dist/lm-semantic-search-daemon_darwin_arm64/lm-semantic-search-daemon",
				"-vv",
			},
			want: []string{
				"sign-and-notarize", "--notarize-timeout", "10m",
				"dist/lm-semantic-search-daemon_darwin_arm64/lm-semantic-search-daemon",
				"-vv", "--entitlements", entitlements,
			},
		},
		{
			name:   "quill CLI release engine shape",
			signer: signerQuill,
			arguments: []string{
				"sign-and-notarize",
				"dist/lm-semantic-search_darwin_arm64/lm-semantic-search",
				"-vv",
			},
			want: []string{
				"sign-and-notarize",
				"dist/lm-semantic-search_darwin_arm64/lm-semantic-search",
				"-vv",
			},
		},
		{
			name:   "quill MCP release engine shape",
			signer: signerQuill,
			arguments: []string{
				"sign-and-notarize",
				"dist/lm-semantic-search-mcp_darwin_arm64/lm-semantic-search-mcp",
				"-vv",
			},
			want: []string{
				"sign-and-notarize",
				"dist/lm-semantic-search-mcp_darwin_arm64/lm-semantic-search-mcp",
				"-vv",
			},
		},
		{
			name:   "codesign daemon prefix is not daemon",
			signer: signerCodesign,
			arguments: []string{
				"--force", "--sign", "identity",
				"dist/lm-semantic-search-daemon-x",
			},
			want: []string{
				"--force", "--sign", "identity",
				"dist/lm-semantic-search-daemon-x",
			},
		},
		{
			name:   "quill daemon prefix is not daemon",
			signer: signerQuill,
			arguments: []string{
				"sign-and-notarize",
				"dist/lm-semantic-search-daemon-x_darwin_arm64/lm-semantic-search-daemon-x",
				"-vv",
			},
			want: []string{
				"sign-and-notarize",
				"dist/lm-semantic-search-daemon-x_darwin_arm64/lm-semantic-search-daemon-x",
				"-vv",
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			got, err := rewriteArguments(testCase.signer, testCase.arguments)
			if err != nil {
				t.Fatalf("rewriteArguments returned error: %v", err)
			}
			if !slices.Equal(got, testCase.want) {
				t.Fatalf("rewriteArguments = %#v, want %#v", got, testCase.want)
			}
		})
	}
}

func TestResolveRealSignerRejectsWrapperExecutable(t *testing.T) {
	wrapperExecutable := filepath.Join(t.TempDir(), "codesign")
	if err := os.WriteFile(wrapperExecutable, []byte("wrapper"), 0o755); err != nil {
		t.Fatalf("WriteFile wrapper returned error: %v", err)
	}
	t.Setenv(realCodesignEnv, wrapperExecutable)

	_, err := resolveRealSigner(signerCodesign, wrapperExecutable)
	if err == nil {
		t.Fatal("resolveRealSigner returned nil error, want recursion rejection")
	}
	if !strings.Contains(err.Error(), "is the wrapper executable") {
		t.Fatalf("resolveRealSigner error = %q, want wrapper executable rejection", err)
	}
}

func TestResolveRealSignerBareNameSkipsWrapperDirectory(t *testing.T) {
	wrapperDirectory := t.TempDir()
	realSignerDirectory := t.TempDir()
	wrapperExecutable := filepath.Join(wrapperDirectory, "codesign")
	realSignerExecutable := filepath.Join(realSignerDirectory, "codesign")
	if err := os.WriteFile(wrapperExecutable, []byte("wrapper"), 0o755); err != nil {
		t.Fatalf("WriteFile wrapper returned error: %v", err)
	}
	if err := os.WriteFile(realSignerExecutable, []byte("real signer"), 0o755); err != nil {
		t.Fatalf("WriteFile real signer returned error: %v", err)
	}
	t.Setenv(realCodesignEnv, "codesign")
	t.Setenv(
		"PATH",
		strings.Join(
			[]string{wrapperDirectory, realSignerDirectory},
			string(os.PathListSeparator),
		),
	)

	resolved, err := resolveRealSigner(signerCodesign, wrapperExecutable)
	if err != nil {
		t.Fatalf("resolveRealSigner returned error: %v", err)
	}
	if resolved != realSignerExecutable {
		t.Fatalf("resolveRealSigner = %q, want %q", resolved, realSignerExecutable)
	}
}
