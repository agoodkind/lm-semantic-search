package tshash

import (
	"encoding/hex"
	"testing"
)

func TestPathPrefixKnownVectors(t *testing.T) {
	t.Parallel()

	cases := []struct {
		input  string
		expect string
	}{
		// Reference MD5("") = d41d8cd98f00b204e9800998ecf8427e
		{"", "d41d8cd9"},
		// Reference MD5("abc") = 900150983cd24fb0d6963f7d28e17f72
		{"abc", "90015098"},
		// Real codebase path from the user's TS snapshot (md5 first 8 hex):
		// /Users/agoodkind/Sites/lmd → hybrid_code_chunks_8c199032
		{"/Users/agoodkind/Sites/lmd", "8c199032"},
	}
	for _, tc := range cases {
		got := PathPrefix(tc.input)
		if got != tc.expect {
			t.Errorf("PathPrefix(%q) = %q want %q", tc.input, got, tc.expect)
		}
	}
}

func TestPathPrefixIsHexLowercase(t *testing.T) {
	t.Parallel()

	got := PathPrefix("the quick brown fox jumps over the lazy dog")
	if len(got) != 8 {
		t.Fatalf("length = %d, want 8", len(got))
	}
	if _, err := hex.DecodeString(got); err != nil {
		t.Fatalf("PathPrefix output is not hex: %q (%v)", got, err)
	}
}
