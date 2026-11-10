package updateopts

import "testing"

func TestIsLocalBuild(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		version string
		dirty   bool
		want    bool
	}{
		{name: "release", version: "202608122028-d1-ec6489d", want: false},
		{name: "semver release", version: "v1.4.2", want: false},
		{name: "prerelease", version: "v1.4.2-rc.1", want: false},
		{name: "dirty release", version: "202608122028-d1-ec6489d", dirty: true, want: true},
		{name: "git describe ahead", version: "202608122028-d1-ec6489d-1-gabc1234", want: true},
		{name: "dev", version: "dev", want: true},
		{name: "unknown", version: "unknown", want: true},
		{name: "unstamped", version: "", want: true},
		{name: "invalid describe hex", version: "v1.4.2-3-gzzzz", want: false},
		{name: "semver prerelease without describe count", version: "v1.4.2-gdeadbee", want: false},
		{name: "unrecognized stamp", version: "custom-build", want: true},
		{name: "malformed rolling timestamp", version: "20260812-d1-ec6489d", want: true},
		{name: "malformed rolling sequence", version: "202608122028-d.1-ec6489d", want: true},
		{name: "malformed rolling commit", version: "202608122028-d1-nothex", want: true},
		{name: "empty describe tag", version: "-3-gdeadbee", want: true},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			got := isLocalBuild(testCase.version, testCase.dirty)
			if got != testCase.want {
				t.Fatalf("isLocalBuild(%q, %v) = %v, want %v", testCase.version, testCase.dirty, got, testCase.want)
			}
		})
	}
}
