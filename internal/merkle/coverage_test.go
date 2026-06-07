package merkle

import "testing"

func TestSnapshotHasFileAndCoversPath(t *testing.T) {
	snapshot := Snapshot{
		ConfigDigest: "",
		Files: map[string]string{
			"Sources/App/Main.swift": "h1",
			"Sources/App/View.swift": "h2",
			"README.md":              "h3",
		},
		Inodes: nil,
	}

	// HasFile matches an exact file only, never a directory or an absent path.
	if !snapshot.HasFile("Sources/App/Main.swift") {
		t.Fatal("HasFile should find an indexed file")
	}
	if snapshot.HasFile("Sources/App") {
		t.Fatal("HasFile must not treat a directory as a file")
	}
	if snapshot.HasFile("Sources/App/Missing.swift") {
		t.Fatal("HasFile must be false for an absent file")
	}

	// CoversPath matches a file or any directory that holds an indexed file.
	coveredCases := []string{"Sources/App/Main.swift", "Sources/App", "Sources", "."}
	for _, path := range coveredCases {
		if !snapshot.CoversPath(path) {
			t.Fatalf("CoversPath(%q) should be true", path)
		}
	}
	uncoveredCases := []string{"Tests", "Sources/AppOther", "Sources/App/Missing.swift"}
	for _, path := range uncoveredCases {
		if snapshot.CoversPath(path) {
			t.Fatalf("CoversPath(%q) should be false", path)
		}
	}

	empty := Snapshot{ConfigDigest: "", Files: map[string]string{}, Inodes: nil}
	if empty.CoversPath(".") {
		t.Fatal("CoversPath('.') should be false for an empty index")
	}
}
