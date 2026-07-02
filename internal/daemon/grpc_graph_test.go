package daemon

import "testing"

func TestGraphToolDisplayTextRows(t *testing.T) {
	result := graphToolDisplayText(
		"query_graph",
		`{"structuredContent":{"rows":[{"name":"A"},{"name":"B"}]},"isError":false}`,
	)
	if result != "Graph query returned 2 rows." {
		t.Fatalf("graphToolDisplayText returned %q", result)
	}
}

func TestGraphToolDisplayTextContentText(t *testing.T) {
	result := graphToolDisplayText(
		"trace_path",
		`{"content":[{"type":"text","text":"Trace found 3 hops."}],"isError":false}`,
	)
	if result != "Trace found 3 hops." {
		t.Fatalf("graphToolDisplayText returned %q", result)
	}
}

func TestGraphToolDisplayTextInvalidJSON(t *testing.T) {
	result := graphToolDisplayText("query_graph", `{`)
	if result != "Graph tool returned an unparseable response." {
		t.Fatalf("graphToolDisplayText returned %q", result)
	}
}
