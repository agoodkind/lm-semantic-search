package response

import (
	"strings"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

func TestFormatProtoHumanUsesDisplayText(t *testing.T) {
	t.Parallel()

	message := &pb.GetIndexResponse{DisplayText: "line one\nline two"}
	formatted, err := FormatProto(ModeHuman, message)
	if err != nil {
		t.Fatalf("FormatProto returned error: %v", err)
	}
	if formatted != "line one\nline two" {
		t.Fatalf("FormatProto returned %q", formatted)
	}
}

func TestFormatProtoSingleLineUsesFirstLine(t *testing.T) {
	t.Parallel()

	message := &pb.GetIndexResponse{DisplayText: "\nline one\nline two"}
	formatted, err := FormatProto(ModeSingleLine, message)
	if err != nil {
		t.Fatalf("FormatProto returned error: %v", err)
	}
	if formatted != "line one" {
		t.Fatalf("FormatProto returned %q", formatted)
	}
}

func TestFormatProtoJSONUsesCompactJSON(t *testing.T) {
	t.Parallel()

	message := &pb.GetIndexResponse{DisplayText: "line one", Tracked: true}
	formatted, err := FormatProto(ModeJSON, message)
	if err != nil {
		t.Fatalf("FormatProto returned error: %v", err)
	}
	if strings.Contains(formatted, "\n") {
		t.Fatalf("FormatProto returned multiline JSON: %q", formatted)
	}
	if !strings.Contains(formatted, "\"displayText\":\"line one\"") {
		t.Fatalf("FormatProto returned unexpected JSON: %q", formatted)
	}
}

// searchable is proto3 optional so a caller can tell "the daemon answered no"
// apart from "the daemon never answered". A message that never sets Searchable
// represents the second case, a probe the daemon has no verdict for, and this
// asserts against the encoded bytes because the omission happens in the
// protojson encoder, not in the pb.GetIndexResponse struct itself.
func TestMarshalCompactJSONOmitsUnsetSearchable(t *testing.T) {
	t.Parallel()

	message := &pb.GetIndexResponse{DisplayText: "line one", Tracked: true}
	formatted, err := MarshalCompactJSON(message)
	if err != nil {
		t.Fatalf("MarshalCompactJSON returned error: %v", err)
	}
	if strings.Contains(formatted, "searchable") {
		t.Fatalf("MarshalCompactJSON carried a searchable key for an unset field: %q", formatted)
	}
}
