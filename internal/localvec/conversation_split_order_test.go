package localvec

import (
	"strings"
	"testing"
)

// TestSplitPiecesRebuildInTheirRecordedOrder pins the ordering the two loaders
// have to share.
//
// One stored path can hold several pieces when a message's text was cut again to
// fit the embedding budget. Those pieces are told apart only by their recorded
// split position, so ordering by path alone concatenates them in whatever order
// the snapshot returned. A text that rebuilds differently from the one delivered
// makes an unchanged message read as changed on every sync, be re-sent, and have
// its derived rows removed as orphans, for as long as the conversation exists.
//
// The rows here arrive in reverse split order, which is what a snapshot is free
// to do.
func TestSplitPiecesRebuildInTheirRecordedOrder(t *testing.T) {
	t.Parallel()

	assemblies := map[int32]*conversationAssembly{
		0: {
			role:              "assistant",
			roleFromBase:      true,
			hasDerivedContent: false,
			parts: []conversationPart{
				{index: 0, splitPart: 2, splitPartRecorded: true, content: "gamma"},
				{index: 0, splitPart: 0, splitPartRecorded: true, content: "alpha"},
				{index: 0, splitPart: 1, splitPartRecorded: true, content: "beta"},
			},
		},
	}

	messages := assembleConversationMessages(assemblies)
	if got := messages[0].Text; got != "alphabetagamma" {
		t.Fatalf("rebuilt text = %q, want %q", got, "alphabetagamma")
	}
}

// TestPathOrderStillWinsOverSplitOrder pins that the path index remains the
// outer key, so a message split across several paths rebuilds in path order and
// only ties inside one path fall through to the recorded split position.
func TestPathOrderStillWinsOverSplitOrder(t *testing.T) {
	t.Parallel()

	assemblies := map[int32]*conversationAssembly{
		0: {
			role:              "assistant",
			roleFromBase:      true,
			hasDerivedContent: false,
			parts: []conversationPart{
				{index: 1, splitPart: 0, splitPartRecorded: true, content: "second"},
				{index: 0, splitPart: 1, splitPartRecorded: true, content: "-tail"},
				{index: 0, splitPart: 0, splitPartRecorded: true, content: "first"},
			},
		},
	}

	messages := assembleConversationMessages(assemblies)
	if got := messages[0].Text; got != "first-tailsecond" {
		t.Fatalf("rebuilt text = %q, want %q", got, "first-tailsecond")
	}
}

// TestALegacyPieceWithoutARecordedPositionStillOrders covers a row written
// before the split position was persisted. It must not be read back as position
// zero and displace a piece that genuinely holds that position.
func TestALegacyPieceWithoutARecordedPositionStillOrders(t *testing.T) {
	t.Parallel()

	assemblies := map[int32]*conversationAssembly{
		0: {
			role:              "assistant",
			roleFromBase:      true,
			hasDerivedContent: false,
			parts: []conversationPart{
				{index: 0, splitPart: 0, splitPartRecorded: false, content: "legacy"},
				{index: 0, splitPart: 0, splitPartRecorded: true, content: "recorded"},
			},
		},
	}

	messages := assembleConversationMessages(assemblies)
	text := messages[0].Text
	if !strings.Contains(text, "legacy") || !strings.Contains(text, "recorded") {
		t.Fatalf("rebuilt text = %q, want both pieces present", text)
	}
	if strings.Index(text, "recorded") > strings.Index(text, "legacy") {
		t.Fatalf(
			"rebuilt text = %q, want the row with a recorded position first, matching the Milvus loader",
			text,
		)
	}
}
