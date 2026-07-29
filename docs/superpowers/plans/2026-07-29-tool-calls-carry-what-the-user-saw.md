# Tool calls carry what the user saw: implementation plan

**Goal:** A stored tool call holds the text the user saw, in one row. The harness's serialization never leaves clyde.

**Shape:** Each provider parser fills a new `Display` field from its own harness's tool shapes. Export and search both read it. `input_json` comes off the wire. The engine stores one row per call instead of three.

## Constraints

Work in a linked worktree; agent-gate blocks the primary checkout and default branches. A nested clyde worktree needs its own untracked `go.work` with `use .`.

clyde `make check` is lint only, so run `make test` too. Every engine `make` needs `GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile`.

No deletion code. No backfill. Never weaken a failing `make` step. `exhaustruct` requires every struct field named. Commit with `-S`.

Leave `[conversation.semantic] enabled = false` until the last task.

---

## Task 1: clyde: `Display` and `DisplayLang` on the tool call

`internal/transcript/transcript.go`, add both fields to `ToolCall`:

```go
	// Display is what the user saw for this call: a shell command, a file path,
	// a search pattern. It belongs to the provider's parser, which is the only
	// layer that knows its own harness's tool shapes.
	Display string `json:"display,omitempty"`
	// DisplayLang names the language Display is written in, "bash" when the call
	// ran a shell and empty otherwise. It belongs to the provider's parser for
	// the same reason, since only that parser knows which of its harness's tools
	// are shells.
	DisplayLang string `json:"display_lang,omitempty"`
```

Each comment states what the field means and who owns it. Neither says anything fills or reads the field, because at this task nothing does. Task 2 extends them once a parser fills them, and Task 5 extends them again once a consumer reads them. A comment never claims behavior the commit it sits in does not have.

`go build ./...` names every literal `exhaustruct` now rejects. Add `Display: ""` and `DisplayLang: ""` to each; the next task fills them.

**Done when:** the tree builds and `go test ./internal/transcript/` passes.

---

## Task 2: clyde: four parsers fill it

Each provider package gets a `tool_display.go` with the same shape, reading its own harness's keys. It returns the display text and the language that text is written in, because the parser is the only layer that knows which of its harness's tools are shells:

```go
// toolDisplayText is what the user saw for one tool call, and the language that
// text is written in. The language is "bash" when the call ran a shell and empty
// otherwise.
//
// A tool whose shape this parser does not recognize shows nothing rather than
// its serialization. An unrecognized tool is a gap to fill here, and showing
// the JSON instead would hide the gap behind text nobody wrote.
func toolDisplayText(name string, input transcript.ToolInputJSON) (string, string) {
	if input.Len() == 0 {
		return "", ""
	}
	var parsed <providerToolInput>
	if err := json.Unmarshal(input.Raw, &parsed); err != nil {
		return "", ""
	}
	if command := strings.TrimSpace(parsed.Command); command != "" {
		return command, "bash"
	}
	for _, candidate := range []string{ /* remaining fields in priority order */ } {
		if trimmed := strings.TrimSpace(candidate); trimmed != "" {
			return trimmed, ""
		}
	}
	return "", ""
}
```

The keys, in priority order, with the first key of each row producing the `bash` language:

| package | keys |
| --- | --- |
| `providers/claude/parser` | `command`, `file_path`, `pattern`, `prompt`, `url`, `query` |
| `providers/codex/store` | `command` (argv array or string), `path`, `pattern`, `query` |
| `providers/cursor/parser` | `command`, `relative_workspace_path`, `target_file`, `query`, `pattern` |
| `providers/zed/parser` | `command`, `path`, `regex`, `query` |

Codex needs two extra cases its siblings do not. A bare JSON string input is a raw payload such as a patch and is returned as itself with an empty language, because `toolInputJSON` encodes non-JSON that way. A shell command arrives as an argv array and joins with spaces.

Fill both fields at each `transcript.ToolCall{` literal in:

- `providers/claude/parser/parse.go`, where an assistant entry's `tool_use` content block becomes a call
- `providers/codex/store/messages.go` in `toolCallHistoryMessage`
- `providers/cursor/parser/mapping.go` in `mapJSONLMessage` and `mapComposerBubble`
- `providers/zed/parser/parser.go` in `agentMessageParts`

Claude's parser also has `toolresult.go`, which fills a call's `Output` and `IsError` from a later `tool_result` block by pointer. It builds no call of its own and needs no change.

**Test each package** with a table over its own keys, asserting a shell call shows its command with the `bash` language, a file tool shows its path with an empty language, a search shows its pattern, an unknown tool shows nothing, and an empty input shows nothing.

**Also test** against a real parsed fixture that the renderer is wired into the parser rather than only existing beside it. Assert an exact expected display string for each of the specific tools the key list covers, and assert that at least 90% of the fixture's tool calls carrying an input also carry display text. The floor rather than a strict every-call assertion is deliberate: the key lists are not exhaustive, so an unrecognized tool must read as a coverage number rather than a red build.

**Extend the field comments.** `transcript.ToolCall` says only what each field means and who owns it, because at Task 1 nothing filled them. Now a parser does, so add that sentence to each: `Display` gains that each provider's parser fills it from its own harness's tool shapes, and `DisplayLang` gains that the same parser names the language. Say nothing yet about a consumer reading either field, since none does until Task 5.

**Done when:** `go test ./internal/providers/...` passes.

---

## Task 3: clyde: export shows it instead of JSON

`internal/transcript/conversation.go`, `toolFullDetailText` currently marshals `tool.Input` into the exported text. A person reads an export and never saw that JSON.

Replace the body so each line is `[tool: <name>]` plus `tool.Display` when it is set, then the output when it is set. Drop the `json.Marshal` branch.

**Test:** a call with `Display` set renders the display text and does not render `file_path`. A call with `Display` empty renders `[tool: Mystery]` alone.

**Done when:** `go test ./internal/transcript/ ./internal/conversation/` passes.

---

## Task 4: engine: the wire carries display, not serialization

`proto/lmsemanticsearch/v1/service.proto`, `ConversationToolCall` becomes five fields:

```proto
message ConversationToolCall {
  // name is the tool name, for example "Bash" or "run_command".
  string name = 1;
  // display is what the user saw for this call: the command a shell ran, the
  // path a file tool touched, the pattern a search used. Empty when clyde's
  // parser does not recognize the tool's shape.
  string display = 2;
  // lang_hint names the language the display text is written in, for example
  // "bash", "json", or "markdown". It is how the engine knows a display text is
  // a shell command it may break into program names and file paths. Empty when
  // unknown.
  string lang_hint = 4;
  // output is the tool result text when captured.
  string output = 5;
  // is_error marks a tool call that returned an error result.
  bool is_error = 6;
  reserved 3;
  reserved "command";
}
```

Field 2 is reused rather than reserved: the engine reads the wire at ingest and never replays it, so nothing stored carries the old meaning. Field 3 is reserved because a shell call's display text is already its command, and a second field would carry the same string twice.

Regenerate, then follow the compiler to rename `InputJSON` to `Display` and drop `Command` on the engine's model and its tests.

**Done when:** `go build ./... && go test ./internal/daemon/` passes.

---

## Task 5: clyde: send display, delete the guesser

`internal/conversation/semsearch/client.go`: `SemToolCall` loses `InputJSON` and `Command`, gains `Display`.

`internal/daemon/conversation_semantic_sync.go`: delete `deriveToolCommandAndLang` and `semanticToolCommandInput`. In `semanticToolCalls`, read both fields the parser already filled:

```go
		if withArguments || withOutput {
			// The provider's parser rendered what the user saw and named the
			// language it is written in. Re-deriving either here would put
			// knowledge of every harness's tool shapes into a layer that must
			// not hold it.
			projected.Display = strings.ToValidUTF8(tool.Display, "")
			projected.LangHint = tool.DisplayLang
		}
```

This file gains no tool names. A shell call is one whose parser said so.

**Test:** a projected call carries the display text, and no projected field contains `file_path`. A shell call carries a `bash` hint and a file read carries none.

**Done when:** `make test` passes in clyde.

---

## Task 6: engine: one row per tool call

`internal/daemon/manager_conversation_tools.go`: rename `conversationToolTokenContent` to `conversationToolContent`. It appends the tool's name, then the shell decomposition when `LangHint == "bash"`, then the display text:

```go
// conversationToolContent is everything one tool call stores: the tool's name,
// the program names and file paths its command decomposes into, then what the
// user saw.
//
// A tool call used to store three rows and two carried the same text, because
// this function appended the arguments to a summary while the arguments also
// took a row of their own. One row cannot repeat itself.
func conversationToolContent(toolCall model.ConversationToolCall) string {
	tokens := make([]string, 0)
	appendConversationToken(&tokens, toolCall.Name)
	display := strings.TrimSpace(toolCall.Display)
	if display != "" && toolCall.LangHint == "bash" {
		appendConversationShellTokens(&tokens, display)
	}
	appendConversationToken(&tokens, toolCall.Display)
	return strings.Join(tokens, "\n")
}
```

The old `InputJSON` append is the whole defect: it copied text that also took a row of its own.

Delete `truncateConversationToolSummary`, `conversationToolSummaryMaxBytes`, `splitConversationToolPayload`, `newConversationToolDispatcher`, `conversationToolExtension`, and `conversationToolExtensions` once the compiler shows nothing uses them.

`internal/daemon/manager_conversations.go`: replace the four-producer block inside `conversationDocumentsToStoredChunks` with one `appendStorableConversationField` per tool call, writing to `conversationToolCallPath(...)` and appending `/<partIndex>` only when it splits.

A split payload must name the tool on every piece, so a search for the tool matches the second piece:

```go
// namedToolPiece keeps the tool's name at the head of every piece of a split
// payload. The first piece already begins with the name, so it is returned as
// it is rather than carrying the name twice.
func namedToolPiece(name string, piece string, partIndex int) string {
	if partIndex == 0 {
		return piece
	}
	trimmedName := strings.TrimSpace(name)
	if trimmedName == "" {
		return piece
	}
	return trimmedName + "\n" + piece
}
```

`internal/daemon/item_source.go`: `conversationNeedsDerivedWork` expects a row under `conversationToolMessagePath(...) + "/"`, which still holds. Prove it with a test rather than by reading, because a classifier expecting a path the generator no longer writes reports work needed forever.

**Test:** a `Bash` call with `Display: "ls -la /tmp"` and `LangHint: "bash"` stores exactly one row at `convtool/claude:a/0/0` containing `Bash`, the decomposed `ls`, and `ls -la /tmp`. A `Read` call with no hint stores one row containing its name and path and no shell tokens. A payload past the budget splits, and every piece begins with the tool name. The derived-work classifier wants no work when the row is present.

Existing tests asserting `/tok`, `/cmd`, or `/in` paths need updating; those paths no longer exist.

**Done when:** `make fmt`, `make build`, `make test`, and `make live` all return zero.

---

## Task 7: engine: prove it against a real store

`test/live/tool_row_live_test.go`, following `test/live/blank_row_live_test.go` for the harness, which boots an isolated daemon and a throwaway Milvus collection that cannot collide with the operator's.

Ingest one message carrying a `Bash` call and a `Read` call. Assert exactly one row under each of `convtool/<id>/0/0` and `convtool/<id>/0/1`, zero rows containing `file_path`, `"command"`, `{"`, or `input_text`, and zero rows holding nothing.

Add `countRowsContaining` beside the existing `countRowsHoldingNothing`.

**Then restore the `Display` append inside `conversationToolContent` and re-run.** The test must fail. A live test that passes on both the fixed and the broken tree proves nothing.

---

## Task 8: Land it and measure

Merge the engine first: clyde builds against the local engine checkout through `go.work`.

Deploy both. Record the baseline with its filters: rows under `relativePath != ''`, `content == ''`, `convtool/%/tok`, `convtool/%/in%`, `convtool/%/cmd`.

Set `enabled = true` under `[conversation.semantic]`. **Read the file first and edit that one line.** It holds several other `enabled` keys and a blind replace hits the wrong one.

**Proven when:** a conversation ingested after the change stores one row per tool call, no row written after the change contains `file_path`, `"command"`, or `input_text`, the tool-row total falls as conversations are re-ingested, and the blank-row count stays flat.

**Recall must hold.** Before turning writes on, run a set of searches a person would actually type and record which conversations each one returns. Run the same set again after conversations have been re-ingested under the new shape. A conversation that the old corpus returned and the new one does not is a regression, and it blocks the change regardless of how much smaller the corpus got. Cover all four ways a tool call is reached: the tool's name, a program name inside a command, a file path, and the command text itself.

---

## Notes

**Export had the same defect.** The spec covers only search, but `toolFullDetailText` dumps the same raw JSON at a human reading an export. That is why `Display` lives on `transcript.ToolCall` rather than only on the search projection: one renderer, two consumers.

**The key lists are not exhaustive.** They come from the tool shapes visible in the current corpus. A tool whose shape is missing stores its name alone rather than wrong text, so a gap reads as a thin row rather than as silent JSON. Task 2's fixture test asserts a coverage floor rather than every call for exactly this reason.

**The provider names the language, not a generic layer.** A shell call's display text is its command, and only the provider's parser knows which of its harness's tools run shells. `DisplayLang` carries that answer forward so `internal/daemon` and the engine hold no harness tool names.
