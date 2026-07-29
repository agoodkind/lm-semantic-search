# Tool calls carry what the user saw: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A stored conversation tool call holds the text the user saw, in one row, and the harness's serialization never leaves clyde.

**Architecture:** Each provider parser gains a `Display` string on `transcript.ToolCall`, filled from that harness's own tool shapes because only the provider knows them. Export renders `Display` instead of dumping JSON. The semantic feeder sends `Display` and stops sending `input_json`, which is removed from the wire. The engine stores one row per tool call instead of three.

**Tech Stack:** Go, protobuf over gRPC, Milvus.

## Global Constraints

Two repositories: clyde at `/Users/agoodkind/Sites/clyde-dev/clyde`, engine at `/Users/agoodkind/Sites/lm-semantic-search`.

Work in a linked worktree. agent-gate blocks writes to a primary checkout and to a default branch.

A nested clyde worktree needs its own untracked `go.work` containing `use .`, or Go resolves to the primary checkout and reports `ok ... [no tests to run]`, which reads like success.

clyde's `make check` runs lint only. Run `make test` separately.

Every engine `make` needs `GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile`. A targeted `go test` in the engine needs `PKG_CONFIG_PATH=<worktree>/.make/cgo/darwin-arm64/lib/pkgconfig` and `CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path'` exported.

No code that deletes stored rows, in any form. Existing rows are removed by hand later.

No backfill. Old and new rows coexist.

Never weaken, skip, or baseline a failing `make` step.

Strict types. No `any`, no `map[string]any`, no empty marker structs.

`exhaustruct` is enforced, so every struct literal names every field.

Commit with `git commit -S`.

Conversation embedding is currently off (`[conversation.semantic] enabled = false` in `~/.config/clyde/config.toml`). Leave it off until Task 9.

---

## File structure

**clyde**

| file | responsibility after this plan |
| --- | --- |
| `internal/transcript/transcript.go` | `ToolCall` gains `Display string`, the text the user saw |
| `internal/transcript/conversation.go` | `toolFullDetailText` renders `Display`, never raw JSON |
| `internal/providers/claude/parser/entry.go` | fills `Display` for Claude tool shapes |
| `internal/providers/codex/store/messages.go` | fills `Display` for Codex tool shapes |
| `internal/providers/cursor/parser/mapping.go` | fills `Display` for Cursor tool shapes |
| `internal/providers/zed/parser/parser.go` | fills `Display` for Zed tool shapes |
| `internal/conversation/semsearch/client.go` | `SemToolCall` drops `InputJSON`, gains `Display` |
| `internal/daemon/conversation_semantic_sync.go` | sends `Display`; `deriveToolCommandAndLang` deleted |

**engine**

| file | responsibility after this plan |
| --- | --- |
| `proto/lmsemanticsearch/v1/service.proto` | `ConversationToolCall` drops `input_json`, gains `display` |
| `internal/model/*` | the Go tool-call model follows the wire |
| `internal/daemon/manager_conversation_tools.go` | one row per call; the `/tok`, `/cmd`, `/in` producers collapse |
| `internal/daemon/manager_conversations.go` | emits one chunk per tool call |
| `internal/daemon/item_source.go` | the derived-work classifier expects one path per call |

---

## Task 1: `ToolCall` carries what the user saw

**Files:**
- Modify: `internal/transcript/transcript.go:120-126`
- Test: `internal/transcript/transcript_display_test.go` (create)

**Interfaces:**
- Produces: `transcript.ToolCall.Display string`, the text the user saw for this call. Empty means the parser has not filled it.

- [ ] **Step 1: Write the failing test**

```go
package transcript

import "testing"

// TestToolCallCarriesDisplayText pins the field every parser fills. The stored
// and exported forms both read it, so a parser that leaves it empty is visible
// here rather than as an empty row much later.
func TestToolCallCarriesDisplayText(t *testing.T) {
	t.Parallel()

	call := ToolCall{
		ID:      "call_1",
		Name:    "Read",
		Input:   ToolInputJSON{Raw: []byte(`{"file_path":"/tmp/phone.png"}`)},
		Display: "/tmp/phone.png",
		Output:  "",
		IsError: false,
	}
	if call.Display != "/tmp/phone.png" {
		t.Fatalf("Display = %q, want the path the user saw", call.Display)
	}
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `go test ./internal/transcript/ -run TestToolCallCarriesDisplayText`
Expected: FAIL, `unknown field Display in struct literal`

- [ ] **Step 3: Add the field**

In `internal/transcript/transcript.go`, replace the `ToolCall` struct:

```go
// ToolCall represents a single tool invocation within an assistant message.
type ToolCall struct {
	ID    string        `json:"id"`    // tool_use_id (links to tool_result in next user message)
	Name  string        `json:"name"`  // e.g. "Bash", "Edit", "Read"
	Input ToolInputJSON `json:"input"` // opaque tool input payload, preserved verbatim
	// Display is what the user saw for this call: a shell command, a file path,
	// a search pattern. Each provider's parser fills it from its own harness's
	// tool shapes, because only that parser knows them. Everything that shows or
	// stores a tool call reads this rather than re-deriving it from Input, so the
	// harness's serialization stays inside the provider package.
	Display string `json:"display,omitempty"`
	Output  string `json:"output"`   // tool result text (loaded on demand, empty by default)
	IsError bool   `json:"is_error"` // true if tool result was an error
}
```

- [ ] **Step 4: Run it and watch it pass**

Run: `go test ./internal/transcript/ -run TestToolCallCarriesDisplayText`
Expected: PASS

- [ ] **Step 5: Fix every struct literal the compiler now rejects**

`exhaustruct` requires every field. Build to find them:

Run: `go build ./...`

Add `Display: "",` to each `transcript.ToolCall{...}` literal the compiler names. Leave the value empty; later tasks fill it per provider.

- [ ] **Step 6: Confirm the tree builds and tests pass**

Run: `go build ./... && go test ./internal/transcript/`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add internal/transcript/ internal/providers/ internal/
git commit -S -m "Carry what the user saw on a tool call"
```

---

## Task 2: Claude's parser fills the display text

**Files:**
- Modify: `internal/providers/claude/parser/entry.go`
- Create: `internal/providers/claude/parser/tool_display.go`
- Test: `internal/providers/claude/parser/tool_display_test.go`

**Interfaces:**
- Consumes: `transcript.ToolCall.Display` from Task 1.
- Produces: `toolDisplayText(name string, input transcript.ToolInputJSON) string` in package `parser`.

- [ ] **Step 1: Write the failing test**

```go
package parser

import (
	"testing"

	"goodkind.io/clyde/internal/transcript"
)

// TestToolDisplayTextIsWhatTheUserSaw pins the rule for Claude's tool shapes.
// A shell call shows its command, a file tool shows its path, a search shows its
// pattern. Nothing shows the JSON wrapper, because the user never saw it.
func TestToolDisplayTextIsWhatTheUserSaw(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		tool  string
		input string
		want  string
	}{
		{"shell shows its command", "Bash", `{"command":"ls -la /tmp","description":"list"}`, "ls -la /tmp"},
		{"read shows its path", "Read", `{"file_path":"/tmp/phone.png"}`, "/tmp/phone.png"},
		{"write shows its path", "Write", `{"file_path":"/tmp/a.go","content":"package main"}`, "/tmp/a.go"},
		{"edit shows its path", "Edit", `{"file_path":"/tmp/a.go","old_string":"x","new_string":"y"}`, "/tmp/a.go"},
		{"glob shows its pattern", "Glob", `{"pattern":"**/*.go"}`, "**/*.go"},
		{"grep shows its pattern", "Grep", `{"pattern":"func main","path":"/tmp"}`, "func main"},
		{"a tool with no known shape shows nothing", "Mystery", `{"whatever":1}`, ""},
		{"an empty input shows nothing", "Read", ``, ""},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			input := transcript.ToolInputJSON{Raw: []byte(testCase.input)}
			got := toolDisplayText(testCase.tool, input)
			if got != testCase.want {
				t.Fatalf("toolDisplayText(%q) = %q, want %q", testCase.tool, got, testCase.want)
			}
		})
	}
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `go test ./internal/providers/claude/parser/ -run TestToolDisplayTextIsWhatTheUserSaw`
Expected: FAIL, `undefined: toolDisplayText`

- [ ] **Step 3: Write the renderer**

Create `internal/providers/claude/parser/tool_display.go`:

```go
package parser

import (
	"encoding/json"
	"strings"

	"goodkind.io/clyde/internal/transcript"
)

// claudeToolInput names every field of a Claude tool input this parser reads.
// The shapes belong here rather than in a shared layer, because they are
// Claude's and no other package should have to know them.
type claudeToolInput struct {
	Command  string `json:"command"`
	FilePath string `json:"file_path"`
	Pattern  string `json:"pattern"`
	Prompt   string `json:"prompt"`
	URL      string `json:"url"`
	Query    string `json:"query"`
}

// toolDisplayText is what the user saw for one tool call: the command a shell
// ran, the path a file tool touched, the pattern a search used.
//
// A tool whose shape this parser does not recognize shows nothing rather than
// its serialization. An unrecognized tool is a gap to fill here, and showing the
// JSON instead would hide the gap behind text nobody wrote.
func toolDisplayText(name string, input transcript.ToolInputJSON) string {
	if input.Len() == 0 {
		return ""
	}
	var parsed claudeToolInput
	if err := json.Unmarshal(input.Raw, &parsed); err != nil {
		return ""
	}
	for _, candidate := range []string{
		parsed.Command,
		parsed.FilePath,
		parsed.Pattern,
		parsed.Prompt,
		parsed.URL,
		parsed.Query,
	} {
		if trimmed := strings.TrimSpace(candidate); trimmed != "" {
			return trimmed
		}
	}
	_ = name
	return ""
}
```

- [ ] **Step 4: Run it and watch it pass**

Run: `go test ./internal/providers/claude/parser/ -run TestToolDisplayTextIsWhatTheUserSaw`
Expected: PASS

- [ ] **Step 5: Fill `Display` where the parser builds a tool call**

In `internal/providers/claude/parser/entry.go`, find each `transcript.ToolCall{` literal and set:

```go
Display: toolDisplayText(name, input),
```

using that literal's own name and input values.

- [ ] **Step 6: Prove a parsed transcript carries it**

Add to `tool_display_test.go`:

```go
// TestParsedToolCallsCarryDisplayText proves the renderer is wired into the
// parser rather than only existing beside it.
func TestParsedToolCallsCarryDisplayText(t *testing.T) {
	t.Parallel()

	messages := parseFixtureWithTools(t)
	saw := 0
	for _, message := range messages {
		for _, tool := range message.Tools {
			if tool.Input.Len() == 0 {
				continue
			}
			saw++
			if tool.Display == "" {
				t.Fatalf("tool %q parsed with an input but no display text", tool.Name)
			}
		}
	}
	if saw == 0 {
		t.Fatal("the fixture carried no tool call with an input, so this proves nothing")
	}
}
```

Write `parseFixtureWithTools` against an existing Claude parser fixture in this package. Reuse whatever fixture helper the package's other tests already use rather than adding a second one.

- [ ] **Step 7: Run it and watch it pass**

Run: `go test ./internal/providers/claude/parser/`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add internal/providers/claude/parser/
git commit -S -m "Fill the display text from Claude's tool shapes"
```

---

## Task 3: Codex's parser fills the display text

**Files:**
- Modify: `internal/providers/codex/store/messages.go:130-146`
- Create: `internal/providers/codex/store/tool_display.go`
- Test: `internal/providers/codex/store/tool_display_test.go`

**Interfaces:**
- Consumes: `transcript.ToolCall.Display` from Task 1.
- Produces: `toolDisplayText(name string, input transcript.ToolInputJSON) string` in package `store`.

- [ ] **Step 1: Write the failing test**

```go
package store

import (
	"testing"

	"goodkind.io/clyde/internal/transcript"
)

// TestCodexToolDisplayTextIsWhatTheUserSaw pins the rule for Codex's tool
// shapes, which differ from Claude's: a shell call carries an argv array rather
// than a command string, and a custom tool carries a raw patch encoded as a
// JSON string.
func TestCodexToolDisplayTextIsWhatTheUserSaw(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		tool  string
		input string
		want  string
	}{
		{"shell shows its joined argv", "shell", `{"command":["bash","-lc","ls /tmp"]}`, "bash -lc ls /tmp"},
		{"a command string shows itself", "shell", `{"command":"ls /tmp"}`, "ls /tmp"},
		{"a raw patch shows itself", "apply_patch", `"*** Begin Patch\n*** End Patch"`, "*** Begin Patch\n*** End Patch"},
		{"an empty input shows nothing", "shell", ``, ""},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			input := transcript.ToolInputJSON{Raw: []byte(testCase.input)}
			got := toolDisplayText(testCase.tool, input)
			if got != testCase.want {
				t.Fatalf("toolDisplayText(%q) = %q, want %q", testCase.tool, got, testCase.want)
			}
		})
	}
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `go test ./internal/providers/codex/store/ -run TestCodexToolDisplayTextIsWhatTheUserSaw`
Expected: FAIL, `undefined: toolDisplayText`

- [ ] **Step 3: Write the renderer**

Create `internal/providers/codex/store/tool_display.go`:

```go
package store

import (
	"encoding/json"
	"strings"

	"goodkind.io/clyde/internal/transcript"
)

// codexToolInput names every field of a Codex tool input this parser reads.
// Codex writes a shell call's command as an argv array, which is why this shape
// differs from the one Claude's parser reads.
type codexToolInput struct {
	Command json.RawMessage `json:"command"`
	Path    string          `json:"path"`
	Pattern string          `json:"pattern"`
	Query   string          `json:"query"`
}

// toolDisplayText is what the user saw for one Codex tool call.
//
// toolInputJSON encodes a non-JSON payload, such as a patch, as a JSON string,
// so a bare string input is that payload and is returned as itself.
func toolDisplayText(name string, input transcript.ToolInputJSON) string {
	if input.Len() == 0 {
		return ""
	}
	var rawString string
	if json.Unmarshal(input.Raw, &rawString) == nil {
		return strings.TrimSpace(rawString)
	}
	var parsed codexToolInput
	if err := json.Unmarshal(input.Raw, &parsed); err != nil {
		return ""
	}
	if command := codexCommandText(parsed.Command); command != "" {
		return command
	}
	for _, candidate := range []string{parsed.Path, parsed.Pattern, parsed.Query} {
		if trimmed := strings.TrimSpace(candidate); trimmed != "" {
			return trimmed
		}
	}
	_ = name
	return ""
}

// codexCommandText renders a shell command that arrives either as an argv array
// or as a single string.
func codexCommandText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var argv []string
	if json.Unmarshal(raw, &argv) == nil {
		return strings.TrimSpace(strings.Join(argv, " "))
	}
	var command string
	if json.Unmarshal(raw, &command) == nil {
		return strings.TrimSpace(command)
	}
	return ""
}
```

- [ ] **Step 4: Run it and watch it pass**

Run: `go test ./internal/providers/codex/store/ -run TestCodexToolDisplayTextIsWhatTheUserSaw`
Expected: PASS

- [ ] **Step 5: Fill `Display` in `toolCallHistoryMessage`**

In `internal/providers/codex/store/messages.go`, inside `toolCallHistoryMessage`, replace the tool literal:

```go
	input := toolInputJSON(raw)
	message.Tools = []transcript.ToolCall{{
		ID:      callID,
		Name:    name,
		Input:   input,
		Display: toolDisplayText(name, input),
		Output:  "",
		IsError: false,
	}}
```

- [ ] **Step 6: Run the package tests**

Run: `go test ./internal/providers/codex/...`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add internal/providers/codex/
git commit -S -m "Fill the display text from Codex's tool shapes"
```

---

## Task 4: Cursor's and Zed's parsers fill the display text

**Files:**
- Modify: `internal/providers/cursor/parser/mapping.go:35-42`, `:76-83`
- Create: `internal/providers/cursor/parser/tool_display.go`
- Test: `internal/providers/cursor/parser/tool_display_test.go`
- Modify: `internal/providers/zed/parser/parser.go:659-666`
- Create: `internal/providers/zed/parser/tool_display.go`
- Test: `internal/providers/zed/parser/tool_display_test.go`

**Interfaces:**
- Consumes: `transcript.ToolCall.Display` from Task 1.
- Produces: `toolDisplayText(name string, input transcript.ToolInputJSON) string` in packages `cursor/parser` and `zed/parser`.

- [ ] **Step 1: Write the failing Cursor test**

```go
package parser

import (
	"testing"

	"goodkind.io/clyde/internal/transcript"
)

// TestCursorToolDisplayTextIsWhatTheUserSaw pins the rule for Cursor's tool
// shapes. Cursor names a file tool's path with relative_workspace_path rather
// than file_path, which is why this parser reads its own shape.
func TestCursorToolDisplayTextIsWhatTheUserSaw(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		tool  string
		input string
		want  string
	}{
		{"a terminal call shows its command", "run_terminal_cmd", `{"command":"go test ./..."}`, "go test ./..."},
		{"a read shows its workspace path", "read_file", `{"relative_workspace_path":"internal/a.go"}`, "internal/a.go"},
		{"a read shows an absolute path", "read_file", `{"target_file":"/tmp/a.go"}`, "/tmp/a.go"},
		{"a search shows its query", "codebase_search", `{"query":"where is auth"}`, "where is auth"},
		{"an empty input shows nothing", "read_file", ``, ""},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			input := transcript.ToolInputJSON{Raw: []byte(testCase.input)}
			got := toolDisplayText(testCase.tool, input)
			if got != testCase.want {
				t.Fatalf("toolDisplayText(%q) = %q, want %q", testCase.tool, got, testCase.want)
			}
		})
	}
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `go test ./internal/providers/cursor/parser/ -run TestCursorToolDisplayTextIsWhatTheUserSaw`
Expected: FAIL, `undefined: toolDisplayText`

- [ ] **Step 3: Write the Cursor renderer**

Create `internal/providers/cursor/parser/tool_display.go`:

```go
package parser

import (
	"encoding/json"
	"strings"

	"goodkind.io/clyde/internal/transcript"
)

// cursorToolInput names every field of a Cursor tool input this parser reads.
// Cursor names a file tool's path relative_workspace_path or target_file, which
// is why these shapes live here rather than in a shared layer.
type cursorToolInput struct {
	Command               string `json:"command"`
	RelativeWorkspacePath string `json:"relative_workspace_path"`
	TargetFile            string `json:"target_file"`
	Query                 string `json:"query"`
	Pattern               string `json:"pattern"`
}

// toolDisplayText is what the user saw for one Cursor tool call.
func toolDisplayText(name string, input transcript.ToolInputJSON) string {
	if input.Len() == 0 {
		return ""
	}
	var parsed cursorToolInput
	if err := json.Unmarshal(input.Raw, &parsed); err != nil {
		return ""
	}
	for _, candidate := range []string{
		parsed.Command,
		parsed.RelativeWorkspacePath,
		parsed.TargetFile,
		parsed.Query,
		parsed.Pattern,
	} {
		if trimmed := strings.TrimSpace(candidate); trimmed != "" {
			return trimmed
		}
	}
	_ = name
	return ""
}
```

- [ ] **Step 4: Fill `Display` in both Cursor literals**

In `mapJSONLMessage`:

```go
		case cursorjsonl.PartTypeToolUse:
			input := transcript.ToolInputJSON{Raw: cloneRaw(part.ToolInput)}
			tools = append(tools, transcript.ToolCall{
				ID:      "",
				Name:    part.ToolName,
				Input:   input,
				Display: toolDisplayText(part.ToolName, input),
				Output:  "",
				IsError: false,
			})
```

In `mapComposerBubble`:

```go
		input := transcript.ToolInputJSON{Raw: rawArgsToJSON(bubble.ToolCall.RawArgs)}
		tools = append(tools, transcript.ToolCall{
			ID:      "",
			Name:    bubble.ToolCall.Name,
			Input:   input,
			Display: toolDisplayText(bubble.ToolCall.Name, input),
			Output:  toolOutput,
			IsError: toolErrored(bubble.ToolCall.Status),
		})
```

- [ ] **Step 5: Run the Cursor tests**

Run: `go test ./internal/providers/cursor/...`
Expected: PASS

- [ ] **Step 6: Write the failing Zed test**

```go
package parser

import (
	"testing"

	"goodkind.io/clyde/internal/transcript"
)

// TestZedToolDisplayTextIsWhatTheUserSaw pins the rule for Zed's tool shapes.
func TestZedToolDisplayTextIsWhatTheUserSaw(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		tool  string
		input string
		want  string
	}{
		{"a terminal call shows its command", "terminal", `{"command":"cargo build"}`, "cargo build"},
		{"a read shows its path", "read_file", `{"path":"src/main.rs"}`, "src/main.rs"},
		{"a search shows its regex", "grep", `{"regex":"fn main"}`, "fn main"},
		{"an empty input shows nothing", "read_file", ``, ""},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			input := transcript.ToolInputJSON{Raw: []byte(testCase.input)}
			got := toolDisplayText(testCase.tool, input)
			if got != testCase.want {
				t.Fatalf("toolDisplayText(%q) = %q, want %q", testCase.tool, got, testCase.want)
			}
		})
	}
}
```

- [ ] **Step 7: Write the Zed renderer**

Create `internal/providers/zed/parser/tool_display.go`:

```go
package parser

import (
	"encoding/json"
	"strings"

	"goodkind.io/clyde/internal/transcript"
)

// zedToolInput names every field of a Zed tool input this parser reads.
type zedToolInput struct {
	Command string `json:"command"`
	Path    string `json:"path"`
	Regex   string `json:"regex"`
	Query   string `json:"query"`
}

// toolDisplayText is what the user saw for one Zed tool call.
func toolDisplayText(name string, input transcript.ToolInputJSON) string {
	if input.Len() == 0 {
		return ""
	}
	var parsed zedToolInput
	if err := json.Unmarshal(input.Raw, &parsed); err != nil {
		return ""
	}
	for _, candidate := range []string{parsed.Command, parsed.Path, parsed.Regex, parsed.Query} {
		if trimmed := strings.TrimSpace(candidate); trimmed != "" {
			return trimmed
		}
	}
	_ = name
	return ""
}
```

- [ ] **Step 8: Fill `Display` in the Zed literal**

In `agentMessageParts`:

```go
			input := transcript.ToolInputJSON{Raw: append([]byte(nil), part.ToolUse.Input...)}
			tools = append(tools, transcript.ToolCall{
				ID:      part.ToolUse.ID,
				Name:    part.ToolUse.Name,
				Input:   input,
				Display: toolDisplayText(part.ToolUse.Name, input),
				Output:  output,
				IsError: isError,
			})
```

- [ ] **Step 9: Run both packages**

Run: `go test ./internal/providers/cursor/... ./internal/providers/zed/...`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add internal/providers/cursor/ internal/providers/zed/
git commit -S -m "Fill the display text from Cursor's and Zed's tool shapes"
```

---

## Task 5: Export shows the display text instead of JSON

**Files:**
- Modify: `internal/transcript/conversation.go:204-226`
- Test: `internal/transcript/conversation_display_test.go` (create)

**Interfaces:**
- Consumes: `transcript.ToolCall.Display` from Tasks 1 through 4.

- [ ] **Step 1: Write the failing test**

```go
package transcript

import (
	"strings"
	"testing"
)

// TestFullDetailShowsWhatTheUserSaw proves the export path renders the display
// text rather than the harness's serialization. An exported transcript is read
// by a person, and a JSON wrapper is not something that person ever saw.
func TestFullDetailShowsWhatTheUserSaw(t *testing.T) {
	t.Parallel()

	tools := []ToolCall{{
		ID:      "call_1",
		Name:    "Read",
		Input:   ToolInputJSON{Raw: []byte(`{"file_path":"/tmp/phone.png"}`)},
		Display: "/tmp/phone.png",
		Output:  "",
		IsError: false,
	}}

	rendered := toolFullDetailText(tools)
	if !strings.Contains(rendered, "/tmp/phone.png") {
		t.Fatalf("rendered = %q, want the path the user saw", rendered)
	}
	if strings.Contains(rendered, "file_path") {
		t.Fatalf("rendered = %q, want no serialization", rendered)
	}
}

// TestFullDetailFallsBackToTheToolName covers a call whose parser filled no
// display text. Showing the name alone is honest; showing the JSON would put
// text on screen that nobody wrote.
func TestFullDetailFallsBackToTheToolName(t *testing.T) {
	t.Parallel()

	tools := []ToolCall{{
		ID:      "call_1",
		Name:    "Mystery",
		Input:   ToolInputJSON{Raw: []byte(`{"whatever":1}`)},
		Display: "",
		Output:  "",
		IsError: false,
	}}

	rendered := toolFullDetailText(tools)
	if rendered != "[tool: Mystery]" {
		t.Fatalf("rendered = %q, want the tool name alone", rendered)
	}
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `go test ./internal/transcript/ -run TestFullDetail`
Expected: FAIL, the rendered text contains `file_path`

- [ ] **Step 3: Render the display text**

Replace `toolFullDetailText` in `internal/transcript/conversation.go`:

```go
func toolFullDetailText(tools []ToolCall) string {
	if len(tools) == 0 {
		return "[used tools]"
	}
	lines := make([]string, 0, len(tools))
	for _, tool := range tools {
		line := "[tool: " + tool.Name + "]"
		// The display text is what the user saw. A call whose parser filled none
		// shows its name alone rather than its serialization, which the user
		// never saw and which no reader of an export wants.
		if display := strings.TrimSpace(tool.Display); display != "" {
			line += " " + display
		}
		if strings.TrimSpace(tool.Output) != "" {
			line += "\n[tool output]\n" + strings.TrimSpace(tool.Output)
		}
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
}
```

- [ ] **Step 4: Run it and watch it pass**

Run: `go test ./internal/transcript/`
Expected: PASS

- [ ] **Step 5: Remove the now-unused JSON import if the compiler says so**

Run: `go build ./internal/transcript/`

If `encoding/json` is now unused in `conversation.go`, remove it. `toolInputDescription` may still use it; check before removing.

- [ ] **Step 6: Run the export tests**

Run: `go test ./internal/conversation/... ./internal/transcript/...`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add internal/transcript/ internal/conversation/
git commit -S -m "Show a tool call's display text in an export rather than its serialization"
```

---

## Task 6: The wire carries the display text and no serialization

**Files:**
- Modify (engine): `proto/lmsemanticsearch/v1/service.proto:296-314`
- Regenerate (engine): `gen/go/lmsemanticsearch/v1/`
- Modify (engine): the Go tool-call model that mirrors the wire
- Test (engine): `internal/daemon/conversation_tool_display_test.go` (create)

**Interfaces:**
- Produces: `ConversationToolCall.display` on the wire; `input_json` is gone.

- [ ] **Step 1: Change the proto**

In `proto/lmsemanticsearch/v1/service.proto`, replace the `ConversationToolCall` message:

```proto
// ConversationToolCall is one structured tool call attached to a conversation
// document. clyde sends what the user saw for the call, because only clyde
// knows each harness's tool shapes. The harness's serialization never reaches
// the engine.
message ConversationToolCall {
  // name is the tool name, for example "Bash" or "run_command".
  string name = 1;
  // display is what the user saw for this call: the command a shell ran, the
  // path a file tool touched, the pattern a search used. Empty when clyde's
  // parser does not recognize the tool's shape.
  string display = 2;
  // command is the shell command string when this tool ran a shell, extracted by
  // clyde from the provider-specific input key. Empty for non-shell tools. The
  // engine decomposes it into program names and file paths, which is knowledge
  // of shell as a language rather than of any harness.
  string command = 3;
  // lang_hint names the payload language for chunking, for example "bash",
  // "json", or "markdown". Empty when unknown.
  string lang_hint = 4;
  // output is the tool result text when captured. Can be large and is sensitive
  // in the same way tool inputs can be.
  string output = 5;
  // is_error marks a tool call that returned an error result.
  bool is_error = 6;
}
```

Field 2 is reused rather than reserved because no stored data carries the old meaning: the engine reads the wire at ingest and never replays it.

- [ ] **Step 2: Regenerate**

Run: `make -C <engine worktree> proto GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile`

If the target has another name, find it in the Makefile rather than guessing.

- [ ] **Step 3: Follow the model through the compiler**

Run: `go build ./...`

Rename `InputJSON` to `Display` on the engine's tool-call model and fix each site the compiler names.

- [ ] **Step 4: Run the engine's daemon tests**

Run: `go test ./internal/daemon/`
Expected: compile errors in tests that set `InputJSON`. Update them to set `Display`.

- [ ] **Step 5: Commit in the engine**

```bash
git add proto/ gen/ internal/
git commit -S -m "Carry what the user saw on the conversation tool call wire"
```

---

## Task 7: clyde sends the display text

**Files:**
- Modify: `internal/conversation/semsearch/client.go:57-64`
- Modify: `internal/daemon/conversation_semantic_sync.go:753-806`
- Test: `internal/daemon/conversation_semantic_tool_display_test.go` (create)

**Interfaces:**
- Consumes: `transcript.ToolCall.Display`, and the engine wire from Task 6.
- Produces: `semsearch.SemToolCall.Display`; `SemToolCall.InputJSON` and `deriveToolCommandAndLang` are gone.

- [ ] **Step 1: Write the failing test**

```go
package daemon

import (
	"strings"
	"testing"

	"goodkind.io/clyde/internal/conversation"
	"goodkind.io/clyde/internal/transcript"
)

// TestSentToolCallsCarryNoSerialization is the rule this change exists for.
// The harness's serialization of a tool call is text the user never saw, and
// sending it both wastes an embedding and fills the corpus with JSON fragments
// nobody wrote.
func TestSentToolCallsCarryNoSerialization(t *testing.T) {
	t.Parallel()

	tools := []transcript.ToolCall{{
		ID:      "call_1",
		Name:    "Read",
		Input:   transcript.ToolInputJSON{Raw: []byte(`{"file_path":"/tmp/phone.png"}`)},
		Display: "/tmp/phone.png",
		Output:  "",
		IsError: false,
	}}

	kinds := conversation.NewContentKindSet(conversation.ContentKindToolCalls)
	sent := semanticToolCalls(tools, kinds)
	if len(sent) != 1 {
		t.Fatalf("sent %d tool calls, want 1", len(sent))
	}
	if sent[0].Display != "/tmp/phone.png" {
		t.Fatalf("Display = %q, want the path the user saw", sent[0].Display)
	}
	for _, field := range []string{sent[0].Display, sent[0].Command} {
		if strings.Contains(field, "file_path") {
			t.Fatalf("a sent field carries serialization: %q", field)
		}
	}
}

// TestAShellCallStillSendsItsCommand pins that the engine keeps what it needs to
// decompose a command into program names and file paths.
func TestAShellCallStillSendsItsCommand(t *testing.T) {
	t.Parallel()

	tools := []transcript.ToolCall{{
		ID:      "call_1",
		Name:    "Bash",
		Input:   transcript.ToolInputJSON{Raw: []byte(`{"command":"ls -la /tmp"}`)},
		Display: "ls -la /tmp",
		Output:  "",
		IsError: false,
	}}

	kinds := conversation.NewContentKindSet(conversation.ContentKindToolCalls)
	sent := semanticToolCalls(tools, kinds)
	if sent[0].Command != "ls -la /tmp" {
		t.Fatalf("Command = %q, want the command the shell ran", sent[0].Command)
	}
	if sent[0].LangHint != "bash" {
		t.Fatalf("LangHint = %q, want bash", sent[0].LangHint)
	}
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `go test ./internal/daemon/ -run TestSentToolCalls`
Expected: FAIL, `sent[0].Display undefined`

- [ ] **Step 3: Change the projection type**

In `internal/conversation/semsearch/client.go`, replace `SemToolCall`:

```go
// SemToolCall is one structured tool call attached to a semantic document. It
// carries what the user saw rather than the harness's serialization of the call,
// because only the text a person saw is worth retrieving.
type SemToolCall struct {
	Name string
	// Display is what the user saw for this call, filled by the provider's parser.
	Display  string
	Command  string
	LangHint string
	Output   string
	IsError  bool
}
```

- [ ] **Step 4: Send the display text and delete the guesser**

In `internal/daemon/conversation_semantic_sync.go`, replace `semanticToolCalls` and delete `deriveToolCommandAndLang` and `semanticToolCommandInput` entirely:

```go
// semanticToolCalls projects a message's tool calls at the selected detail level.
//
// The three tool kinds are nested rather than parallel, and
// [conversation.NewContentKindSet] collapses them, so exactly one applies: the
// summary level carries the tool's name alone, the call level adds what the user
// saw, and the output level adds what the tool returned. Selecting no tool kind
// drops the calls entirely.
//
// The harness's serialization is never sent. Each provider's parser already
// rendered what the user saw into Display, so re-deriving it here would put
// knowledge of every harness's tool shapes into a layer that must not hold it.
func semanticToolCalls(tools []transcript.ToolCall, kinds conversation.ContentKindSet) []semsearch.SemToolCall {
	summariesOnly := kinds.Has(conversation.ContentKindToolSummaries)
	withArguments := kinds.Has(conversation.ContentKindToolCalls)
	withOutput := kinds.Has(conversation.ContentKindToolOutputs)
	if !summariesOnly && !withArguments && !withOutput {
		return nil
	}
	out := make([]semsearch.SemToolCall, 0, len(tools))
	for _, tool := range tools {
		projected := semsearch.SemToolCall{
			Name:     tool.Name,
			Display:  "",
			Command:  "",
			LangHint: "",
			Output:   "",
			IsError:  tool.IsError,
		}
		if withArguments || withOutput {
			projected.Display = strings.ToValidUTF8(tool.Display, "")
			projected.Command, projected.LangHint = semanticToolCommandAndLang(tool)
		}
		if withOutput {
			projected.Output = tool.Output
		}
		out = append(out, projected)
	}
	return out
}

// semanticToolCommandAndLang reports the shell command a tool ran and the
// language its payload is written in.
//
// A shell tool's display text is the command, because that is what the user saw,
// so the command needs no second derivation. The language hint tells the engine
// how to split the payload, which is knowledge of formats rather than of any
// harness.
func semanticToolCommandAndLang(tool transcript.ToolCall) (string, string) {
	display := strings.TrimSpace(tool.Display)
	if display == "" {
		return "", ""
	}
	if isShellToolName(tool.Name) {
		return display, "bash"
	}
	return "", ""
}

// isShellToolName reports whether a tool name is a shell across the harnesses
// clyde reads. The names are few, stable, and shared, so listing them here is
// smaller than routing a flag through every parser.
func isShellToolName(name string) bool {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "bash", "shell", "run_terminal_cmd", "terminal", "local_shell":
		return true
	default:
		return false
	}
}
```

- [ ] **Step 5: Follow the client through the compiler**

Run: `go build ./...`

Update the semsearch client's conversion into the generated protobuf tool call: set `Display` and drop `InputJson`.

- [ ] **Step 6: Run it and watch it pass**

Run: `go test ./internal/daemon/ -run 'TestSentToolCalls|TestAShellCall'`
Expected: PASS

- [ ] **Step 7: Run the whole clyde suite**

Run: `make test`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add internal/
git commit -S -m "Send what the user saw for a tool call and delete the command guesser"
```

---

## Task 8: The engine stores one row per tool call

**Files:**
- Modify (engine): `internal/daemon/manager_conversation_tools.go:99-110`
- Modify (engine): `internal/daemon/manager_conversations.go:645-674`
- Modify (engine): `internal/daemon/item_source.go:451-467`
- Test (engine): `internal/daemon/conversation_one_row_per_tool_test.go` (create)

**Interfaces:**
- Consumes: the wire from Task 6.
- Produces: one stored row per tool call at `convtool/<conversation>/<message>/<toolIndex>`.

- [ ] **Step 1: Write the failing test**

```go
package daemon

import (
	"context"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

// TestOneRowPerToolCall is the rule this change exists for. A tool call used to
// store three rows, and two of them carried the same text: the summary row ended
// with a verbatim copy of the arguments row. Measured at 93% literal containment
// across 364 calls in one conversation.
func TestOneRowPerToolCall(t *testing.T) {
	t.Parallel()

	documents := []model.ConversationDocument{{
		ConversationID: "claude:a",
		MessageIndex:   0,
		Role:           "assistant",
		Tools: []model.ConversationToolCall{{
			Name:     "Bash",
			Display:  "ls -la /tmp",
			Command:  "ls -la /tmp",
			LangHint: "bash",
			Output:   "",
			IsError:  false,
		}},
	}}

	chunks, err := conversationDocumentsToStoredChunks(context.Background(), documents)
	if err != nil {
		t.Fatalf("conversationDocumentsToStoredChunks returned error: %v", err)
	}

	toolRows := make([]model.StoredChunk, 0)
	for _, chunk := range chunks {
		if strings.HasPrefix(chunk.RelativePath, "convtool/") {
			toolRows = append(toolRows, chunk)
		}
	}
	if len(toolRows) != 1 {
		paths := make([]string, 0, len(toolRows))
		for _, row := range toolRows {
			paths = append(paths, row.RelativePath)
		}
		t.Fatalf("stored %d tool rows, want 1: %v", len(toolRows), paths)
	}
	if toolRows[0].RelativePath != "convtool/claude:a/0/0" {
		t.Fatalf("path = %q, want convtool/claude:a/0/0", toolRows[0].RelativePath)
	}
	for _, want := range []string{"Bash", "ls", "ls -la /tmp"} {
		if !strings.Contains(toolRows[0].Content, want) {
			t.Fatalf("row content %q is missing %q", toolRows[0].Content, want)
		}
	}
}

// TestASplitToolCallNamesTheToolOnEveryPiece covers the payload large enough to
// split. A search for the tool must match the second piece as well as the first,
// so every piece begins with the name.
func TestASplitToolCallNamesTheToolOnEveryPiece(t *testing.T) {
	t.Parallel()

	documents := []model.ConversationDocument{{
		ConversationID: "claude:a",
		MessageIndex:   0,
		Role:           "assistant",
		Tools: []model.ConversationToolCall{{
			Name:     "Write",
			Display:  strings.Repeat("alpha beta gamma ", 400),
			Command:  "",
			LangHint: "",
			Output:   "",
			IsError:  false,
		}},
	}}

	chunks, err := conversationDocumentsToStoredChunks(context.Background(), documents, 200)
	if err != nil {
		t.Fatalf("conversationDocumentsToStoredChunks returned error: %v", err)
	}

	pieces := 0
	for _, chunk := range chunks {
		if !strings.HasPrefix(chunk.RelativePath, "convtool/") {
			continue
		}
		pieces++
		if !strings.HasPrefix(chunk.Content, "Write") {
			t.Fatalf("piece at %q does not name the tool: %q", chunk.RelativePath, chunk.Content)
		}
	}
	if pieces < 2 {
		t.Fatalf("expected the payload to split, got %d pieces", pieces)
	}
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `go test ./internal/daemon/ -run TestOneRowPerToolCall`
Expected: FAIL, `stored 3 tool rows, want 1`

- [ ] **Step 3: Build one row's content**

In `internal/daemon/manager_conversation_tools.go`, replace `conversationToolTokenContent`:

```go
// conversationToolContent is everything one tool call stores: the tool's name,
// the program names and file paths its shell command decomposes into, and then
// what the user saw.
//
// A tool call used to store three rows and two carried the same text, because
// this function appended the arguments to the summary while the arguments also
// took a row of their own. One row cannot repeat itself.
func conversationToolContent(toolCall model.ConversationToolCall) string {
	tokens := make([]string, 0)
	appendConversationToken(&tokens, toolCall.Name)
	command := strings.TrimSpace(toolCall.Command)
	if command != "" {
		appendConversationShellTokens(&tokens, command)
	}
	appendConversationToken(&tokens, toolCall.Display)
	return strings.Join(tokens, "\n")
}
```

Delete `truncateConversationToolSummary` and the `conversationToolSummaryMaxBytes` constant if nothing else uses them. Check with the compiler rather than by eye.

- [ ] **Step 4: Store one chunk per call**

In `internal/daemon/manager_conversations.go`, replace the per-tool block inside `conversationDocumentsToStoredChunks`:

```go
		for toolIndex, toolCall := range document.Tools {
			toolPath := conversationToolCallPath(conversationID, document.MessageIndex, toolIndex)
			chunks = appendStorableConversationField(
				chunks,
				conversationToolContent(toolCall),
				budget,
				func(piece string, partIndex int, multipart bool) model.StoredChunk {
					chunkRelativePath := toolPath
					if multipart {
						chunkRelativePath = fmt.Sprintf("%s/%d", toolPath, partIndex)
					}
					return newConversationStoredChunk(
						document,
						conversationID,
						parentConversationID,
						chunkRelativePath,
						namedToolPiece(toolCall.Name, piece, partIndex),
						"",
						0,
						0,
					)
				},
			)
		}
```

Add beside it:

```go
// namedToolPiece keeps the tool's name at the head of every piece of a split
// payload, so a search for the tool matches the second piece as well as the
// first. The first piece already begins with the name, so it is returned as it
// is rather than carrying the name twice.
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

The `dispatcher` and `splitConversationToolPayload` calls go away with this block. Delete `splitConversationToolPayload`, `newConversationToolDispatcher`, `conversationToolExtension`, and `conversationToolExtensions` if the compiler shows nothing else uses them.

- [ ] **Step 5: Run it and watch it pass**

Run: `go test ./internal/daemon/ -run 'TestOneRowPerToolCall|TestASplitToolCall'`
Expected: PASS

- [ ] **Step 6: Fix the derived-work classifier**

In `internal/daemon/item_source.go`, `conversationNeedsDerivedWork` expects a tool row under `conversationToolMessagePath(...) + "/"`. That prefix still holds, because a call's row is `convtool/<conversation>/<message>/<toolIndex>`. Confirm with a test rather than by reading:

```go
// TestDerivedWorkExpectsTheOneToolRow pins the backfill classifier to the shape
// the generator now writes. A classifier expecting a path the generator no longer
// produces reports work needed on every pass forever.
func TestDerivedWorkExpectsTheOneToolRow(t *testing.T) {
	t.Parallel()

	documents := []model.ConversationDocument{{
		ConversationID: "claude:a",
		MessageIndex:   0,
		Role:           "assistant",
		Tools: []model.ConversationToolCall{{
			Name: "Bash", Display: "ls", Command: "ls", LangHint: "bash", Output: "", IsError: false,
		}},
	}}
	chunks, err := conversationDocumentsToStoredChunks(context.Background(), documents)
	if err != nil {
		t.Fatalf("conversationDocumentsToStoredChunks returned error: %v", err)
	}
	stored := make(map[string]string, len(chunks))
	for _, chunk := range chunks {
		stored[chunk.RelativePath] = "hash"
	}
	if conversationNeedsDerivedWork("claude:a", documents, stored) {
		t.Fatal("the classifier wants work for a conversation whose rows are all present")
	}
}
```

- [ ] **Step 7: Run the full daemon package**

Run: `go test ./internal/daemon/`
Expected: PASS. Update the existing tool tests that assert `/tok`, `/cmd`, or `/in` paths; those paths no longer exist.

- [ ] **Step 8: Run every engine gate**

Run each and require rc=0:

```
make fmt   GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile
make build GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile
make test  GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile
make live  GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile
```

- [ ] **Step 9: Commit**

```bash
git add internal/
git commit -S -m "Store one row per conversation tool call"
```

---

## Task 9: Prove it against a real store

**Files:**
- Create (engine): `test/live/tool_row_live_test.go`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the live test**

Follow `test/live/blank_row_live_test.go` in the same package for the harness, which boots an isolated daemon on a throwaway socket and registers a fresh Milvus collection that cannot collide with the operator's.

```go
//go:build live

package live

import (
	"fmt"
	"strings"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

// TestOneRowPerToolCallEndToEnd proves against a real store that a tool call
// lands as one row holding what the user saw, with no harness serialization.
func TestOneRowPerToolCallEndToEnd(t *testing.T) {
	harness := newHarness(t)

	conversationID := "toolrow-shapes"
	documents := map[string][]*pb.ConversationDocument{
		conversationID: {{
			ConversationId: conversationID,
			MessageIndex:   0,
			Role:           "assistant",
			TimestampUnix:  1712345000,
			Text:           "running two tools",
			Tools: []*pb.ConversationToolCall{
				{
					Name:     "Bash",
					Display:  "ls -la /work/" + conversationID,
					Command:  "ls -la /work/" + conversationID,
					LangHint: "bash",
				},
				{
					Name:    "Read",
					Display: "/work/" + conversationID + "/main.go",
				},
			},
		}},
	}

	job := harness.upsert(
		documents,
		pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN,
		true,
		false,
	)
	requireCompleted(t, job, "ingest")

	for toolIndex := range 2 {
		path := fmt.Sprintf("convtool/%s/0/%d", conversationID, toolIndex)
		if count := harness.countRowsWithPrefix(path); count != 1 {
			t.Fatalf("tool %d stored %d rows under %q, want 1", toolIndex, count, path)
		}
	}

	for _, marker := range []string{"file_path", "\"command\"", "{\"", "input_text"} {
		if count := harness.countRowsContaining(marker); count != 0 {
			t.Fatalf("%d stored rows carry the serialization marker %q", count, marker)
		}
	}

	if empty := harness.countRowsHoldingNothing(); empty != 0 {
		t.Fatalf("%d stored rows hold nothing", empty)
	}
	_ = strings.TrimSpace
}
```

Add `countRowsContaining` beside `countRowsHoldingNothing` in the same file, using the `like` filter form the harness already uses for prefixes.

- [ ] **Step 2: Run it**

Run: `go test -tags live -count=1 -v -timeout 20m ./test/live/ -run TestOneRowPerToolCallEndToEnd`
Expected: PASS

- [ ] **Step 3: Prove the test detects the defect**

Temporarily restore the arguments append inside `conversationToolContent` and re-run. The test must fail. Restore the file afterward.

A live test that passes on both the fixed and the broken tree proves nothing.

- [ ] **Step 4: Commit**

```bash
git add test/live/
git commit -S -m "Prove one row per tool call against a real store"
```

---

## Task 10: Land it and measure

- [ ] **Step 1: Open both pull requests**

The engine change and the clyde change are coupled by the wire. Merge the engine first, since clyde builds against the local engine checkout through `go.work`.

- [ ] **Step 2: Deploy the engine**

```bash
git -C /Users/agoodkind/Sites/lm-semantic-search fetch origin
git -C /Users/agoodkind/Sites/lm-semantic-search merge --ff-only origin/main
make -C /Users/agoodkind/Sites/lm-semantic-search deploy GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile
```

- [ ] **Step 3: Deploy clyde**

```bash
make -C /Users/agoodkind/Sites/clyde-dev/clyde deploy
```

- [ ] **Step 4: Record the baseline before turning writes on**

Run `scratchpad/verify.py` from the session directory, or re-derive it: count rows under `relativePath != ''`, rows under `content == ''`, and rows under each of `convtool/%/tok`, `convtool/%/in%`, `convtool/%/cmd`. State the filter beside every count.

- [ ] **Step 5: Turn writes on**

Set `enabled = true` under `[conversation.semantic]` in `~/.config/clyde/config.toml`. Read the file first and edit that one line; the file holds several other `enabled` keys and a blind replace hits the wrong one.

- [ ] **Step 6: Measure**

A conversation ingested after this change stores one row per tool call, and no stored row written after the change contains `file_path`, `"command"`, or `input_text`.

The tool-row total falls rather than rises as conversations are re-ingested, because three rows become one.

The blank-row count stays flat.

---

## Self-review

**Spec coverage.** Every section of the spec maps to a task: the rule and what sends it to Tasks 2 through 4, the wire to Task 6, what the engine stores to Task 8, size and splitting to Task 8 Step 4, existing rows to the reconcile behavior stated in Task 10 Step 6, tool results to the unchanged output field in Task 6, and what proves it to Task 9.

**One gap found and closed.** The spec says nothing about export, but `toolFullDetailText` dumps the same raw JSON, so an export has the same defect. Task 5 fixes it, and it is the reason `Display` lives on `transcript.ToolCall` rather than only on the search projection.

**Type consistency.** `Display` is the field name in `transcript.ToolCall` (Task 1), `semsearch.SemToolCall` (Task 7), and the proto (Task 6). `toolDisplayText(name string, input transcript.ToolInputJSON) string` is the same signature in all four provider packages, each unexported in its own package.

**Known risk.** Task 2 through 4 name the JSON keys each harness uses. Those lists come from the shapes visible in this session's data and are not exhaustive. A tool whose shape is missing stores its name alone rather than wrong text, which is a visible gap rather than a silent one, and Task 9 Step 1 does not assert full coverage.
