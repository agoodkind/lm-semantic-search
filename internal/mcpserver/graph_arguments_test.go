package mcpserver

import (
	"encoding/json"
	"testing"
)

func decodeArguments(t *testing.T, argsJSON string) map[string]json.RawMessage {
	t.Helper()

	var args map[string]json.RawMessage
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		t.Fatalf("unmarshal arguments: %v", err)
	}
	return args
}

func requireJSONValue(t *testing.T, args map[string]json.RawMessage, key string, expected string) {
	t.Helper()

	value, found := args[key]
	if !found {
		t.Fatalf("missing %q in %#v", key, args)
	}
	if string(value) != expected {
		t.Fatalf("%s = %s, want %s", key, value, expected)
	}
}

func TestMarshalTracePathArgumentsIncludesSetOptionalValues(t *testing.T) {
	t.Parallel()

	depth := 3
	argsJSON, err := MarshalTracePathArguments("handleRequest", "both", &depth)
	if err != nil {
		t.Fatalf("MarshalTracePathArguments returned error: %v", err)
	}

	args := decodeArguments(t, argsJSON)
	requireJSONValue(t, args, "function_name", `"handleRequest"`)
	requireJSONValue(t, args, "direction", `"both"`)
	requireJSONValue(t, args, "depth", `3`)
	if _, found := args["project"]; found {
		t.Fatal("trace_path arguments must not include project")
	}
}

func TestMarshalTracePathArgumentsOmitsUnsetOptionalValues(t *testing.T) {
	t.Parallel()

	argsJSON, err := MarshalTracePathArguments("handleRequest", "", nil)
	if err != nil {
		t.Fatalf("MarshalTracePathArguments returned error: %v", err)
	}

	args := decodeArguments(t, argsJSON)
	requireJSONValue(t, args, "function_name", `"handleRequest"`)
	if _, found := args["direction"]; found {
		t.Fatal("trace_path arguments must omit empty direction")
	}
	if _, found := args["depth"]; found {
		t.Fatal("trace_path arguments must omit unset depth")
	}
	if _, found := args["project"]; found {
		t.Fatal("trace_path arguments must not include project")
	}
}

func TestMarshalGetArchitectureArgumentsIncludesPathWhenSet(t *testing.T) {
	t.Parallel()

	argsJSON, err := MarshalGetArchitectureArguments("internal/daemon")
	if err != nil {
		t.Fatalf("MarshalGetArchitectureArguments returned error: %v", err)
	}

	args := decodeArguments(t, argsJSON)
	requireJSONValue(t, args, "path", `"internal/daemon"`)
	if _, found := args["project"]; found {
		t.Fatal("get_architecture arguments must not include project")
	}
}

func TestMarshalGetArchitectureArgumentsOmitsUnsetPath(t *testing.T) {
	t.Parallel()

	argsJSON, err := MarshalGetArchitectureArguments("")
	if err != nil {
		t.Fatalf("MarshalGetArchitectureArguments returned error: %v", err)
	}

	args := decodeArguments(t, argsJSON)
	if len(args) != 0 {
		t.Fatalf("get_architecture arguments = %#v, want empty object", args)
	}
}

func TestMarshalManageADRArgumentsIncludesSetOptionalValues(t *testing.T) {
	t.Parallel()

	sections := []string{"Context", "Decision"}
	argsJSON, err := MarshalManageADRArguments("update", "Use gRPC only.", sections)
	if err != nil {
		t.Fatalf("MarshalManageADRArguments returned error: %v", err)
	}

	args := decodeArguments(t, argsJSON)
	requireJSONValue(t, args, "mode", `"update"`)
	requireJSONValue(t, args, "content", `"Use gRPC only."`)
	requireJSONValue(t, args, "sections", `["Context","Decision"]`)
	if _, found := args["project"]; found {
		t.Fatal("manage_adr arguments must not include project")
	}
}

func TestMarshalManageADRArgumentsOmitsUnsetOptionalValues(t *testing.T) {
	t.Parallel()

	argsJSON, err := MarshalManageADRArguments("get", "", nil)
	if err != nil {
		t.Fatalf("MarshalManageADRArguments returned error: %v", err)
	}

	args := decodeArguments(t, argsJSON)
	requireJSONValue(t, args, "mode", `"get"`)
	if _, found := args["content"]; found {
		t.Fatal("manage_adr arguments must omit empty content")
	}
	if _, found := args["sections"]; found {
		t.Fatal("manage_adr arguments must omit unset sections")
	}
	if _, found := args["project"]; found {
		t.Fatal("manage_adr arguments must not include project")
	}
}
