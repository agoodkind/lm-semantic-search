package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

func TestSetProfileReplacesNullConfig(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(configPath, []byte("null"), persistedConfigFileMode); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}

	if err := SetProfile(configPath, ProfileOffline); err != nil {
		t.Fatalf("SetProfile returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	var document map[string]string
	if err := json.Unmarshal(data, &document); err != nil {
		t.Fatalf("Unmarshal returned error: %v", err)
	}
	if len(document) != 1 {
		t.Fatalf("config field count = %d want 1", len(document))
	}
	if document[profileJSONField] != ProfileOffline {
		t.Fatalf(
			"profile = %q want %q",
			document[profileJSONField],
			ProfileOffline,
		)
	}
}

func TestSetOfflineModelPreservesConfig(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.json")
	initialData := []byte(`{"profile":"offline","futureField":{"enabled":true}}`)
	if err := os.WriteFile(configPath, initialData, persistedConfigFileMode); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}

	if err := SetOfflineModel(configPath, offlinemodel.BGESmall); err != nil {
		t.Fatalf("SetOfflineModel returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	var document map[string]json.RawMessage
	if err := json.Unmarshal(data, &document); err != nil {
		t.Fatalf("Unmarshal returned error: %v", err)
	}
	if string(document[profileJSONField]) != `"offline"` {
		t.Fatalf("profile was not preserved: %s", document[profileJSONField])
	}
	if string(document[offlineEmbeddingModelJSONField]) != `"bge-small"` {
		t.Fatalf(
			"offlineEmbeddingModel = %s want %q",
			document[offlineEmbeddingModelJSONField],
			offlinemodel.BGESmall,
		)
	}
	if _, found := document["futureField"]; !found {
		t.Fatal("futureField was not preserved")
	}
}

func TestSetOfflineModelRejectsUnknownWithoutWriting(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.json")
	initialData := []byte("{\"profile\":\"offline\"}\n")
	if err := os.WriteFile(configPath, initialData, persistedConfigFileMode); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}

	if err := SetOfflineModel(configPath, "unknown"); err == nil {
		t.Fatal("SetOfflineModel returned no error")
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	if string(data) != string(initialData) {
		t.Fatalf("config changed after invalid model: %q", data)
	}
}

func TestSetOfflineModelEmptyIsNoOp(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.json")
	if err := SetOfflineModel(configPath, ""); err != nil {
		t.Fatalf("SetOfflineModel returned error: %v", err)
	}
	if _, err := os.Stat(configPath); !os.IsNotExist(err) {
		t.Fatalf("empty model wrote config or stat failed: %v", err)
	}
}

func TestWritePersistedConfigReplacesFile(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.json")
	initialData := []byte(`{"profile":"standard"}`)
	if err := os.WriteFile(configPath, initialData, persistedConfigFileMode); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	wantData := []byte("{\"profile\":\"offline\"}\n")

	if err := writePersistedConfig(configPath, wantData); err != nil {
		t.Fatalf("writePersistedConfig returned error: %v", err)
	}

	gotData, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	if string(gotData) != string(wantData) {
		t.Fatalf("config = %q want %q", gotData, wantData)
	}
}
