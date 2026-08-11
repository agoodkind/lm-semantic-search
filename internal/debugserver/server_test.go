package debugserver

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"
	"time"
)

func TestServerServesDebugSurfaces(t *testing.T) {
	server, err := NewWithResidencySnapshot("127.0.0.1:0", nil)
	if err != nil {
		t.Fatalf("NewWithResidencySnapshot returned error: %v", err)
	}

	ctx := context.Background()
	if startErr := server.Start(ctx); startErr != nil {
		t.Fatalf("Start returned error: %v", startErr)
	}
	t.Cleanup(func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdownCtx)
	})

	base := "http://" + server.Addr()

	varsResp := getOK(t, base+"/debug/vars")
	defer varsResp.Body.Close()
	body, readErr := io.ReadAll(varsResp.Body)
	if readErr != nil {
		t.Fatalf("read /debug/vars body: %v", readErr)
	}
	var decoded map[string]json.RawMessage
	if jsonErr := json.Unmarshal(body, &decoded); jsonErr != nil {
		t.Fatalf("/debug/vars body is not JSON: %v", jsonErr)
	}

	pprofResp := getOK(t, base+"/debug/pprof/")
	pprofResp.Body.Close()
}

func TestServerServesResidencySnapshot(t *testing.T) {
	server, err := NewWithResidencySnapshot(
		"127.0.0.1:0",
		func() ([]byte, error) {
			return []byte(`{"collections":[{"collection":"test","timer_armed":true}]}`), nil
		},
	)
	if err != nil {
		t.Fatalf("NewWithResidencySnapshot returned error: %v", err)
	}
	if startErr := server.Start(context.Background()); startErr != nil {
		t.Fatalf("Start returned error: %v", startErr)
	}
	t.Cleanup(func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdownCtx)
	})

	response := getOK(t, "http://"+server.Addr()+"/debug/residency")
	defer response.Body.Close()
	if contentType := response.Header.Get("Content-Type"); contentType != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", contentType)
	}
	body, readErr := io.ReadAll(response.Body)
	if readErr != nil {
		t.Fatalf("read residency body: %v", readErr)
	}
	var decoded struct {
		Collections []struct {
			Collection string `json:"collection"`
			TimerArmed bool   `json:"timer_armed"`
		} `json:"collections"`
	}
	if jsonErr := json.Unmarshal(body, &decoded); jsonErr != nil {
		t.Fatalf("decode residency body: %v", jsonErr)
	}
	if len(decoded.Collections) != 1 || decoded.Collections[0].Collection != "test" ||
		!decoded.Collections[0].TimerArmed {
		t.Fatalf("residency snapshot = %+v", decoded)
	}
}

func TestNewRejectsNonLoopbackHost(t *testing.T) {
	for _, addr := range []string{"0.0.0.0:0", "8.8.8.8:0"} {
		if _, err := NewWithResidencySnapshot(addr, nil); err == nil {
			t.Errorf("NewWithResidencySnapshot(%q) expected error, got nil", addr)
		}
	}
}

func TestShutdownStopsServer(t *testing.T) {
	server, err := NewWithResidencySnapshot("127.0.0.1:0", nil)
	if err != nil {
		t.Fatalf("NewWithResidencySnapshot returned error: %v", err)
	}

	if startErr := server.Start(context.Background()); startErr != nil {
		t.Fatalf("Start returned error: %v", startErr)
	}

	base := "http://" + server.Addr()
	resp := getOK(t, base+"/debug/vars")
	resp.Body.Close()

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if shutdownErr := server.Shutdown(shutdownCtx); shutdownErr != nil {
		t.Fatalf("Shutdown returned error: %v", shutdownErr)
	}

	if _, getErr := http.Get(base + "/debug/vars"); getErr == nil {
		t.Fatalf("expected request to fail after Shutdown, got success")
	}
}

func getOK(t *testing.T, url string) *http.Response {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatalf("GET %s: %v", url, err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		t.Fatalf("GET %s: status = %d, want 200", url, resp.StatusCode)
	}
	return resp
}
