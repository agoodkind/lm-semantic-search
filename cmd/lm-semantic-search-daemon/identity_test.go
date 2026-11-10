package main

import (
	"context"
	"log/slog"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
)

type identityCaptureHandler struct {
	record slog.Record
}

func (handler *identityCaptureHandler) Enabled(context.Context, slog.Level) bool {
	return true
}

func (handler *identityCaptureHandler) Handle(_ context.Context, record slog.Record) error {
	handler.record = record.Clone()
	return nil
}

func (handler *identityCaptureHandler) WithAttrs([]slog.Attr) slog.Handler {
	return handler
}

func (handler *identityCaptureHandler) WithGroup(string) slog.Handler {
	return handler
}

func TestDaemonIdentityReportsEffectiveResidencyTimeouts(t *testing.T) {
	handler := &identityCaptureHandler{}
	previous := slog.Default()
	slog.SetDefault(slog.New(handler))
	t.Cleanup(func() { slog.SetDefault(previous) })

	logDaemonIdentity(context.Background(), config.Config{
		SocketPath:                        "/tmp/daemon.sock",
		StateRoot:                         "/tmp/state",
		MilvusCollectionLoadWaitTimeoutMS: 15000,
		MilvusCollectionIdleTimeoutMS:     900000,
	})

	attributes := map[string]int64{}
	handler.record.Attrs(func(attribute slog.Attr) bool {
		if attribute.Value.Kind() == slog.KindInt64 {
			attributes[attribute.Key] = attribute.Value.Int64()
		}
		return true
	})
	if got := attributes["milvus_collection_load_wait_timeout_ms"]; got != 15000 {
		t.Fatalf("milvus_collection_load_wait_timeout_ms = %d, want 15000", got)
	}
	if got := attributes["milvus_collection_idle_timeout_ms"]; got != 900000 {
		t.Fatalf("milvus_collection_idle_timeout_ms = %d, want 900000", got)
	}
}
