// Package debugserver exposes Go runtime profiling (net/http/pprof) and
// the expvar /debug/vars surface over a loopback-only HTTP listener for
// live daemon introspection. It is the first HTTP server in the process
// and is constrained to the local machine: New rejects any non-loopback
// bind host so the profiling endpoints are never reachable off-host.
package debugserver

import (
	"context"
	"errors"
	"expvar"
	"fmt"
	"log/slog"
	"net"
	"net/http"

	// Importing net/http/pprof also registers its handlers on
	// http.DefaultServeMux via the package init. That side effect is
	// inert here because Server serves its own dedicated mux and never
	// the default one, so do not drop this import: the explicit
	// pprof.* handler functions below come from it.
	"net/http/pprof"
	"net/netip"
	"time"
)

// readHeaderTimeout bounds how long the debug listener waits for request
// headers, closing the gosec G112 Slowloris exposure on the loopback
// profiling surface.
const readHeaderTimeout = 5 * time.Second

// Server wraps an [http.Server] that serves the profiling and expvar
// surfaces over a loopback address.
type Server struct {
	httpServer *http.Server
	addr       string
}

// New validates that the host portion of addr is a loopback address and
// returns a Server bound to a dedicated mux carrying the net/http/pprof
// and expvar /debug/vars handlers. A non-loopback host is rejected with
// a wrapped error so the profiling surface cannot be exposed off-host.
func New(addr string) (*Server, error) {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		slog.Error("debugserver: split host/port failed", "addr", addr, "err", err)
		return nil, fmt.Errorf("debugserver: split host/port from %q: %w", addr, err)
	}

	parsedHost, err := netip.ParseAddr(host)
	if err != nil {
		slog.Error("debugserver: parse host failed", "addr", addr, "err", err)
		return nil, fmt.Errorf("debugserver: parse host %q: %w", host, err)
	}

	if !parsedHost.IsLoopback() {
		return nil, fmt.Errorf("debugserver: refusing non-loopback bind host %q", host)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/debug/pprof/", pprof.Index)
	mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
	mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
	mux.Handle("/debug/vars", expvar.Handler())

	return &Server{
		httpServer: &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: readHeaderTimeout},
		addr:       addr,
	}, nil
}

// Start eagerly binds the configured address, records the actual bound
// address (so a caller using port 0 can discover it via Addr), and
// serves requests in a panic-recovered goroutine. A bind failure is
// returned synchronously so the caller learns about it before the
// goroutine starts.
func (server *Server) Start(ctx context.Context) error {
	listener, err := (&net.ListenConfig{}).Listen(ctx, "tcp", server.addr)
	if err != nil {
		return fmt.Errorf("debugserver: listen on %q: %w", server.addr, err)
	}
	server.addr = listener.Addr().String()

	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.ErrorContext(ctx, "debugserver goroutine panic", "err", fmt.Errorf("panic: %v", recovered))
			}
		}()
		if serveErr := server.httpServer.Serve(listener); serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
			slog.ErrorContext(ctx, "debugserver serve failed", "addr", server.addr, "err", serveErr)
		}
	}()

	return nil
}

// Shutdown gracefully stops the underlying [http.Server].
func (server *Server) Shutdown(ctx context.Context) error {
	if err := server.httpServer.Shutdown(ctx); err != nil {
		slog.ErrorContext(ctx, "debugserver: shutdown failed", "addr", server.addr, "err", err)
		return fmt.Errorf("debugserver: shutdown: %w", err)
	}
	return nil
}

// Addr returns the bound address, which reflects the actual port chosen
// by the kernel when the configured address requested port 0.
func (server *Server) Addr() string {
	return server.addr
}
