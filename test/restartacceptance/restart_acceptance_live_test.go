//go:build restartacceptance && restartacceptancelive

package restartacceptance

import (
	"context"
	"os"
	"os/signal"
	"syscall"
	"testing"
	"time"
)

const restartAcceptanceLifecycleTimeout = 90 * time.Minute

func TestRestartAcceptance(t *testing.T) {
	if err := validateRestartAcceptanceConfirmations(
		os.Getenv(restartAcceptanceOptIn),
		os.Getenv(productionDatabaseConfirmation),
	); err != nil {
		t.Fatal(err)
	}
	operations, err := newRealAcceptanceLifecycleOperations()
	if err != nil {
		t.Fatal(err)
	}
	signalContext, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	ctx, cancel := context.WithTimeout(signalContext, restartAcceptanceLifecycleTimeout)
	defer cancel()
	if err := executeRestartAcceptance(ctx, operations); err != nil {
		t.Fatal(err)
	}
}
