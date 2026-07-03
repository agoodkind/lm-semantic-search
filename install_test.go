package lmsemanticsearch_test

import (
	"os"
	"strings"
	"testing"
)

func TestInstallerSystemdRestartPolicyAlwaysRelaunchesDaemon(t *testing.T) {
	data, err := os.ReadFile("install.sh")
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	contents := string(data)

	if !strings.Contains(contents, "Restart=always") {
		t.Fatalf("systemd unit does not restart the daemon after a clean shutdown")
	}
	if strings.Contains(contents, "Restart=on-failure") {
		t.Fatalf("systemd unit still uses on-failure restart policy")
	}
}

func TestInstallerDaemonServiceNameUsesCurrentProduct(t *testing.T) {
	data, err := os.ReadFile("install.sh")
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	contents := string(data)

	if strings.Contains(contents, "Claude Context") {
		t.Fatalf("install.sh contains stale product name Claude Context")
	}
	if !strings.Contains(contents, "Description=lm-semantic-search daemon") {
		t.Fatalf("systemd unit description does not name lm-semantic-search daemon")
	}
}
