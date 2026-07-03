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
