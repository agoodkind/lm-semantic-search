//go:build milvusintegration

// Package milvusintegration runs the daemon's collection-load controls against
// a real Milvus. Every test starts its own throwaway Milvus stack in Docker (a
// uniquely named compose project with its own containers, volumes, and
// loopback-only ports chosen by the kernel), drives the production code against
// it, reads the outcome back from Milvus itself, and tears the stack down. It
// runs only under `make milvus-integration`, which sets LMS_MILVUS_INTEGRATION=1;
// `make test` skips it. Nothing here touches the operator's milvus-standalone
// project, its volumes, or its ports.
package milvusintegration

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

const (
	integrationEnv  = "LMS_MILVUS_INTEGRATION"
	etcdImage       = "quay.io/coreos/etcd:v3.5.18"
	minioImage      = "quay.io/minio/minio:RELEASE.2024-12-18T13-15-44Z"
	milvusImage     = "milvusdb/milvus:v2.6.18"
	minioRootUser   = "minioadmin"
	throwawayPrefix = "lmstest-"
	// liveMilvusContainer is the operator's real Milvus container. The harness
	// never addresses it; it only checks after teardown that it is still up, so
	// a run that somehow reached it fails loudly.
	liveMilvusContainer = "milvus-standalone"
	stackReadyTimeout   = 4 * time.Minute
	composeStopSeconds  = "30"
	fileMode            = 0o600
)

// throwawayStack is one Docker compose project started for a single test.
type throwawayStack struct {
	project    string
	grpcPort   int
	healthPort int
	minioPort  int
	memLimit   string
}

func (stack throwawayStack) milvusAddress() string {
	return net.JoinHostPort("127.0.0.1", strconv.Itoa(stack.grpcPort))
}

// requireIntegration skips unless the Makefile target opted in, then fails
// fast when docker is missing.
func requireIntegration(t *testing.T) {
	t.Helper()
	if os.Getenv(integrationEnv) == "" {
		t.Skipf("set %s=1 (make milvus-integration) to run against a throwaway Milvus stack", integrationEnv)
	}
	if _, lookErr := exec.LookPath("docker"); lookErr != nil {
		t.Fatalf("required binary docker not on PATH: %v", lookErr)
	}
}

// freePort asks the kernel for an unused loopback port.
func freePort(t *testing.T) int {
	t.Helper()
	listener, listenErr := net.Listen("tcp", "127.0.0.1:0")
	if listenErr != nil {
		t.Fatalf("find free port: %v", listenErr)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	if closeErr := listener.Close(); closeErr != nil {
		t.Fatalf("release probe port: %v", closeErr)
	}
	return port
}

// startThrowawayStack writes a compose file for a uniquely named project,
// starts it, waits for every service's healthcheck, and registers teardown of
// containers, network, and volumes for test cleanup and for SIGINT/SIGTERM.
// memLimit caps the Milvus container, for example "4g".
func startThrowawayStack(t *testing.T, memLimit string) throwawayStack {
	t.Helper()
	idBytes := make([]byte, 4)
	if _, randErr := rand.Read(idBytes); randErr != nil {
		t.Fatalf("random project id: %v", randErr)
	}
	stack := throwawayStack{
		project:    throwawayPrefix + hex.EncodeToString(idBytes),
		grpcPort:   freePort(t),
		healthPort: freePort(t),
		minioPort:  freePort(t),
		memLimit:   memLimit,
	}
	composePath := filepath.Join(t.TempDir(), "docker-compose.yml")
	if writeErr := os.WriteFile(composePath, []byte(composeFile(stack)), fileMode); writeErr != nil {
		t.Fatalf("write compose file: %v", writeErr)
	}
	compose := func(runCtx context.Context, args ...string) ([]byte, error) {
		command := exec.CommandContext(runCtx, "docker", append([]string{"compose", "-p", stack.project, "-f", composePath}, args...)...)
		return command.CombinedOutput()
	}
	teardown := sync.OnceFunc(func() {
		downCtx, cancel := context.WithTimeout(context.Background(), stackReadyTimeout)
		defer cancel()
		started := time.Now()
		output, downErr := compose(downCtx, "down", "--volumes", "--remove-orphans", "--timeout", composeStopSeconds)
		if downErr != nil {
			t.Errorf("tear down %s: %v\n%s (remove it by hand: docker compose -p %s down -v)", stack.project, downErr, output, stack.project)
			return
		}
		t.Logf("tore down %s in %s", stack.project, time.Since(started).Round(time.Second))
		assertNoLeftovers(t, stack.project)
		assertLiveStackUntouched(t)
	})
	t.Cleanup(teardown)
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, os.Interrupt, syscall.SIGTERM)
	go func() {
		if _, ok := <-signals; ok {
			teardown()
			os.Exit(130)
		}
	}()
	t.Cleanup(func() {
		signal.Stop(signals)
		close(signals)
	})

	t.Logf("starting throwaway stack %s (grpc %d, health %d, minio %d, milvus mem_limit %s)", stack.project, stack.grpcPort, stack.healthPort, stack.minioPort, memLimit)
	started := time.Now()
	upCtx, cancelUp := context.WithTimeout(context.Background(), stackReadyTimeout+time.Minute)
	defer cancelUp()
	output, upErr := compose(upCtx, "up", "--detach", "--wait", "--wait-timeout", strconv.Itoa(int(stackReadyTimeout.Seconds())))
	if upErr != nil {
		t.Fatalf("start throwaway stack: %v\n%s", upErr, output)
	}
	t.Logf("stack %s healthy after %s", stack.project, time.Since(started).Round(time.Second))
	return stack
}

// assertNoLeftovers fails when any container, volume, or network of the
// project outlived the teardown.
func assertNoLeftovers(t *testing.T, project string) {
	t.Helper()
	checks := [][]string{
		{"ps", "--all", "--filter", "name=" + project, "--format", "{{.Names}}"},
		{"volume", "ls", "--filter", "name=" + project, "--format", "{{.Name}}"},
		{"network", "ls", "--filter", "name=" + project, "--format", "{{.Name}}"},
	}
	for _, args := range checks {
		output, err := exec.Command("docker", args...).CombinedOutput()
		if err != nil {
			t.Errorf("docker %s: %v\n%s", strings.Join(args, " "), err, output)
			continue
		}
		if leftovers := strings.TrimSpace(string(output)); leftovers != "" {
			t.Errorf("docker %s still lists %q after teardown of %s", strings.Join(args, " "), leftovers, project)
		}
	}
}

// assertLiveStackUntouched fails when the operator's Milvus container exists
// on this host and is not running after the test, which would mean the run
// reached beyond its own stack.
func assertLiveStackUntouched(t *testing.T) {
	t.Helper()
	output, err := exec.Command("docker", "inspect", "--format", "{{.State.Status}}", liveMilvusContainer).CombinedOutput()
	if err != nil {
		// No live stack on this host is the normal CI shape.
		return
	}
	if status := strings.TrimSpace(string(output)); status != "running" {
		t.Errorf("live container %s is %q after the test; the throwaway stack must never reach it", liveMilvusContainer, status)
	}
}

// composeFile mirrors the operator's docker-compose.yml with unique names,
// loopback published ports, no restart policy, and a memory cap on Milvus.
func composeFile(stack throwawayStack) string {
	return fmt.Sprintf(`services:
  etcd:
    container_name: %[1]s-etcd
    image: %[2]s
    environment:
      ETCD_AUTO_COMPACTION_MODE: revision
      ETCD_AUTO_COMPACTION_RETENTION: "1000"
      ETCD_QUOTA_BACKEND_BYTES: "4294967296"
      ETCD_SNAPSHOT_COUNT: "50000"
    command: etcd -advertise-client-urls=http://etcd:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    volumes:
      - etcd:/etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 5s
      timeout: 10s
      retries: 12

  minio:
    container_name: %[1]s-minio
    image: %[3]s
    environment:
      MINIO_ROOT_USER: %[5]s
      MINIO_ROOT_PASSWORD: %[5]s
    command: minio server /minio_data
    ports:
      - "127.0.0.1:%[8]d:9000"
    volumes:
      - minio:/minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 5s
      timeout: 10s
      retries: 12

  standalone:
    container_name: %[1]s-milvus
    image: %[4]s
    command: ["milvus", "run", "standalone"]
    mem_limit: %[9]s
    security_opt:
      - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 5s
      start_period: 120s
      timeout: 10s
      retries: 24
    ports:
      - "127.0.0.1:%[6]d:19530"
      - "127.0.0.1:%[7]d:9091"
    depends_on:
      etcd:
        condition: service_healthy
      minio:
        condition: service_healthy

volumes:
  etcd:
  minio:
  milvus:
`, stack.project, etcdImage, minioImage, milvusImage, minioRootUser, stack.grpcPort, stack.healthPort, stack.minioPort, stack.memLimit)
}
