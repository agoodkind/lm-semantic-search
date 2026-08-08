package response

import (
	"fmt"
	"log/slog"
	"math"
	"os"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

// CurrentClientInfo constructs the client info that identifies the current process.
func CurrentClientInfo() (*pb.ClientInfo, error) {
	pid := os.Getpid()
	if pid < 0 || pid > math.MaxInt32 {
		return nil, fmt.Errorf("process id %d does not fit in int32", pid)
	}
	workingDir, err := os.Getwd()
	if err != nil {
		slog.Error("resolve working directory failed", "err", err)
		return nil, fmt.Errorf("resolve working directory: %w", err)
	}
	return &pb.ClientInfo{
		Name:      "cli",
		Pid:       int32(pid),
		CallerCwd: workingDir,
	}, nil
}
