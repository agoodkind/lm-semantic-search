package main

import (
	"context"
	"errors"
	"strings"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/render"
	statusdisplay "goodkind.io/lm-semantic-search/status"
)

// newStatusSource returns the source the status display reads. Each call reads
// one status reply from the daemon and converts it into raw display values.
func newStatusSource(options cliOptions) statusdisplay.Source {
	return func() (statusdisplay.Snapshot, error) {
		reply, err := fetchStatusResponse(options)
		if err != nil {
			return statusdisplay.Snapshot{}, err
		}
		snapshot := render.StatusSnapshot(reply)
		snapshot.Notices = statusNotices(reply, snapshot)
		return snapshot, nil
	}
}

// statusNotices returns the lines the daemon placed before the records in its
// display text: the maintenance banner, the dependency-health banner, and the
// correlation header. The daemon renders the records from the same values as
// the snapshot, and the notices are the text in front of them.
func statusNotices(reply *pb.GetStatusResponse, snapshot statusdisplay.Snapshot) []string {
	text := strings.TrimSpace(reply.GetDisplayText())
	records := strings.TrimSpace(statusdisplay.Dump(snapshot))
	prefix := strings.TrimSuffix(strings.TrimSuffix(text, records), "\n")
	if prefix == "" {
		return nil
	}
	return strings.Split(prefix, "\n")
}

// fetchStatusResponse reads one status reply, rejecting an unexpected reply type
// rather than rendering a zero value as if the daemon had reported it.
func fetchStatusResponse(options cliOptions) (*pb.GetStatusResponse, error) {
	result, err := callDaemon(options, func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
		return client.GetStatus(ctx, &pb.GetStatusRequest{})
	})
	if err != nil {
		return nil, err
	}
	reply, ok := result.(*pb.GetStatusResponse)
	if !ok {
		return nil, errors.New("unexpected response type from GetStatus")
	}
	return reply, nil
}
