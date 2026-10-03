package render

import (
	"fmt"
	"strconv"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	statusdisplay "goodkind.io/lm-semantic-search/status"
)

// constantMetrics never change while the daemon runs. A delta column beside
// them would only ever read +0 and add noise to every line of the screen.
var constantMetrics = map[string]bool{
	"index_slots_total": true,
}

// StatusSnapshot converts one status reply into the raw values the generic
// status display shows. The daemon renders its display text with this same
// conversion, and the dump and the reply text read the same.
func StatusSnapshot(response *pb.GetStatusResponse) statusdisplay.Snapshot {
	daemon := response.GetDaemon()
	title := fmt.Sprintf("lm-semantic-search  version=%s  pid=%d",
		daemon.GetVersion(), daemon.GetPid())

	counters := make([]statusdisplay.Field, 0, len(response.GetMetrics()))
	for _, metric := range response.GetMetrics() {
		counters = append(counters, statusField(metric))
	}

	activity := make([][]statusdisplay.Field, 0, len(response.GetActivity()))
	for _, row := range response.GetActivity() {
		fields := make([]statusdisplay.Field, 0, len(row.GetMetrics()))
		for _, metric := range row.GetMetrics() {
			fields = append(fields, statusField(metric))
		}
		activity = append(activity, fields)
	}

	return statusdisplay.Snapshot{
		Title:    title,
		Details:  []string{"socket=" + daemon.GetSocketPath()},
		Notices:  nil,
		Identity: statusIdentity(response),
		RunID:    statusRunID(response),
		Counters: counters,
		Activity: activity,
	}
}

// statusIdentity lists the records that identify the process a reply came
// from. A reply with no daemon identity has none.
func statusIdentity(response *pb.GetStatusResponse) []statusdisplay.Field {
	daemon := response.GetDaemon()
	if daemon == nil {
		return nil
	}
	identity := []statusdisplay.Field{
		identityField("version", daemon.GetVersion()),
		identityField("commit", daemon.GetCommit()),
		identityField("pid", strconv.FormatInt(int64(daemon.GetPid()), 10)),
		identityField("socket", daemon.GetSocketPath()),
	}
	if readAt := response.GetReadAt(); readAt != nil {
		identity = append(identity,
			identityField("read_at", readAt.AsTime().UTC().Format(time.RFC3339Nano)))
	}
	return identity
}

func identityField(name string, value string) statusdisplay.Field {
	return statusdisplay.Field{
		Group:   "",
		Name:    name,
		Unit:    "",
		Value:   statusdisplay.Text(value),
		NoDelta: false,
	}
}

// statusRunID identifies one continuous run of one daemon process. A different
// pid or a different start time means the counters restarted from zero. A reply
// with no daemon identity has no run ID.
func statusRunID(response *pb.GetStatusResponse) string {
	daemon := response.GetDaemon()
	if daemon == nil {
		return ""
	}
	startedAt := daemon.GetStartedAt().AsTime().UnixNano()
	return fmt.Sprintf("%d@%d", daemon.GetPid(), startedAt)
}

// statusField converts one wire metric into a display field.
func statusField(metric *pb.Metric) statusdisplay.Field {
	return statusdisplay.Field{
		Group:   metric.GetGroup(),
		Name:    metric.GetName(),
		Unit:    metric.GetUnit(),
		Value:   statusValue(metric),
		NoDelta: constantMetrics[metric.GetName()],
	}
}

// statusValue converts the set oneof member of a wire metric. An unset value
// stays absent, which the display prints as null.
func statusValue(metric *pb.Metric) statusdisplay.Value {
	switch value := metric.GetValue().(type) {
	case *pb.Metric_IntValue:
		return statusdisplay.Int(value.IntValue)
	case *pb.Metric_DoubleValue:
		return statusdisplay.Float(value.DoubleValue)
	case *pb.Metric_BoolValue:
		return statusdisplay.Bool(value.BoolValue)
	case *pb.Metric_StringValue:
		return statusdisplay.Text(value.StringValue)
	default:
		return statusdisplay.Value{}
	}
}
