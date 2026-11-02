package telemetry

func IncAuditKafkaProduce(status string) {}

func Produce() {
	IncAuditKafkaProduce("error")
}
