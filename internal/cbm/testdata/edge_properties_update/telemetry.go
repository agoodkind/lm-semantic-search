package telemetry

func IncAuditKafkaProduce(status string) {}

func Produce() {
	IncAuditKafkaProduce("error")
}

func ProduceWarning() {
	IncAuditKafkaProduce("warning")
}
