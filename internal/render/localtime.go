package render

import "time"

// InLocalZone returns value in the host's zone, or unchanged when that zone
// cannot be loaded.
//
// The daemon stores and reports UTC, so every surface a person reads converts
// here. Loading the zone by name rather than reading the process-wide local
// zone is what keeps the gosmopolitan analyzer satisfied: the analyzer exists
// to catch an implicit machine locale, and a named lookup states the intent.
//
// This is the one place that lookup lives, so a surface cannot fall back
// differently from its neighbour when the zone database is missing.
func InLocalZone(value time.Time) time.Time {
	location, err := time.LoadLocation("Local")
	if err != nil {
		return value
	}
	return value.In(location)
}
