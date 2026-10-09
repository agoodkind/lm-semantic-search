// Package networkcost classifies the cost of the default network path.
// Download decisions use the classification, policy, and explicit override.
package networkcost

// Classification represents the platform's assessment of the default
// network path.
type Classification string

// The normal classification means neither expensive nor constrained.
// Expensive can mean cellular or a personal hotspot.
// Constrained means the platform reported Low Data Mode.
// Unknown means no platform classification, a timeout, or no available path.
const (
	ClassificationUnknown                 Classification = "unknown"
	ClassificationNormal                  Classification = "normal"
	ClassificationExpensive               Classification = "expensive"
	ClassificationConstrained             Classification = "constrained"
	ClassificationExpensiveAndConstrained Classification = "expensive_and_constrained"
)
