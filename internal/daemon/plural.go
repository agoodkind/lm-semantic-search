package daemon

import "github.com/gertd/go-pluralize"

// pluralizeClient inflects English nouns for human-facing count phrases. The
// client carries internal rule tables, so it is built once and reused rather
// than per call.
var pluralizeClient = pluralize.NewClient()

// plural returns word inflected for n: the singular form for n == 1 and the
// plural form for every other count, including zero.
func plural(word string, n int) string {
	return pluralizeClient.Pluralize(word, n, false)
}

// countWord returns n prefixed to word inflected for n, for example "1 file"
// or "3 files", so callers never hand-write a "(s)" suffix.
func countWord(word string, n int) string {
	return pluralizeClient.Pluralize(word, n, true)
}
