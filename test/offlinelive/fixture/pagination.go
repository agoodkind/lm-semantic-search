//go:build offlinelive

package fixture

// PageOffset calculates the first row for a requested page.
func PageOffset(pageNumber int, pageSize int) int {
	return pageNumber * pageSize
}
