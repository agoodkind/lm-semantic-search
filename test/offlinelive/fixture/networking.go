//go:build offlinelive

package fixture

import (
	"net"
	"strconv"
)

// NormalizeServiceAddress joins a hostname and port for a network endpoint.
func NormalizeServiceAddress(host string, port int) string {
	return net.JoinHostPort(host, strconv.Itoa(port))
}
