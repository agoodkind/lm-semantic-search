//go:build restartacceptance || live

package sandboxharness

import (
	"fmt"

	"google.golang.org/grpc"
	"google.golang.org/grpc/encoding"
)

// gRPC requires an unused dynamic server argument and dynamic codec values.
// The handler processes only ServerStream; the codec processes only raw bytes.
var (
	_ grpc.StreamHandler = (*EmbeddingStoreProxy)(nil).streamHandler
	_ encoding.Codec     = rawCodec{}
)

func (proxy *EmbeddingStoreProxy) streamHandler(_ any, stream grpc.ServerStream) error {
	return proxy.forward(stream)
}

type rawCodec struct{}

func (rawCodec) Name() string {
	return "proto"
}

func (rawCodec) Marshal(value any) ([]byte, error) {
	body, ok := value.([]byte)
	if !ok {
		return nil, fmt.Errorf("raw codec cannot marshal %T", value)
	}
	return body, nil
}

func (rawCodec) Unmarshal(body []byte, value any) error {
	target, ok := value.(*[]byte)
	if !ok {
		return fmt.Errorf("raw codec cannot unmarshal into %T", value)
	}
	*target = append((*target)[:0], body...)
	return nil
}
