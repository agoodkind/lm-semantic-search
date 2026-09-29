// Package vectorcodec defines the byte form and checksum of a canonical
// vector. The library and every vector adapter compute the checksum the same
// way.
package vectorcodec

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
)

// float32Bytes is the byte width of one encoded vector value.
const float32Bytes = 4

// Encode returns the little-endian IEEE 754 bytes of values.
func Encode(values []float32) []byte {
	encoded := make([]byte, len(values)*float32Bytes)
	for index, value := range values {
		binary.LittleEndian.PutUint32(encoded[index*float32Bytes:], math.Float32bits(value))
	}
	return encoded
}

// Decode returns the values of bytes produced by [Encode].
func Decode(encoded []byte) ([]float32, error) {
	if len(encoded)%float32Bytes != 0 {
		return nil, fmt.Errorf("encoded vector has %d bytes, not a multiple of %d", len(encoded), float32Bytes)
	}
	values := make([]float32, len(encoded)/float32Bytes)
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(encoded[index*float32Bytes:]))
	}
	return values, nil
}

// Checksum returns the lowercase hex SHA-256 of the encoded values.
func Checksum(values []float32) string {
	sum := sha256.Sum256(Encode(values))
	return hex.EncodeToString(sum[:])
}

// Validate reports whether values has the dimension, contains only finite
// values, and has a nonzero norm. A COSINE score is undefined for a zero or
// nonfinite vector.
func Validate(values []float32, dimension int) error {
	if len(values) != dimension {
		return fmt.Errorf("vector has %d values, want %d", len(values), dimension)
	}
	var squaredNorm float64
	for index, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("vector value %d is not finite", index)
		}
		squaredNorm += float64(value) * float64(value)
	}
	if squaredNorm == 0 {
		return errors.New("vector has zero norm")
	}
	return nil
}
