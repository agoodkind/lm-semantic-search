// Package collection defines the backend-neutral model of a searchable
// collection: the declared scalar schema, typed scalar values, the filter tree
// and its compiler, ranked search requests and hits, and the [Store] interface
// a vector backend implements.
package collection

// ScalarType is the closed set of scalar column types a collection can
// declare.
type ScalarType string

const (
	// ScalarTypeString is a variable-length string column with a maximum length.
	ScalarTypeString ScalarType = "string"
	// ScalarTypeBool is a boolean column.
	ScalarTypeBool ScalarType = "bool"
	// ScalarTypeInt64 is a 64-bit signed integer column.
	ScalarTypeInt64 ScalarType = "int64"
)

// ScalarColumn declares one scalar column of a collection. MaxLength applies
// only to a string column and is zero for every other type.
type ScalarColumn struct {
	Name      string     `json:"name"`
	Type      ScalarType `json:"type"`
	Nullable  bool       `json:"nullable"`
	MaxLength int32      `json:"max_length,omitempty"`
}

// Declaration is the scalar schema of a collection. The declared string column
// ItemIDColumn stores the client item id.
type Declaration struct {
	ItemIDColumn string         `json:"item_id_column"`
	Scalars      []ScalarColumn `json:"scalars"`
}

// ScalarValue is one typed value of a declared scalar column. Type selects the
// field that stores the value. Null marks a null value of a nullable column,
// and the value fields then stay zero.
type ScalarValue struct {
	Type   ScalarType `json:"type"`
	Null   bool       `json:"null,omitempty"`
	String string     `json:"string,omitempty"`
	Bool   bool       `json:"bool,omitempty"`
	Int64  int64      `json:"int64,omitempty"`
}

// StringScalar returns a string scalar value.
func StringScalar(value string) ScalarValue {
	return ScalarValue{Type: ScalarTypeString, Null: false, String: value, Bool: false, Int64: 0}
}

// BoolScalar returns a bool scalar value.
func BoolScalar(value bool) ScalarValue {
	return ScalarValue{Type: ScalarTypeBool, Null: false, String: "", Bool: value, Int64: 0}
}

// Int64Scalar returns an int64 scalar value.
func Int64Scalar(value int64) ScalarValue {
	return ScalarValue{Type: ScalarTypeInt64, Null: false, String: "", Bool: false, Int64: value}
}

// EmptyScalar returns the zero scalar value: no type, not null.
func EmptyScalar() ScalarValue {
	return ScalarValue{Type: "", Null: false, String: "", Bool: false, Int64: 0}
}
