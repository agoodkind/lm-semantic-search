package status

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"strconv"
)

const (
	jsonNoticesKey  = "notices"
	jsonIdentityKey = "identity"
	jsonCountersKey = "counters"
	jsonActivityKey = "activity"
	jsonValueKey    = "value"
	jsonUnitKey     = "unit"
	jsonNull        = "null"
)

// WriteJSON writes one JSON object on one line followed by a newline.
// The top-level keys are notices, identity, counters, and activity, in order.
// Field objects include value and include unit when Field.Unit is not empty.
// An absent Value encodes as null.
// Zero, false, and the empty string encode as values.
// WriteJSON omits Title, Details, RunID, Group, and NoDelta.
// WriteJSON returns an error without writing bytes for NaN or infinite floats.
// WriteJSON does not escape HTML characters.
func WriteJSON(writer io.Writer, snapshot Snapshot) error {
	encoding := newJSONEncoding()
	if err := encoding.snapshot(snapshot); err != nil {
		return err
	}
	encoding.buffer.WriteString("\n")
	if _, err := writer.Write(encoding.buffer.Bytes()); err != nil {
		slog.Error("write status snapshot failed", "err", err)
		return fmt.Errorf("write status snapshot: %w", err)
	}
	return nil
}

type jsonEncoding struct {
	buffer  *bytes.Buffer
	encoder *json.Encoder
}

func newJSONEncoding() jsonEncoding {
	buffer := &bytes.Buffer{}
	encoder := json.NewEncoder(buffer)
	encoder.SetEscapeHTML(false)
	return jsonEncoding{buffer: buffer, encoder: encoder}
}

func (e jsonEncoding) snapshot(snapshot Snapshot) error {
	e.buffer.WriteString("{")
	e.key(jsonNoticesKey)
	if err := e.notices(snapshot.Notices); err != nil {
		return err
	}
	e.buffer.WriteString(",")
	e.key(jsonIdentityKey)
	if err := e.fields(snapshot.Identity); err != nil {
		return err
	}
	e.buffer.WriteString(",")
	e.key(jsonCountersKey)
	if err := e.fields(snapshot.Counters); err != nil {
		return err
	}
	e.buffer.WriteString(",")
	e.key(jsonActivityKey)
	if err := e.activity(snapshot.Activity); err != nil {
		return err
	}
	e.buffer.WriteString("}")
	return nil
}

func (e jsonEncoding) key(name string) {
	e.buffer.WriteString(strconv.Quote(name))
	e.buffer.WriteString(":")
}

func (e jsonEncoding) notices(notices []string) error {
	e.buffer.WriteString("[")
	for index, notice := range notices {
		if index > 0 {
			e.buffer.WriteString(",")
		}
		if err := e.text(notice); err != nil {
			return err
		}
	}
	e.buffer.WriteString("]")
	return nil
}

func (e jsonEncoding) activity(rows [][]Field) error {
	e.buffer.WriteString("[")
	for index, row := range rows {
		if index > 0 {
			e.buffer.WriteString(",")
		}
		if err := e.fields(row); err != nil {
			return err
		}
	}
	e.buffer.WriteString("]")
	return nil
}

func (e jsonEncoding) fields(fields []Field) error {
	e.buffer.WriteString("{")
	for index, field := range fields {
		if index > 0 {
			e.buffer.WriteString(",")
		}
		if err := e.field(field); err != nil {
			return err
		}
	}
	e.buffer.WriteString("}")
	return nil
}

func (e jsonEncoding) field(field Field) error {
	if err := e.text(field.Name); err != nil {
		return err
	}
	e.buffer.WriteString(":{")
	e.key(jsonValueKey)
	if err := e.value(field.Value); err != nil {
		return err
	}
	if field.Unit != "" {
		e.buffer.WriteString(",")
		e.key(jsonUnitKey)
		if err := e.text(field.Unit); err != nil {
			return err
		}
	}
	e.buffer.WriteString("}")
	return nil
}

func (e jsonEncoding) value(value Value) error {
	switch value.kind {
	case kindInteger:
		e.buffer.WriteString(strconv.FormatInt(value.integer, 10))
		return nil
	case kindNumber:
		return e.number(value.number)
	case kindFlag:
		e.buffer.WriteString(strconv.FormatBool(value.flag))
		return nil
	case kindText:
		return e.text(value.text)
	case kindAbsent:
		e.buffer.WriteString(jsonNull)
		return nil
	default:
		e.buffer.WriteString(jsonNull)
		return nil
	}
}

func (e jsonEncoding) text(value string) error {
	return e.trimLine(e.encoder.Encode(value))
}

func (e jsonEncoding) number(value float64) error {
	return e.trimLine(e.encoder.Encode(value))
}

func (e jsonEncoding) trimLine(err error) error {
	if err != nil {
		slog.Error("encode status snapshot failed", "err", err)
		return fmt.Errorf("encode status snapshot: %w", err)
	}
	e.buffer.Truncate(e.buffer.Len() - 1)
	return nil
}
