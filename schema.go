package structllm

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"time"
)

// Schema is the subset of JSON Schema that structllm generates and validates.
// It is serialisable, so it can be sent to providers that support
// schema-constrained output.
type Schema struct {
	Type                 string             `json:"type,omitempty"`
	Description          string             `json:"description,omitempty"`
	Properties           map[string]*Schema `json:"properties,omitempty"`
	Required             []string           `json:"required,omitempty"`
	AdditionalProperties any                `json:"additionalProperties,omitempty"` // false or *Schema
	Items                *Schema            `json:"items,omitempty"`
	Enum                 []any              `json:"enum,omitempty"`
	Format               string             `json:"format,omitempty"`
	Pattern              string             `json:"pattern,omitempty"`
	MinLength            *int               `json:"minLength,omitempty"`
	MaxLength            *int               `json:"maxLength,omitempty"`
	MinItems             *int               `json:"minItems,omitempty"`
	MaxItems             *int               `json:"maxItems,omitempty"`
	Minimum              *float64           `json:"minimum,omitempty"`
	Maximum              *float64           `json:"maximum,omitempty"`

	order    []string // property order as declared in the struct (for stable prompts)
	nullable bool     // pointer fields: null is allowed
}

// MarshalJSON keeps "properties" in struct declaration order, which makes
// prompts deterministic and easier for models to follow.
func (s *Schema) MarshalJSON() ([]byte, error) {
	type plain Schema
	nullType := func(b []byte) []byte {
		if !s.nullable || s.Type == "" {
			return b
		}
		prefix := `{"type":"` + s.Type + `"`
		return []byte(strings.Replace(string(b), prefix, `{"type":["`+s.Type+`","null"]`, 1))
	}
	if len(s.Properties) == 0 {
		b, err := json.Marshal((*plain)(s))
		return nullType(b), err
	}
	cp := *s
	props := cp.Properties
	cp.Properties = nil
	base, err := json.Marshal((*plain)(&cp))
	if err != nil {
		return nil, err
	}
	var b strings.Builder
	b.WriteString(`{"properties":{`)
	for i, name := range s.order {
		if i > 0 {
			b.WriteByte(',')
		}
		k, _ := json.Marshal(name)
		v, err := json.Marshal(props[name])
		if err != nil {
			return nil, err
		}
		b.Write(k)
		b.WriteByte(':')
		b.Write(v)
	}
	b.WriteByte('}')
	if len(base) > 2 {
		b.WriteByte(',')
		b.Write(base[1 : len(base)-1])
	}
	b.WriteByte('}')
	return nullType([]byte(b.String())), nil
}

// SchemaFor builds a JSON Schema for T from its type and struct tags.
//
// Supported tags (alongside the usual `json:"name,omitempty"`):
//
//	desc:"text"          description shown to the model
//	enum:"a,b,c"         allowed values (strings, or numbers for numeric fields)
//	minLen:"1" maxLen:"200"
//	minItems:"1" maxItems:"5"
//	min:"0" max:"100"    numeric bounds
//	pattern:"^[A-Z]+$"   regular expression for strings
//	format:"email"       informational format (date-time is set for time.Time)
//
// Fields are required unless they are pointers or tagged omitempty.
func SchemaFor[T any]() (*Schema, error) {
	return schemaOf(reflect.TypeFor[T](), map[reflect.Type]bool{})
}

var timeType = reflect.TypeFor[time.Time]()

func schemaOf(t reflect.Type, seen map[reflect.Type]bool) (*Schema, error) {
	if t.Kind() == reflect.Pointer {
		s, err := schemaOf(t.Elem(), seen)
		if s != nil {
			s.nullable = true
		}
		return s, err
	}
	if t == timeType {
		return &Schema{Type: "string", Format: "date-time"}, nil
	}
	switch t.Kind() {
	case reflect.String:
		return &Schema{Type: "string"}, nil
	case reflect.Bool:
		return &Schema{Type: "boolean"}, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return &Schema{Type: "integer"}, nil
	case reflect.Float32, reflect.Float64:
		return &Schema{Type: "number"}, nil
	case reflect.Slice, reflect.Array:
		items, err := schemaOf(t.Elem(), seen)
		if err != nil {
			return nil, err
		}
		return &Schema{Type: "array", Items: items}, nil
	case reflect.Map:
		if t.Key().Kind() != reflect.String {
			return nil, fmt.Errorf("structllm: map key must be string, got %s", t.Key())
		}
		val, err := schemaOf(t.Elem(), seen)
		if err != nil {
			return nil, err
		}
		return &Schema{Type: "object", AdditionalProperties: val}, nil
	case reflect.Interface:
		return &Schema{}, nil // any JSON value
	case reflect.Struct:
		return structSchema(t, seen)
	default:
		return nil, fmt.Errorf("structllm: unsupported type %s", t)
	}
}

func structSchema(t reflect.Type, seen map[reflect.Type]bool) (*Schema, error) {
	if seen[t] {
		return nil, fmt.Errorf("structllm: recursive type %s is not supported", t)
	}
	seen[t] = true
	defer delete(seen, t)

	s := &Schema{Type: "object", Properties: map[string]*Schema{}, AdditionalProperties: false}
	if err := addFields(s, t, seen); err != nil {
		return nil, err
	}
	return s, nil
}

func addFields(s *Schema, t reflect.Type, seen map[reflect.Type]bool) error {
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if !f.IsExported() {
			continue
		}
		name, omitempty, skip := jsonName(f)
		if skip {
			continue
		}
		// Embedded structs without a json name are flattened, like encoding/json.
		if f.Anonymous && f.Tag.Get("json") == "" {
			ft := f.Type
			if ft.Kind() == reflect.Pointer {
				ft = ft.Elem()
			}
			if ft.Kind() == reflect.Struct {
				if err := addFields(s, ft, seen); err != nil {
					return err
				}
				continue
			}
		}
		fs, err := schemaOf(f.Type, seen)
		if err != nil {
			return fmt.Errorf("%s.%s: %w", t.Name(), f.Name, err)
		}
		if err := applyTags(fs, f); err != nil {
			return fmt.Errorf("%s.%s: %w", t.Name(), f.Name, err)
		}
		if _, dup := s.Properties[name]; !dup {
			s.order = append(s.order, name)
		}
		s.Properties[name] = fs
		if !omitempty && f.Type.Kind() != reflect.Pointer {
			s.Required = append(s.Required, name)
		}
	}
	return nil
}

func jsonName(f reflect.StructField) (name string, omitempty, skip bool) {
	tag := f.Tag.Get("json")
	if tag == "-" {
		return "", false, true
	}
	parts := strings.Split(tag, ",")
	name = parts[0]
	if name == "" {
		name = f.Name
	}
	for _, p := range parts[1:] {
		if p == "omitempty" || p == "omitzero" {
			omitempty = true
		}
	}
	return name, omitempty, false
}

func applyTags(s *Schema, f reflect.StructField) error {
	target := s
	// Length/enum/pattern tags on a slice apply to its items, except the
	// item-count tags which apply to the array itself.
	if s.Type == "array" && s.Items != nil {
		target = s.Items
	}
	if d := f.Tag.Get("desc"); d != "" {
		s.Description = d
	}
	if v := f.Tag.Get("enum"); v != "" {
		for _, e := range strings.Split(v, ",") {
			e = strings.TrimSpace(e)
			if target.Type == "integer" || target.Type == "number" {
				n, err := strconv.ParseFloat(e, 64)
				if err != nil {
					return fmt.Errorf("enum value %q is not a number", e)
				}
				target.Enum = append(target.Enum, n)
			} else {
				target.Enum = append(target.Enum, e)
			}
		}
	}
	if v := f.Tag.Get("pattern"); v != "" {
		target.Pattern = v
	}
	if v := f.Tag.Get("format"); v != "" {
		target.Format = v
	}
	ints := []struct {
		tag string
		dst **int
	}{{"minLen", &target.MinLength}, {"maxLen", &target.MaxLength}, {"minItems", &s.MinItems}, {"maxItems", &s.MaxItems}}
	for _, it := range ints {
		if v := f.Tag.Get(it.tag); v != "" {
			n, err := strconv.Atoi(v)
			if err != nil || n < 0 {
				return fmt.Errorf("%s=%q must be a non-negative integer", it.tag, v)
			}
			*it.dst = &n
		}
	}
	floats := []struct {
		tag string
		dst **float64
	}{{"min", &target.Minimum}, {"max", &target.Maximum}}
	for _, it := range floats {
		if v := f.Tag.Get(it.tag); v != "" {
			n, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return fmt.Errorf("%s=%q must be a number", it.tag, v)
			}
			*it.dst = &n
		}
	}
	if target.Pattern != "" {
		if _, err := compilePattern(target.Pattern); err != nil {
			return fmt.Errorf("pattern %q: %w", target.Pattern, err)
		}
	}
	return nil
}
