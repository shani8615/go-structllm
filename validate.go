package structllm

import (
	"fmt"
	"math"
	"regexp"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

// Issue is one validation problem at a JSON path such as $.tags[2].
type Issue struct {
	Path    string `json:"path"`
	Message string `json:"message"`
}

func (i Issue) String() string { return i.Path + ": " + i.Message }

// ValidationError lists every issue found in a response.
type ValidationError struct {
	Issues []Issue
}

func (e *ValidationError) Error() string {
	parts := make([]string, len(e.Issues))
	for i, is := range e.Issues {
		parts[i] = is.String()
	}
	return "structllm: invalid response: " + strings.Join(parts, "; ")
}

var patternCache sync.Map // string -> *regexp.Regexp

func compilePattern(p string) (*regexp.Regexp, error) {
	if re, ok := patternCache.Load(p); ok {
		return re.(*regexp.Regexp), nil
	}
	re, err := regexp.Compile(p)
	if err != nil {
		return nil, err
	}
	patternCache.Store(p, re)
	return re, nil
}

// Validate checks a decoded JSON value (from json.Unmarshal into any)
// against s and returns all issues, sorted by path.
func (s *Schema) Validate(v any) []Issue {
	var out []Issue
	validate(s, v, "$", &out)
	sort.SliceStable(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out
}

func validate(s *Schema, v any, path string, out *[]Issue) {
	add := func(format string, args ...any) {
		*out = append(*out, Issue{Path: path, Message: fmt.Sprintf(format, args...)})
	}
	if s == nil || s.Type == "" && len(s.Enum) == 0 {
		return // any value allowed
	}
	if v == nil {
		if s.nullable {
			return
		}
		add("must not be null (expected %s)", s.Type)
		return
	}

	switch s.Type {
	case "object":
		obj, ok := v.(map[string]any)
		if !ok {
			add("expected object, got %s", jsonType(v))
			return
		}
		for _, name := range s.Required {
			if _, ok := obj[name]; !ok {
				*out = append(*out, Issue{Path: path + "." + name, Message: "is required"})
			}
		}
		keys := make([]string, 0, len(obj))
		for k := range obj {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			if ps, ok := s.Properties[k]; ok {
				validate(ps, obj[k], path+"."+k, out)
				continue
			}
			switch ap := s.AdditionalProperties.(type) {
			case *Schema:
				validate(ap, obj[k], path+"."+k, out)
			case bool:
				if !ap && s.Properties != nil {
					*out = append(*out, Issue{Path: path + "." + k, Message: "is not an allowed field"})
				}
			}
		}
	case "array":
		arr, ok := v.([]any)
		if !ok {
			add("expected array, got %s", jsonType(v))
			return
		}
		if s.MinItems != nil && len(arr) < *s.MinItems {
			add("must have at least %d items, has %d", *s.MinItems, len(arr))
		}
		if s.MaxItems != nil && len(arr) > *s.MaxItems {
			add("must have at most %d items, has %d", *s.MaxItems, len(arr))
		}
		for i, item := range arr {
			validate(s.Items, item, fmt.Sprintf("%s[%d]", path, i), out)
		}
	case "string":
		str, ok := v.(string)
		if !ok {
			add("expected string, got %s", jsonType(v))
			return
		}
		n := utf8.RuneCountInString(str)
		if s.MinLength != nil && n < *s.MinLength {
			add("must be at least %d characters, is %d", *s.MinLength, n)
		}
		if s.MaxLength != nil && n > *s.MaxLength {
			add("must be at most %d characters, is %d", *s.MaxLength, n)
		}
		if s.Pattern != "" {
			if re, err := compilePattern(s.Pattern); err == nil && !re.MatchString(str) {
				add("must match pattern %s", s.Pattern)
			}
		}
		if s.Format == "date-time" {
			if _, err := time.Parse(time.RFC3339, str); err != nil {
				add("must be an RFC 3339 date-time (e.g. 2026-01-02T15:04:05Z)")
			}
		}
	case "integer", "number":
		f, ok := v.(float64)
		if !ok {
			add("expected %s, got %s", s.Type, jsonType(v))
			return
		}
		if s.Type == "integer" && f != math.Trunc(f) {
			add("must be a whole number")
		}
		if s.Minimum != nil && f < *s.Minimum {
			add("must be >= %v", *s.Minimum)
		}
		if s.Maximum != nil && f > *s.Maximum {
			add("must be <= %v", *s.Maximum)
		}
	case "boolean":
		if _, ok := v.(bool); !ok {
			add("expected boolean, got %s", jsonType(v))
			return
		}
	}

	if len(s.Enum) > 0 && !slices.Contains(s.Enum, v) {
		vals := make([]string, len(s.Enum))
		for i, e := range s.Enum {
			vals[i] = fmt.Sprint(e)
		}
		add("must be one of [%s]", strings.Join(vals, ", "))
	}
}

func jsonType(v any) string {
	switch v.(type) {
	case nil:
		return "null"
	case map[string]any:
		return "object"
	case []any:
		return "array"
	case string:
		return "string"
	case float64:
		return "number"
	case bool:
		return "boolean"
	default:
		return fmt.Sprintf("%T", v)
	}
}
