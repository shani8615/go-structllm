// Package structllm gets typed, validated Go values out of large language
// models.
//
// Describe the output you want as a Go struct. structllm turns it into a
// JSON Schema, asks the model for JSON, validates the reply against the
// schema (and your own Validate method, if any) and, when the reply is
// invalid, sends the exact problems back to the model so it can fix them —
// a bounded number of times. You get a T or a precise error, never a
// half-parsed value.
//
//	type Ticket struct {
//	    Severity string   `json:"severity" enum:"low,medium,high,critical"`
//	    Summary  string   `json:"summary" maxLen:"200"`
//	    Tags     []string `json:"tags" maxItems:"5"`
//	}
//	t, err := structllm.Extract[Ticket](ctx, client, "Triage this alert: ...")
//
// The model is any Client: use the openai subpackage for OpenAI-compatible
// APIs (OpenAI, Azure OpenAI, Ollama, vLLM, Groq, ...) or wrap your own SDK
// with ClientFunc.
package structllm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// Role of a chat message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Message is one chat turn.
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`
}

// Request is what structllm sends to a Client.
type Request struct {
	Messages []Message
	// Schema is the expected output. Clients whose provider supports
	// schema-constrained decoding should pass it on; others can ignore it,
	// because the schema is also included in the system prompt.
	Schema     *Schema
	SchemaName string
}

// Client is a chat model. It returns the assistant's text.
type Client interface {
	Complete(ctx context.Context, req Request) (string, error)
}

// ClientFunc adapts a function to Client.
type ClientFunc func(ctx context.Context, req Request) (string, error)

func (f ClientFunc) Complete(ctx context.Context, req Request) (string, error) { return f(ctx, req) }

// Validator lets T add business rules the schema can't express
// ("end must be after start"). Its error message is sent back to the model.
type Validator interface {
	Validate() error
}

type options struct {
	maxRepairs int
	system     string
	history    []Message
	onAttempt  func(Attempt)
}

// Option configures Extract.
type Option func(*options)

// MaxRepairs sets how many times an invalid reply is sent back for fixing
// (default 2, so at most 3 model calls).
func MaxRepairs(n int) Option { return func(o *options) { o.maxRepairs = max(n, 0) } }

// System adds instructions to the system prompt, before the schema.
func System(text string) Option { return func(o *options) { o.system = text } }

// History inserts earlier conversation turns before the prompt.
func History(msgs ...Message) Option { return func(o *options) { o.history = msgs } }

// OnAttempt is called after every model call, for logging and metrics.
// The raw reply may contain user data: log it with care.
func OnAttempt(f func(Attempt)) Option { return func(o *options) { o.onAttempt = f } }

// Attempt describes one model call.
type Attempt struct {
	Number int     // 1-based
	Raw    string  // the model's reply
	Issues []Issue // empty if the reply was valid
	Err    error   // client error, if any
}

// ErrNoJSON is reported when a reply contains no JSON value at all.
var ErrNoJSON = errors.New("structllm: no JSON found in model reply")

// ExhaustedError is returned when the reply is still invalid after all repairs.
type ExhaustedError struct {
	Attempts int
	Last     error // *ValidationError or a JSON error
	Raw      string
}

func (e *ExhaustedError) Error() string {
	return fmt.Sprintf("structllm: no valid response after %d attempts: %v", e.Attempts, e.Last)
}

func (e *ExhaustedError) Unwrap() error { return e.Last }

// Extract asks the model for a T and returns it once it passes validation.
func Extract[T any](ctx context.Context, c Client, prompt string, opts ...Option) (T, error) {
	var zero T
	o := options{maxRepairs: 2}
	for _, opt := range opts {
		opt(&o)
	}
	schema, err := SchemaFor[T]()
	if err != nil {
		return zero, err
	}
	name := schemaName[T]()
	sys, err := systemPrompt(o.system, schema)
	if err != nil {
		return zero, err
	}

	msgs := make([]Message, 0, 3+len(o.history)+2*o.maxRepairs)
	msgs = append(msgs, Message{Role: RoleSystem, Content: sys})
	msgs = append(msgs, o.history...)
	msgs = append(msgs, Message{Role: RoleUser, Content: prompt})

	var last error
	var raw string
	for attempt := 1; attempt <= o.maxRepairs+1; attempt++ {
		raw, err = c.Complete(ctx, Request{Messages: msgs, Schema: schema, SchemaName: name})
		if err != nil {
			if o.onAttempt != nil {
				o.onAttempt(Attempt{Number: attempt, Err: err})
			}
			return zero, fmt.Errorf("structllm: model call failed: %w", err)
		}
		v, perr := parse[T](raw, schema)
		if o.onAttempt != nil {
			o.onAttempt(Attempt{Number: attempt, Raw: raw, Issues: issuesOf(perr)})
		}
		if perr == nil {
			return v, nil
		}
		last = perr
		msgs = append(msgs,
			Message{Role: RoleAssistant, Content: raw},
			Message{Role: RoleUser, Content: repairPrompt(perr)})
	}
	return zero, &ExhaustedError{Attempts: o.maxRepairs + 1, Last: last, Raw: raw}
}

// Parse validates a reply you already have (for example from a streaming
// call) and decodes it into T.
func Parse[T any](reply string) (T, error) {
	var zero T
	schema, err := SchemaFor[T]()
	if err != nil {
		return zero, err
	}
	return parse[T](reply, schema)
}

func parse[T any](reply string, schema *Schema) (T, error) {
	var zero T
	js, ok := extractJSON(reply)
	if !ok {
		return zero, ErrNoJSON
	}
	var generic any
	if err := json.Unmarshal([]byte(js), &generic); err != nil {
		return zero, fmt.Errorf("structllm: reply is not valid JSON: %w", err)
	}
	if issues := schema.Validate(generic); len(issues) > 0 {
		return zero, &ValidationError{Issues: issues}
	}
	var v T
	if err := json.Unmarshal([]byte(js), &v); err != nil {
		return zero, &ValidationError{Issues: []Issue{{Path: "$", Message: err.Error()}}}
	}
	if val, ok := any(&v).(Validator); ok {
		if err := val.Validate(); err != nil {
			return zero, &ValidationError{Issues: []Issue{{Path: "$", Message: err.Error()}}}
		}
	} else if val, ok := any(v).(Validator); ok {
		if err := val.Validate(); err != nil {
			return zero, &ValidationError{Issues: []Issue{{Path: "$", Message: err.Error()}}}
		}
	}
	return v, nil
}

func issuesOf(err error) []Issue {
	var ve *ValidationError
	if errors.As(err, &ve) {
		return ve.Issues
	}
	if err != nil {
		return []Issue{{Path: "$", Message: err.Error()}}
	}
	return nil
}

func systemPrompt(extra string, s *Schema) (string, error) {
	js, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return "", err
	}
	var b strings.Builder
	if extra != "" {
		b.WriteString(strings.TrimSpace(extra))
		b.WriteString("\n\n")
	}
	b.WriteString("Respond with a single JSON value that conforms to this JSON Schema. ")
	b.WriteString("Output only the JSON: no prose, no Markdown, no code fences.\n\n")
	b.Write(js)
	return b.String(), nil
}

func repairPrompt(err error) string {
	var b strings.Builder
	b.WriteString("Your previous response was not valid. Fix these problems and reply with only the corrected JSON:\n")
	var ve *ValidationError
	switch {
	case errors.As(err, &ve):
		for _, is := range ve.Issues {
			b.WriteString("- ")
			b.WriteString(is.String())
			b.WriteByte('\n')
		}
	case errors.Is(err, ErrNoJSON):
		b.WriteString("- the response did not contain any JSON\n")
	default:
		b.WriteString("- ")
		b.WriteString(err.Error())
		b.WriteByte('\n')
	}
	return b.String()
}

func schemaName[T any]() string {
	var zero T
	n := fmt.Sprintf("%T", zero)
	if i := strings.LastIndexByte(n, '.'); i >= 0 {
		n = n[i+1:]
	}
	n = strings.Trim(n, "*[]")
	if n == "" {
		return "response"
	}
	return n
}

// extractJSON returns the first complete JSON object or array in s,
// tolerating code fences and surrounding prose.
func extractJSON(s string) (string, bool) {
	s = strings.TrimSpace(s)
	if json.Valid([]byte(s)) {
		return s, true
	}
	for i := 0; i < len(s); i++ {
		if s[i] != '{' && s[i] != '[' {
			continue
		}
		if end, ok := matchJSON(s, i); ok {
			cand := s[i : end+1]
			if json.Valid([]byte(cand)) {
				return cand, true
			}
		}
	}
	return "", false
}

// matchJSON finds the index closing the object/array opened at s[start],
// respecting strings and escapes.
func matchJSON(s string, start int) (int, bool) {
	depth := 0
	inStr, esc := false, false
	for i := start; i < len(s); i++ {
		c := s[i]
		if inStr {
			switch {
			case esc:
				esc = false
			case c == '\\':
				esc = true
			case c == '"':
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{', '[':
			depth++
		case '}', ']':
			depth--
			if depth == 0 {
				return i, true
			}
		}
	}
	return 0, false
}
