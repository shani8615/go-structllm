// Package openai is a minimal structllm.Client for OpenAI-compatible chat
// completion APIs: OpenAI, Azure OpenAI (with BaseURL set), Ollama, vLLM,
// LM Studio, Groq, Together and others. It uses only net/http.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/shani8615/go-structllm"
)

// ResponseFormat controls how the schema is passed to the API.
type ResponseFormat string

const (
	// FormatJSONSchema sends response_format=json_schema (OpenAI, recent
	// vLLM/Ollama). Best results where supported.
	FormatJSONSchema ResponseFormat = "json_schema"
	// FormatJSONObject sends response_format=json_object (older servers).
	FormatJSONObject ResponseFormat = "json_object"
	// FormatNone relies on the prompt alone.
	FormatNone ResponseFormat = "none"
)

// Client calls POST {BaseURL}/chat/completions.
type Client struct {
	BaseURL        string // default https://api.openai.com/v1
	APIKey         string // sent as a Bearer token; never logged or returned in errors
	Model          string
	Temperature    *float64
	ResponseFormat ResponseFormat // default FormatJSONSchema
	Header         http.Header    // extra headers, e.g. api-key for Azure
	HTTPClient     *http.Client   // default: 60s timeout

	// Temporary failures (429, 5xx, network errors) are retried with
	// exponential backoff and jitter, honouring Retry-After.
	MaxRetries     int           // default 2; ignored when DisableRetries is set
	DisableRetries bool          // turn retries off entirely
	RetryBaseDelay time.Duration // first backoff; default 500ms, doubled per retry
	MaxRetryWait   time.Duration // never wait longer than this per retry; default 30s
}

// New returns a Client for the given model.
func New(apiKey, model string) *Client {
	return &Client{APIKey: apiKey, Model: model}
}

// APIError is a non-2xx response.
type APIError struct {
	StatusCode int
	Type       string // error type, or the provider's status such as "UNAVAILABLE"
	Code       string
	Message    string
	RetryAfter time.Duration // from the Retry-After header, if any
}

func (e *APIError) Error() string {
	return fmt.Sprintf("openai: HTTP %d: %s", e.StatusCode, e.Message)
}

// Temporary reports whether retrying may help (rate limits, server errors).
func (e *APIError) Temporary() bool {
	return e.StatusCode == http.StatusTooManyRequests || e.StatusCode >= 500
}

type chatRequest struct {
	Model          string          `json:"model"`
	Messages       []chatMessage   `json:"messages"`
	Temperature    *float64        `json:"temperature,omitempty"`
	ResponseFormat *responseFormat `json:"response_format,omitempty"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type responseFormat struct {
	Type       string      `json:"type"`
	JSONSchema *jsonSchema `json:"json_schema,omitempty"`
}

type jsonSchema struct {
	Name   string            `json:"name"`
	Schema *structllm.Schema `json:"schema"`
	Strict bool              `json:"strict"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
			Refusal string `json:"refusal"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
}

// ErrRefused is returned when the model declines to answer.
var ErrRefused = errors.New("openai: model refused the request")

// ErrTruncated is returned when the reply hit the token limit.
var ErrTruncated = errors.New("openai: reply truncated (finish_reason=length)")

var nameRe = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// Complete implements structllm.Client.
func (c *Client) Complete(ctx context.Context, req structllm.Request) (string, error) {
	if c.Model == "" {
		return "", errors.New("openai: Model is required")
	}
	body := chatRequest{Model: c.Model, Temperature: c.Temperature}
	for _, m := range req.Messages {
		body.Messages = append(body.Messages, chatMessage{Role: string(m.Role), Content: m.Content})
	}
	switch c.ResponseFormat {
	case "", FormatJSONSchema:
		if req.Schema != nil {
			name := nameRe.ReplaceAllString(req.SchemaName, "_")
			if name == "" {
				name = "response"
			}
			body.ResponseFormat = &responseFormat{Type: "json_schema", JSONSchema: &jsonSchema{Name: name, Schema: req.Schema}}
		}
	case FormatJSONObject:
		body.ResponseFormat = &responseFormat{Type: "json_object"}
	case FormatNone:
	default:
		return "", fmt.Errorf("openai: unknown ResponseFormat %q", c.ResponseFormat)
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return "", err
	}
	base := strings.TrimRight(c.BaseURL, "/")
	if base == "" {
		base = "https://api.openai.com/v1"
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return "", err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
	for k, vs := range c.Header {
		for _, v := range vs {
			httpReq.Header.Add(k, v)
		}
	}
	hc := c.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: 60 * time.Second}
	}

	retries := c.MaxRetries
	if retries <= 0 {
		retries = 2
	}
	if c.DisableRetries {
		retries = 0
	}
	for attempt := 0; ; attempt++ {
		content, err := c.do(ctx, hc, base, payload)
		if err == nil || attempt >= retries || !retryable(ctx, err) {
			return content, err
		}
		wait := c.backoff(attempt, err)
		if wait < 0 {
			return "", err // server asked us to wait longer than MaxRetryWait
		}
		t := time.NewTimer(wait)
		select {
		case <-ctx.Done():
			t.Stop()
			return "", err
		case <-t.C:
		}
	}
}

func (c *Client) do(ctx context.Context, hc *http.Client, base string, payload []byte) (string, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return "", err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
	for k, vs := range c.Header {
		for _, v := range vs {
			httpReq.Header.Add(k, v)
		}
	}
	resp, err := hc.Do(httpReq)
	if err != nil {
		return "", &transportError{err}
	}
	defer func() { _ = resp.Body.Close() }()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return "", &transportError{err}
	}
	if resp.StatusCode/100 != 2 {
		return "", parseAPIError(resp, raw)
	}
	var out chatResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", fmt.Errorf("openai: decode response: %w", err)
	}
	if len(out.Choices) == 0 {
		return "", errors.New("openai: response has no choices")
	}
	ch := out.Choices[0]
	if ch.Message.Refusal != "" {
		return "", fmt.Errorf("%w: %s", ErrRefused, ch.Message.Refusal)
	}
	if ch.FinishReason == "length" {
		return "", ErrTruncated
	}
	return ch.Message.Content, nil
}

// transportError is a network-level failure (connection refused/reset, timeout).
type transportError struct{ err error }

func (e *transportError) Error() string { return "openai: request failed: " + e.err.Error() }
func (e *transportError) Unwrap() error { return e.err }

func retryable(ctx context.Context, err error) bool {
	if ctx.Err() != nil {
		return false
	}
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.Temporary()
	}
	var te *transportError
	return errors.As(err, &te)
}

// backoff returns how long to wait before retry number attempt+1, or -1 if
// the server's Retry-After exceeds MaxRetryWait.
func (c *Client) backoff(attempt int, err error) time.Duration {
	maxWait := c.MaxRetryWait
	if maxWait <= 0 {
		maxWait = 30 * time.Second
	}
	var apiErr *APIError
	if errors.As(err, &apiErr) && apiErr.RetryAfter > 0 {
		if apiErr.RetryAfter > maxWait {
			return -1
		}
		return apiErr.RetryAfter
	}
	base := c.RetryBaseDelay
	if base <= 0 {
		base = 500 * time.Millisecond
	}
	d := min(base<<attempt, maxWait)
	return d/2 + rand.N(d/2+1) // jitter
}

// errorBody is the error envelope. OpenAI and most compatible servers send
// {"error":{...}}; Gemini's OpenAI-compatible endpoint sends [{"error":{...}}].
type errorBody struct {
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Status  string `json:"status"` // Gemini: "NOT_FOUND", "UNAVAILABLE", ...
		Code    any    `json:"code"`
	} `json:"error"`
}

func parseAPIError(resp *http.Response, raw []byte) *APIError {
	e := &APIError{StatusCode: resp.StatusCode, Message: http.StatusText(resp.StatusCode)}
	e.RetryAfter = parseRetryAfter(resp.Header.Get("Retry-After"))

	var body errorBody
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) > 0 && trimmed[0] == '[' {
		var arr []errorBody
		if json.Unmarshal(trimmed, &arr) == nil && len(arr) > 0 {
			body = arr[0]
		}
	} else if json.Unmarshal(trimmed, &body) != nil {
		// Some servers send {"error":"plain text"}.
		var s struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(trimmed, &s) == nil && s.Error != "" {
			e.Message = s.Error
		}
		return e
	}
	if body.Error != nil {
		if body.Error.Message != "" {
			e.Message = body.Error.Message
		}
		e.Type = body.Error.Type
		if e.Type == "" {
			e.Type = body.Error.Status
		}
		if body.Error.Code != nil {
			e.Code = fmt.Sprint(body.Error.Code)
		}
	}
	return e
}

// parseRetryAfter accepts delay-seconds or an HTTP date.
func parseRetryAfter(v string) time.Duration {
	if v == "" {
		return 0
	}
	if secs, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && secs >= 0 {
		return time.Duration(secs) * time.Second
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := time.Until(t); d > 0 {
			return d
		}
	}
	return 0
}
