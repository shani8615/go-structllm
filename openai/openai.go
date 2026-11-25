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
	"net/http"
	"regexp"
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
}

// New returns a Client for the given model.
func New(apiKey, model string) *Client {
	return &Client{APIKey: apiKey, Model: model}
}

// APIError is a non-2xx response.
type APIError struct {
	StatusCode int
	Type       string
	Code       string
	Message    string
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
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    any    `json:"code"`
	} `json:"error"`
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
	resp, err := hc.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("openai: request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return "", fmt.Errorf("openai: read response: %w", err)
	}

	var out chatResponse
	jsonErr := json.Unmarshal(raw, &out)
	if resp.StatusCode/100 != 2 {
		e := &APIError{StatusCode: resp.StatusCode, Message: http.StatusText(resp.StatusCode)}
		if jsonErr == nil && out.Error != nil {
			e.Message, e.Type, e.Code = out.Error.Message, out.Error.Type, fmt.Sprint(out.Error.Code)
		}
		return "", e
	}
	if jsonErr != nil {
		return "", fmt.Errorf("openai: decode response: %w", jsonErr)
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
