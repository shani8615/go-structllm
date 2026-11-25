package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/shani8615/go-structllm"
)

type Sentiment struct {
	Label string  `json:"label" enum:"positive,negative,neutral"`
	Score float64 `json:"score" min:"0" max:"1"`
}

func TestEndToEndWithRepair(t *testing.T) {
	var bodies []map[string]any
	replies := []string{`{"label":"great","score":0.9}`, `{"label":"positive","score":0.9}`}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("path=%s auth=%q", r.URL.Path, r.Header.Get("Authorization"))
		}
		var b map[string]any
		_ = json.NewDecoder(r.Body).Decode(&b)
		bodies = append(bodies, b)
		reply := replies[0]
		replies = replies[1:]
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": map[string]any{"content": reply}, "finish_reason": "stop"}},
		})
	}))
	defer srv.Close()

	c := &Client{BaseURL: srv.URL + "/v1", APIKey: "test-key", Model: "gpt-test"}
	got, err := structllm.Extract[Sentiment](context.Background(), c, "I love it")
	if err != nil {
		t.Fatal(err)
	}
	if got.Label != "positive" || len(bodies) != 2 {
		t.Fatalf("got=%+v calls=%d", got, len(bodies))
	}
	rf := bodies[0]["response_format"].(map[string]any)
	js := rf["json_schema"].(map[string]any)
	if rf["type"] != "json_schema" || js["name"] != "Sentiment" || js["schema"] == nil {
		t.Fatalf("response_format=%v", rf)
	}
	if n := len(bodies[1]["messages"].([]any)); n != 4 {
		t.Fatalf("repair call should carry system, user, assistant, user messages; got %d", n)
	}
}

func TestAPIErrorDoesNotLeakKey(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(429)
		_, _ = io.WriteString(w, `{"error":{"message":"Rate limit reached","type":"requests","code":"rate_limit_exceeded"}}`)
	}))
	defer srv.Close()
	c := &Client{BaseURL: srv.URL, APIKey: "super-secret-key", Model: "m"}
	_, err := c.Complete(context.Background(), structllm.Request{Messages: []structllm.Message{{Role: "user", Content: "hi"}}})
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 429 || !apiErr.Temporary() || apiErr.Code != "rate_limit_exceeded" {
		t.Fatalf("err=%v", err)
	}
	if strings.Contains(err.Error(), "super-secret-key") {
		t.Fatal("API key leaked into error")
	}
}

func TestRefusalAndTruncation(t *testing.T) {
	for name, tc := range map[string]struct {
		body string
		want error
	}{
		"refusal":  {`{"choices":[{"message":{"content":"","refusal":"I can't help with that"},"finish_reason":"stop"}]}`, ErrRefused},
		"truncate": {`{"choices":[{"message":{"content":"{\"label\":"},"finish_reason":"length"}]}`, ErrTruncated},
	} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, tc.body) }))
		c := &Client{BaseURL: srv.URL, Model: "m"}
		_, err := c.Complete(context.Background(), structllm.Request{})
		srv.Close()
		if !errors.Is(err, tc.want) {
			t.Errorf("%s: err=%v", name, err)
		}
	}
}

func TestJSONObjectFormat(t *testing.T) {
	var rf map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var b map[string]any
		_ = json.NewDecoder(r.Body).Decode(&b)
		rf, _ = b["response_format"].(map[string]any)
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()
	c := &Client{BaseURL: srv.URL, Model: "m", ResponseFormat: FormatJSONObject}
	_, _ = c.Complete(context.Background(), structllm.Request{Schema: &structllm.Schema{Type: "object"}})
	if rf["type"] != "json_object" || rf["json_schema"] != nil {
		t.Fatalf("response_format=%v", rf)
	}
}

func TestClientErrorsAndFormats(t *testing.T) {
	ctx := context.Background()
	if _, err := (&Client{}).Complete(ctx, structllm.Request{}); err == nil || !strings.Contains(err.Error(), "Model is required") {
		t.Fatalf("missing model: %v", err)
	}
	if _, err := (&Client{Model: "m", ResponseFormat: "yaml"}).Complete(ctx, structllm.Request{}); err == nil {
		t.Fatal("unknown format must fail")
	}
	var gotFormat any = "unset"
	var gotHeader string
	for name, body := range map[string]string{
		"no choices": `{"choices":[]}`,
		"bad json":   `not json`,
		"plain 500":  ``,
	} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var b map[string]any
			_ = json.NewDecoder(r.Body).Decode(&b)
			gotFormat = b["response_format"]
			gotHeader = r.Header.Get("api-key")
			if name == "plain 500" {
				w.WriteHeader(500)
			}
			_, _ = io.WriteString(w, body)
		}))
		c := &Client{BaseURL: srv.URL, Model: "m", ResponseFormat: FormatNone, Header: http.Header{"api-key": {"azure-key"}}}
		_, err := c.Complete(ctx, structllm.Request{Schema: &structllm.Schema{Type: "object"}})
		srv.Close()
		if err == nil {
			t.Errorf("%s: expected error", name)
		}
		if name == "plain 500" {
			var apiErr *APIError
			if !errors.As(err, &apiErr) || apiErr.Message != "Internal Server Error" {
				t.Errorf("plain 500: %v", err)
			}
		}
	}
	if gotFormat != nil || gotHeader != "azure-key" {
		t.Fatalf("FormatNone must omit response_format (got %v); extra headers must be sent (got %q)", gotFormat, gotHeader)
	}
	if _, err := (&Client{BaseURL: "http://127.0.0.1:1", Model: "m"}).Complete(ctx, structllm.Request{}); err == nil {
		t.Fatal("unreachable server must fail")
	}
}
