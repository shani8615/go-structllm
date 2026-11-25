package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

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
	c := &Client{BaseURL: srv.URL, APIKey: "super-secret-key", Model: "m", DisableRetries: true}
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
		c := &Client{BaseURL: srv.URL, Model: "m", ResponseFormat: FormatNone, Header: http.Header{"api-key": {"azure-key"}}, RetryBaseDelay: time.Millisecond}
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
	if _, err := (&Client{BaseURL: "http://127.0.0.1:1", Model: "m", RetryBaseDelay: time.Millisecond}).Complete(ctx, structllm.Request{}); err == nil {
		t.Fatal("unreachable server must fail")
	}
}

// Regression fixtures: real error messages from Gemini's OpenAI-compatible
// endpoint, which wraps errors in a JSON array. They were previously
// reported as the bare status text ("Not Found", "Service Unavailable").
const (
	geminiModelRetired = `[{"error":{"code":404,"message":"This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.8-flash ...","status":"NOT_FOUND"}}]`
	geminiHighDemand   = `[{"error":{"code":503,"message":"This model is currently experiencing high demand ... Please try again later.","status":"UNAVAILABLE"}}]`
)

func serveStatus(t *testing.T, status int, body string, hits *atomic.Int32) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestGeminiArrayErrorBodies(t *testing.T) {
	for _, tc := range []struct {
		status    int
		body      string
		wantMsg   string
		wantType  string
		temporary bool
	}{
		{404, geminiModelRetired, "is no longer available to new users", "NOT_FOUND", false},
		{503, geminiHighDemand, "currently experiencing high demand", "UNAVAILABLE", true},
	} {
		var hits atomic.Int32
		srv := serveStatus(t, tc.status, tc.body, &hits)
		c := &Client{BaseURL: srv.URL, Model: "m", DisableRetries: true}
		_, err := c.Complete(context.Background(), structllm.Request{})
		var apiErr *APIError
		if !errors.As(err, &apiErr) {
			t.Fatalf("%d: err=%v", tc.status, err)
		}
		if !strings.Contains(apiErr.Message, tc.wantMsg) || apiErr.Type != tc.wantType ||
			apiErr.Code != fmt.Sprint(tc.status) || apiErr.Temporary() != tc.temporary {
			t.Errorf("%d: got %+v", tc.status, apiErr)
		}
		if !strings.Contains(err.Error(), tc.wantMsg) {
			t.Errorf("%d: provider message must appear in the error, got %q", tc.status, err)
		}
	}
}

func TestRetriesTemporaryErrorsThenSucceeds(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if hits.Add(1) <= 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, geminiHighDemand)
			return
		}
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()
	c := &Client{BaseURL: srv.URL, Model: "m", RetryBaseDelay: time.Millisecond}
	got, err := c.Complete(context.Background(), structllm.Request{})
	if err != nil || got != "{}" || hits.Load() != 3 {
		t.Fatalf("got=%q err=%v hits=%d", got, err, hits.Load())
	}
}

func TestDoesNotRetryPermanentErrors(t *testing.T) {
	var hits atomic.Int32
	srv := serveStatus(t, 404, geminiModelRetired, &hits)
	c := &Client{BaseURL: srv.URL, Model: "m", RetryBaseDelay: time.Millisecond}
	if _, err := c.Complete(context.Background(), structllm.Request{}); err == nil || hits.Load() != 1 {
		t.Fatalf("404 must not be retried: err=%v hits=%d", err, hits.Load())
	}
}

func TestRetryLimitAndDisable(t *testing.T) {
	var hits atomic.Int32
	srv := serveStatus(t, 503, geminiHighDemand, &hits)
	c := &Client{BaseURL: srv.URL, Model: "m", MaxRetries: 3, RetryBaseDelay: time.Millisecond}
	if _, err := c.Complete(context.Background(), structllm.Request{}); err == nil || hits.Load() != 4 {
		t.Fatalf("want 1 call + 3 retries, got %d (err=%v)", hits.Load(), err)
	}
	hits.Store(0)
	c.DisableRetries = true
	_, _ = c.Complete(context.Background(), structllm.Request{})
	if hits.Load() != 1 {
		t.Fatalf("DisableRetries: %d calls", hits.Load())
	}
}

func TestHonoursRetryAfter(t *testing.T) {
	var hits atomic.Int32
	var first time.Time
	var gap time.Duration
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		switch hits.Add(1) {
		case 1:
			first = time.Now()
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
		default:
			gap = time.Since(first)
			_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
		}
	}))
	defer srv.Close()
	c := &Client{BaseURL: srv.URL, Model: "m", RetryBaseDelay: time.Millisecond}
	if _, err := c.Complete(context.Background(), structllm.Request{}); err != nil {
		t.Fatal(err)
	}
	if gap < 900*time.Millisecond {
		t.Fatalf("Retry-After: 1 ignored, retried after %v", gap)
	}

	// A Retry-After longer than MaxRetryWait is not waited out.
	var hits2 atomic.Int32
	srv2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits2.Add(1)
		w.Header().Set("Retry-After", "3600")
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv2.Close()
	start := time.Now()
	_, err := (&Client{BaseURL: srv2.URL, Model: "m", MaxRetryWait: time.Second}).Complete(context.Background(), structllm.Request{})
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.RetryAfter != time.Hour || hits2.Load() != 1 || time.Since(start) > time.Second {
		t.Fatalf("err=%v hits=%d", err, hits2.Load())
	}
}

func TestRetryStopsWhenContextEnds(t *testing.T) {
	var hits atomic.Int32
	srv := serveStatus(t, 503, geminiHighDemand, &hits)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	c := &Client{BaseURL: srv.URL, Model: "m", RetryBaseDelay: time.Second}
	start := time.Now()
	if _, err := c.Complete(ctx, structllm.Request{}); err == nil || time.Since(start) > 500*time.Millisecond {
		t.Fatalf("must give up when ctx ends: err=%v after %v", err, time.Since(start))
	}
}

func TestParseRetryAfter(t *testing.T) {
	if parseRetryAfter("7") != 7*time.Second || parseRetryAfter("") != 0 || parseRetryAfter("soon") != 0 {
		t.Fatal("seconds form")
	}
	if d := parseRetryAfter(time.Now().Add(5 * time.Second).UTC().Format(http.TimeFormat)); d < 3*time.Second || d > 6*time.Second {
		t.Fatalf("http-date form: %v", d)
	}
	rec := httptest.NewRecorder()
	rec.WriteHeader(502)
	if got := parseAPIError(rec.Result(), []byte(`{"error":"bad gateway upstream"}`)); got.Message != "bad gateway upstream" {
		t.Fatalf("string error body: %+v", got)
	}
}
