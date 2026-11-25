package structllm

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

type Ticket struct {
	Severity string   `json:"severity" enum:"low,medium,high,critical" desc:"How urgent the issue is"`
	Summary  string   `json:"summary" minLen:"1" maxLen:"40"`
	Tags     []string `json:"tags" maxItems:"3" pattern:"^[a-z-]+$"`
	Score    int      `json:"score" min:"0" max:"10"`
	Owner    *string  `json:"owner"`
	Note     string   `json:"note,omitempty"`
}

type Window struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

func (w Window) Validate() error {
	if !w.End.After(w.Start) {
		return errors.New("end must be after start")
	}
	return nil
}

// scripted returns replies in order and records every request.
type scripted struct {
	replies []string
	reqs    []Request
}

func (s *scripted) Complete(_ context.Context, r Request) (string, error) {
	s.reqs = append(s.reqs, r)
	if len(s.replies) == 0 {
		return "", errors.New("no more replies")
	}
	out := s.replies[0]
	s.replies = s.replies[1:]
	return out, nil
}

func TestSchemaFor(t *testing.T) {
	s, err := SchemaFor[Ticket]()
	if err != nil {
		t.Fatal(err)
	}
	js, _ := json.Marshal(s)
	got := string(js)
	for _, want := range []string{
		`"properties":{"severity":`, // declaration order kept
		`"enum":["low","medium","high","critical"]`,
		`"description":"How urgent the issue is"`,
		`"maxItems":3`, `"pattern":"^[a-z-]+$"`,
		`"required":["severity","summary","tags","score"]`, // pointer + omitempty optional
		`"additionalProperties":false`,
		`"owner":{"type":["string","null"]}`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("schema missing %s\n%s", want, got)
		}
	}
	if strings.Index(got, `"summary"`) > strings.Index(got, `"tags"`) {
		t.Error("property order not preserved")
	}
}

func TestSchemaRejectsRecursiveTypes(t *testing.T) {
	type Node struct {
		Children []Node `json:"children"`
	}
	if _, err := SchemaFor[Node](); err == nil {
		t.Fatal("recursive type must be rejected")
	}
}

func TestExtractFirstTry(t *testing.T) {
	c := &scripted{replies: []string{"Sure! ```json\n{\"severity\":\"high\",\"summary\":\"DB down\",\"tags\":[\"db\"],\"score\":9,\"owner\":null}\n```"}}
	got, err := Extract[Ticket](context.Background(), c, "triage")
	if err != nil {
		t.Fatal(err)
	}
	if got.Severity != "high" || got.Score != 9 || got.Owner != nil {
		t.Fatalf("got %+v", got)
	}
	if len(c.reqs) != 1 || c.reqs[0].Schema == nil || c.reqs[0].SchemaName != "Ticket" {
		t.Fatalf("requests=%+v", c.reqs)
	}
	if !strings.Contains(c.reqs[0].Messages[0].Content, `"severity"`) {
		t.Fatal("schema must be in the system prompt")
	}
}

func TestExtractRepairsWithExactIssues(t *testing.T) {
	c := &scripted{replies: []string{
		`{"severity":"urgent","summary":"DB down","tags":["DB"],"score":11,"extra":1}`,
		`{"severity":"critical","summary":"DB down","tags":["db"],"score":10}`,
	}}
	var attempts []Attempt
	got, err := Extract[Ticket](context.Background(), c, "triage", OnAttempt(func(a Attempt) { attempts = append(attempts, a) }))
	if err != nil {
		t.Fatal(err)
	}
	if got.Severity != "critical" || len(c.reqs) != 2 || len(attempts) != 2 {
		t.Fatalf("got=%+v calls=%d", got, len(c.reqs))
	}
	repair := c.reqs[1].Messages[len(c.reqs[1].Messages)-1].Content
	for _, want := range []string{
		"$.severity: must be one of [low, medium, high, critical]",
		"$.tags[0]: must match pattern",
		"$.score: must be <= 10",
		"$.extra: is not an allowed field",
	} {
		if !strings.Contains(repair, want) {
			t.Errorf("repair prompt missing %q:\n%s", want, repair)
		}
	}
	if c.reqs[1].Messages[len(c.reqs[1].Messages)-2].Role != RoleAssistant {
		t.Error("previous reply must be in the conversation")
	}
}

func TestExtractGivesUp(t *testing.T) {
	c := &scripted{replies: []string{"no idea", "still no", "nope"}}
	_, err := Extract[Ticket](context.Background(), c, "triage", MaxRepairs(2))
	var ex *ExhaustedError
	if !errors.As(err, &ex) || ex.Attempts != 3 || !errors.Is(err, ErrNoJSON) {
		t.Fatalf("err=%v", err)
	}
}

func TestClientErrorIsNotRetried(t *testing.T) {
	boom := errors.New("rate limited")
	calls := 0
	c := ClientFunc(func(context.Context, Request) (string, error) { calls++; return "", boom })
	if _, err := Extract[Ticket](context.Background(), c, "x"); !errors.Is(err, boom) || calls != 1 {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
}

func TestCustomValidator(t *testing.T) {
	c := &scripted{replies: []string{
		`{"start":"2026-01-02T10:00:00Z","end":"2026-01-02T09:00:00Z"}`,
		`{"start":"2026-01-02T10:00:00Z","end":"2026-01-02T11:00:00Z"}`,
	}}
	w, err := Extract[Window](context.Background(), c, "when?")
	if err != nil || w.End.Hour() != 11 {
		t.Fatalf("w=%+v err=%v", w, err)
	}
	if !strings.Contains(c.reqs[1].Messages[len(c.reqs[1].Messages)-1].Content, "end must be after start") {
		t.Fatal("Validate() error must be sent back to the model")
	}
}

func TestParseFormatsAndTypes(t *testing.T) {
	if _, err := Parse[Window](`{"start":"yesterday","end":"2026-01-02T11:00:00Z"}`); err == nil ||
		!strings.Contains(err.Error(), "$.start: must be an RFC 3339 date-time") {
		t.Fatalf("err=%v", err)
	}
	if _, err := Parse[Ticket](`{"severity":"low","summary":"x","tags":[],"score":1.5}`); err == nil ||
		!strings.Contains(err.Error(), "$.score: must be a whole number") {
		t.Fatalf("err=%v", err)
	}
	if _, err := Parse[Ticket](`{"severity":"low","summary":"x","tags":[],"score":1}`); err != nil {
		t.Fatalf("valid reply rejected: %v", err)
	}
}

func TestExtractJSON(t *testing.T) {
	cases := map[string]string{
		`{"a":1}`:                            `{"a":1}`,
		"Here you go:\n```json\n[1,2]\n```":  `[1,2]`,
		`prefix {"s":"has } brace"} suffix`:  `{"s":"has } brace"}`,
		`{broken {"ok":true}`:                `{"ok":true}`,
		`text with "quote\" {" then {"x":1}`: `{"x":1}`,
	}
	for in, want := range cases {
		if got, ok := extractJSON(in); !ok || got != want {
			t.Errorf("extractJSON(%q) = %q,%v want %q", in, got, ok, want)
		}
	}
	if _, ok := extractJSON("no json here"); ok {
		t.Error("expected no JSON")
	}
}
