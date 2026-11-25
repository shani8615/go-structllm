package structllm

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

type Base struct {
	ID string `json:"id" desc:"stable identifier"`
}

type Everything struct {
	Base                       // embedded: flattened
	Count    int64             `json:"count" enum:"1,2,3"`
	Ratio    float32           `json:"ratio" min:"0" max:"1"`
	OK       bool              `json:"ok"`
	Labels   map[string]string `json:"labels"`
	Scores   map[string]int    `json:"scores,omitempty"`
	Extra    any               `json:"extra,omitempty"`
	When     time.Time         `json:"when"`
	Codes    []string          `json:"codes" minItems:"1" minLen:"2" format:"iata"`
	Grid     [2]int            `json:"grid"`
	Skipped  string            `json:"-"`
	internal string            //nolint:unused // unexported fields are ignored
	NoTag    string
}

func TestSchemaCoversAllKinds(t *testing.T) {
	s, err := SchemaFor[Everything]()
	if err != nil {
		t.Fatal(err)
	}
	p := s.Properties
	checks := map[string]string{
		"id": "string", "count": "integer", "ratio": "number", "ok": "boolean", "labels": "object",
		"when": "string", "codes": "array", "grid": "array", "NoTag": "string",
	}
	for name, typ := range checks {
		if p[name] == nil || p[name].Type != typ {
			t.Errorf("%s: got %+v want type %s", name, p[name], typ)
		}
	}
	if _, ok := p["Skipped"]; ok {
		t.Error(`json:"-" field must be skipped`)
	}
	if _, ok := p["internal"]; ok {
		t.Error("unexported field must be skipped")
	}
	if p["extra"].Type != "" {
		t.Error("interface field accepts any JSON")
	}
	if p["when"].Format != "date-time" || p["codes"].Items.Format != "iata" || *p["codes"].Items.MinLength != 2 {
		t.Errorf("format/minLen not applied: %+v %+v", p["when"], p["codes"].Items)
	}
	if p["count"].Enum[0] != float64(1) {
		t.Errorf("numeric enum: %v", p["count"].Enum)
	}
	for _, r := range s.Required {
		if r == "scores" || r == "extra" {
			t.Errorf("%s is omitempty and must not be required", r)
		}
	}

	valid := `{"id":"x","count":2,"ratio":0.5,"ok":true,"labels":{"a":"b"},"when":"2026-01-02T03:04:05Z","codes":["KHI"],"grid":[1,2],"NoTag":"n"}`
	var v any
	_ = json.Unmarshal([]byte(valid), &v)
	if issues := s.Validate(v); len(issues) != 0 {
		t.Fatalf("valid object rejected: %v", issues)
	}
	bad := `{"id":1,"count":5,"ratio":2,"ok":"yes","labels":{"a":1},"when":"2026-01-02T03:04:05Z","codes":[],"grid":"no","NoTag":null}`
	_ = json.Unmarshal([]byte(bad), &v)
	got := map[string]bool{}
	for _, is := range s.Validate(v) {
		got[is.Path] = true
	}
	for _, path := range []string{"$.id", "$.count", "$.ratio", "$.ok", "$.labels.a", "$.codes", "$.grid", "$.NoTag"} {
		if !got[path] {
			t.Errorf("expected an issue at %s, got %v", path, got)
		}
	}
	if issues := s.Validate([]any{}); len(issues) != 1 || !strings.Contains(issues[0].Message, "expected object, got array") {
		t.Errorf("top-level type mismatch: %v", issues)
	}
}

func TestSchemaTagErrors(t *testing.T) {
	type BadEnum struct {
		N int `json:"n" enum:"one"`
	}
	type BadLen struct {
		S string `json:"s" maxLen:"-1"`
	}
	type BadMin struct {
		F float64 `json:"f" min:"low"`
	}
	type BadPattern struct {
		S string `json:"s" pattern:"("`
	}
	type BadMap struct {
		M map[int]string `json:"m"`
	}
	type BadKind struct {
		C chan int `json:"c"`
	}
	for name, f := range map[string]func() error{
		"enum":    func() error { _, err := SchemaFor[BadEnum](); return err },
		"len":     func() error { _, err := SchemaFor[BadLen](); return err },
		"min":     func() error { _, err := SchemaFor[BadMin](); return err },
		"pattern": func() error { _, err := SchemaFor[BadPattern](); return err },
		"map":     func() error { _, err := SchemaFor[BadMap](); return err },
		"kind":    func() error { _, err := SchemaFor[BadKind](); return err },
	} {
		if f() == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	if _, err := Parse[BadKind](`{}`); err == nil {
		t.Error("Parse must surface schema errors")
	}
}

func TestSystemAndHistoryOptions(t *testing.T) {
	c := &scripted{replies: []string{`{"severity":"low","summary":"ok","tags":[],"score":0}`}}
	_, err := Extract[Ticket](context.Background(), c, "now",
		System("You are a triage bot."),
		History(Message{Role: RoleUser, Content: "earlier"}, Message{Role: RoleAssistant, Content: "noted"}))
	if err != nil {
		t.Fatal(err)
	}
	m := c.reqs[0].Messages
	if !strings.HasPrefix(m[0].Content, "You are a triage bot.") || m[1].Content != "earlier" || m[3].Content != "now" {
		t.Fatalf("messages=%+v", m)
	}
}

func TestRepairPromptForBrokenJSON(t *testing.T) {
	c := &scripted{replies: []string{`{"severity":"low",`, `{"severity":"low","summary":"ok","tags":[],"score":0}`}}
	if _, err := Extract[Ticket](context.Background(), c, "x"); err != nil {
		t.Fatal(err)
	}
	last := c.reqs[1].Messages[len(c.reqs[1].Messages)-1].Content
	if !strings.Contains(last, "did not contain any JSON") {
		t.Fatalf("repair prompt: %s", last)
	}
}

type ptrValidated struct {
	N int `json:"n"`
}

func (p *ptrValidated) Validate() error {
	if p.N%2 != 0 {
		return errors.New("n must be even")
	}
	return nil
}

func TestPointerReceiverValidatorAndErrorText(t *testing.T) {
	_, err := Parse[ptrValidated](`{"n":3}`)
	var ve *ValidationError
	if !errors.As(err, &ve) || !strings.Contains(err.Error(), "n must be even") {
		t.Fatalf("err=%v", err)
	}
	if _, err := Parse[ptrValidated](`{"n":4}`); err != nil {
		t.Fatal(err)
	}
	ex := &ExhaustedError{Attempts: 3, Last: ve}
	if !strings.Contains(ex.Error(), "3 attempts") || !errors.Is(ex, ex.Last) {
		t.Fatalf("ExhaustedError=%v", ex)
	}
}

func TestJSONTypeNames(t *testing.T) {
	for v, want := range map[any]string{nil: "null", "s": "string", 1.0: "number", true: "boolean"} {
		if got := jsonType(v); got != want {
			t.Errorf("jsonType(%v)=%s want %s", v, got, want)
		}
	}
	if jsonType(map[string]any{}) != "object" || jsonType([]any{}) != "array" || jsonType(int8(1)) != "int8" {
		t.Error("jsonType composite")
	}
}
