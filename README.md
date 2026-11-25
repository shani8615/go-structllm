# go-structllm

[![ci](https://github.com/shani8615/go-structllm/actions/workflows/ci.yml/badge.svg)](https://github.com/shani8615/go-structllm/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/shani8615/go-structllm.svg)](https://pkg.go.dev/github.com/shani8615/go-structllm)

**Typed, validated LLM output for Go.** Describe what you want as a Go struct and get back a `T` or a precise error, never a half-parsed value. It works like Instructor/Pydantic, but for Go, and it has no dependencies outside the standard library.

```bash
go get github.com/shani8615/go-structllm
```

```go
type Ticket struct {
    Severity string   `json:"severity" enum:"low,medium,high,critical" desc:"How urgent it is"`
    Summary  string   `json:"summary"  minLen:"1" maxLen:"200"`
    Tags     []string `json:"tags"     maxItems:"5" pattern:"^[a-z-]+$"`
    Owner    *string  `json:"owner"`   // pointer: optional and nullable
}

client := openai.New(os.Getenv("OPENAI_API_KEY"), "gpt-4o-mini")

t, err := structllm.Extract[Ticket](ctx, client, "Triage this alert: "+alert)
```

## How it works

1. **Schema from your struct.** `SchemaFor[T]()` builds a JSON Schema from field types and tags. Properties keep their declaration order, so prompts are deterministic.
2. **Ask.** The schema goes into the system prompt. It is also passed as `response_format: json_schema` to providers that support constrained decoding.
3. **Validate.** The reply is checked against the schema: types, required fields, enums, lengths, item counts, numeric bounds, patterns, RFC 3339 dates, and unknown fields. If `T` has a `Validate() error` method, that runs too.
4. **Repair.** Invalid replies go back to the model with the exact problems, for example `$.tags[2]: must match pattern ^[a-z-]+$`. This happens at most `MaxRepairs` times (default 2).
5. **Return** a `T`, or an `*ExhaustedError` that wraps the last `*ValidationError`.

The model's reply is never trusted. Parsing and validation in code decide what reaches your program, not instructions like "respond with EXACTLY…".

## Struct tags

| Tag | Example | Applies to |
|---|---|---|
| `desc` | `desc:"ISO 4217 code"` | any field |
| `enum` | `enum:"low,high"` | strings, numbers (or their slice items) |
| `minLen` / `maxLen` | `maxLen:"200"` | strings (counted in characters) |
| `minItems` / `maxItems` | `maxItems:"5"` | slices |
| `min` / `max` | `min:"0" max:"1"` | numbers |
| `pattern` | `pattern:"^[A-Z]{3}$"` | strings |
| `format` | `format:"email"` | strings; `time.Time` is `date-time` automatically |

Fields are required unless they are pointers or `omitempty`. Embedded structs are flattened, as in `encoding/json`.

## Business rules

```go
type Window struct {
    Start time.Time `json:"start"`
    End   time.Time `json:"end"`
}

func (w Window) Validate() error {
    if !w.End.After(w.Start) {
        return errors.New("end must be after start") // sent back to the model
    }
    return nil
}
```

## Options

```go
structllm.Extract[T](ctx, client, prompt,
    structllm.System("You are a precise invoice parser."),
    structllm.History(prevTurns...),
    structllm.MaxRepairs(3),
    structllm.OnAttempt(func(a structllm.Attempt) { metrics.Observe(a.Number, len(a.Issues)) }),
)
```

Already have a reply, for example from a streaming call? Use `structllm.Parse[T](reply)`.

## Providers

`openai.Client` speaks the OpenAI chat-completions protocol, so it also works with **Azure OpenAI, Ollama, vLLM, LM Studio, Groq and Together**:

```go
c := &openai.Client{BaseURL: "http://localhost:11434/v1", Model: "llama3.1"} // Ollama
c.ResponseFormat = openai.FormatJSONObject // for servers without json_schema support
```

For anything else, wrap your SDK in a `structllm.ClientFunc`:

```go
client := structllm.ClientFunc(func(ctx context.Context, req structllm.Request) (string, error) {
    return myProvider.Chat(ctx, req.Messages)
})
```

`openai.APIError` exposes `StatusCode` and `Temporary()`, so you can combine it with [go-resilience](https://github.com/shani8615/go-resilience) for retries and circuit breaking. API keys never appear in errors.

## Testing your code

`Client` is a single-method interface, so tests can use a scripted fake. See `example_test.go`.

## License

MIT
