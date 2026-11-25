// Command demo shows the validate-and-repair loop without an API key: a
// scripted "model" first returns an invalid ticket, then fixes it after
// receiving the exact validation errors.
//
//	go run ./examples/demo
//
// To use a real model instead, replace `model` with
// openai.New(os.Getenv("OPENAI_API_KEY"), "gpt-4o-mini").
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/shani8615/go-structllm"
)

type Ticket struct {
	Severity string   `json:"severity" enum:"low,medium,high,critical"`
	Summary  string   `json:"summary" maxLen:"60"`
	Tags     []string `json:"tags" maxItems:"3" pattern:"^[a-z-]+$"`
}

func main() {
	replies := []string{
		`{"severity":"URGENT","summary":"Postgres primary is down","tags":["DB","Outage"],"eta":"1h"}`,
		`{"severity":"critical","summary":"Postgres primary is down","tags":["db","outage"]}`,
	}
	model := structllm.ClientFunc(func(_ context.Context, req structllm.Request) (string, error) {
		if n := len(req.Messages); req.Messages[n-1].Role == structllm.RoleUser && n > 2 {
			fmt.Println("\033[33m↺ sent the problems back to the model\033[0m")
		}
		r := replies[0]
		replies = replies[1:]
		return r, nil
	})

	fmt.Println("\033[1m$ structllm.Extract[Ticket](ctx, model, alert)\033[0m")
	t, err := structllm.Extract[Ticket](context.Background(), model, "Alert: pg-primary-1 unreachable",
		structllm.OnAttempt(func(a structllm.Attempt) {
			fmt.Printf("\033[2mattempt %d → %s\033[0m\n", a.Number, a.Raw)
			for _, is := range a.Issues {
				fmt.Printf("  \033[31m✗ %s\033[0m\n", is)
			}
		}))
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("\033[32m✓ Ticket{Severity:%q, Summary:%q, Tags:[%s]}\033[0m\n",
		t.Severity, t.Summary, strings.Join(t.Tags, " "))
}
