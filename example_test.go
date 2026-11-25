package structllm_test

import (
	"context"
	"fmt"

	"github.com/shani8615/go-structllm"
)

type Invoice struct {
	Vendor   string  `json:"vendor" minLen:"1"`
	Currency string  `json:"currency" enum:"USD,EUR,PKR" desc:"ISO 4217 code"`
	Total    float64 `json:"total" min:"0"`
}

func ExampleExtract() {
	// A fake model: the first answer breaks the schema, the second is fixed.
	replies := []string{
		`{"vendor":"Acme","currency":"dollars","total":120.5}`,
		`{"vendor":"Acme","currency":"USD","total":120.5}`,
	}
	model := structllm.ClientFunc(func(_ context.Context, req structllm.Request) (string, error) {
		r := replies[0]
		replies = replies[1:]
		return r, nil
	})

	inv, err := structllm.Extract[Invoice](context.Background(), model,
		"Extract the invoice: 'Acme Corp, total due $120.50'",
		structllm.OnAttempt(func(a structllm.Attempt) {
			for _, is := range a.Issues {
				fmt.Println("attempt", a.Number, "issue:", is)
			}
		}))
	if err != nil {
		panic(err)
	}
	fmt.Printf("%s %.2f %s\n", inv.Vendor, inv.Total, inv.Currency)
	// Output:
	// attempt 1 issue: $.currency: must be one of [USD, EUR, PKR]
	// Acme 120.50 USD
}
