// Command responses-mock runs a mock OpenAI Responses API server.
//
// It echoes back the user's message as streaming SSE. If the message
// starts with "calltool NAME {args}", it returns a tool call instead.
//
// Usage:
//
//	go run ./cmd/responses-mock -addr :8089 -delay 50ms
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/tmc/langchaingo/llms/openai/responsestest"
)

func main() {
	addr := flag.String("addr", ":8089", "listen address")
	delay := flag.Duration("delay", 50*time.Millisecond, "delay between SSE chunks")
	flag.Parse()

	h := responsestest.Handler(*delay)
	fmt.Printf("responses-mock listening on %s (chunk delay %s)\n", *addr, *delay)
	log.Fatal(http.ListenAndServe(*addr, h))
}
