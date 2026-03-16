// This example demonstrates using the OpenAI Responses API over WebSocket
// for a persistent, multi-turn conversation with tool calling.
//
// The WebSocket transport enables ~40% faster execution for tool-heavy
// workflows by keeping a connection open and sending only incremental
// inputs per turn (via automatic previous_response_id chaining).
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

func main() {
	llm, err := openai.New(openai.WithModel("gpt-5.4"))
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	// Open a persistent WebSocket session to the Responses API.
	session, err := llm.OpenResponsesSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	// Turn 1: Ask a question that requires a tool call.
	fmt.Println("Asking about the weather...")
	resp, err := session.Send(ctx,
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "What is the weather like in Boston?"),
		},
		llms.WithTools(weatherTools),
	)
	if err != nil {
		log.Fatal(err)
	}

	// Check if the model wants to call a tool.
	if len(resp.Choices[0].ToolCalls) > 0 {
		tc := resp.Choices[0].ToolCalls[0]
		fmt.Printf("Model requested tool: %s(%s)\n", tc.FunctionCall.Name, tc.FunctionCall.Arguments)

		// Execute the tool locally.
		toolOutput := executeWeatherTool(tc.FunctionCall.Arguments)
		fmt.Printf("Tool result: %s\n", toolOutput)

		// Turn 2: Send the tool result back.
		// Only the tool output is sent — the WebSocket session automatically
		// chains to the previous response via previous_response_id.
		resp, err = session.SendToolResults(ctx,
			[]openai.ToolResult{{CallID: tc.ID, Output: toolOutput}},
		)
		if err != nil {
			log.Fatal(err)
		}
	}

	fmt.Printf("\nAssistant: %s\n", resp.Choices[0].Content)

	// Turn 3: Follow-up question on the same session.
	// The model remembers the conversation context via server-side caching.
	fmt.Println("\nAsking a follow-up...")
	resp, err = session.Send(ctx,
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "Should I bring an umbrella?"),
		},
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Assistant: %s\n", resp.Choices[0].Content)
	fmt.Printf("\nSession used %s responses (last ID: %s)\n", "3", session.LastResponseID())
}

func executeWeatherTool(argsJSON string) string {
	var args struct {
		Location string `json:"location"`
	}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return fmt.Sprintf("error: %v", err)
	}

	weather := map[string]string{
		"boston":  `{"temp": "62°F", "condition": "partly cloudy", "humidity": "45%"}`,
		"chicago": `{"temp": "55°F", "condition": "windy and overcast", "humidity": "60%"}`,
	}

	result, ok := weather[strings.ToLower(args.Location)]
	if !ok {
		return fmt.Sprintf(`{"error": "no weather data for %s"}`, args.Location)
	}
	return result
}

var weatherTools = []llms.Tool{
	{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:        "getCurrentWeather",
			Description: "Get the current weather in a given location",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{
						"type":        "string",
						"description": "The city name, e.g. Boston",
					},
				},
				"required": []string{"location"},
			},
		},
	},
}
