// This example demonstrates using ChatGPT OAuth authentication instead of
// an API key. It uses the auth.json file produced by "npx @openai/codex login"
// to authenticate via your ChatGPT subscription.
//
// ChatGPT auth uses the Responses API (WebSocket), so we open a session
// and use it as an llms.Model via GenerateContent.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

func main() {
	// WithChatGPTAuth loads OAuth tokens from ~/.codex/auth.json (or the
	// specified path) and sets the base URL to the ChatGPT backend API.
	// No OPENAI_API_KEY needed.
	llm, err := openai.New(
		openai.WithChatGPTAuth(""),
		openai.WithModel("gpt-5.4"),
	)
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	// Open an HTTP-based Responses API session. The ChatGPT backend uses
	// HTTP POST + SSE (not WebSocket). The session implements llms.Model,
	// so it works with GenerateFromSinglePrompt and agents.
	session, err := llm.OpenResponsesHTTPSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	resp, err := session.GenerateContent(ctx,
		[]llms.MessageContent{
			{Role: llms.ChatMessageTypeSystem, Parts: []llms.ContentPart{llms.TextContent{Text: "You are a helpful assistant."}}},
			{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextContent{Text: "What is the capital of France? Answer in one sentence."}}},
		},
	)
	if err != nil {
		log.Fatal(err)
	}

	if len(resp.Choices) > 0 {
		fmt.Println(resp.Choices[0].Content)
	}
}
