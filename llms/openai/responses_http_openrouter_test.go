package openai

import (
	"encoding/json"
	"net/http"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/internal/httprr"
	"github.com/tmc/langchaingo/llms"
)

func newOpenRouterResponsesTestClient(t *testing.T, opts ...Option) *LLM {
	t.Helper()

	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENROUTER_API_KEY")

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	if !rr.Recording() {
		t.Parallel()
	}

	apiKey := "test-api-key"
	if key := os.Getenv("OPENROUTER_API_KEY"); key != "" && rr.Recording() {
		apiKey = key
	}

	clientOpts := []Option{
		WithToken(apiKey),
		WithBaseURL("https://openrouter.ai/api/v1"),
		WithHTTPClient(rr.Client()),
	}
	clientOpts = append(clientOpts, opts...)

	llm, err := New(clientOpts...)
	require.NoError(t, err)
	return llm
}

func TestResponsesHTTPOpenRouter_BasicCompletion(t *testing.T) {
	llm := newOpenRouterResponsesTestClient(t, WithModel("minimax/minimax-m2.5"))
	ctx := t.Context()

	session, err := llm.OpenResponsesHTTPSession(ctx)
	require.NoError(t, err)
	defer session.Close()

	resp, err := session.GenerateContent(ctx,
		[]llms.MessageContent{
			{Role: llms.ChatMessageTypeSystem, Parts: []llms.ContentPart{llms.TextContent{Text: "You are a helpful assistant. Be concise."}}},
			{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextContent{Text: "What is the capital of France? One sentence."}}},
		},
	)
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)
	assert.Contains(t, resp.Choices[0].Content, "Paris")
}

func TestResponsesHTTPOpenRouter_MultiTurn(t *testing.T) {
	llm := newOpenRouterResponsesTestClient(t, WithModel("minimax/minimax-m2.5"))
	ctx := t.Context()

	session, err := llm.OpenResponsesHTTPSession(ctx)
	require.NoError(t, err)
	defer session.Close()

	// Turn 1.
	resp1, err := session.GenerateContent(ctx,
		[]llms.MessageContent{
			{Role: llms.ChatMessageTypeSystem, Parts: []llms.ContentPart{llms.TextContent{Text: "You are a helpful assistant. Be concise."}}},
			{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextContent{Text: "Remember the number 42."}}},
		},
	)
	require.NoError(t, err)
	require.Len(t, resp1.Choices, 1)

	// Turn 2: replay full history (OpenRouter is stateless).
	resp2, err := session.GenerateContent(ctx,
		[]llms.MessageContent{
			{Role: llms.ChatMessageTypeSystem, Parts: []llms.ContentPart{llms.TextContent{Text: "You are a helpful assistant. Be concise."}}},
			{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextContent{Text: "Remember the number 42."}}},
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.TextContent{Text: resp1.Choices[0].Content}}},
			{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextContent{Text: "What number did I ask you to remember?"}}},
		},
	)
	require.NoError(t, err)
	require.Len(t, resp2.Choices, 1)
	assert.Contains(t, resp2.Choices[0].Content, "42")
}

// TestResponsesHTTPOpenRouter_FunctionCall tests a single function call round-trip:
// the model calls a tool, we send the result back, and the model responds with text.
func TestResponsesHTTPOpenRouter_FunctionCall(t *testing.T) {
	llm := newOpenRouterResponsesTestClient(t, WithModel("openai/gpt-4.1-mini"))
	ctx := t.Context()

	session, err := llm.OpenResponsesHTTPSession(ctx)
	require.NoError(t, err)
	defer session.Close()

	weatherTool := llms.Tool{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:        "get_weather",
			Description: "Get the current weather for a city.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"city": {"type": "string", "description": "The city name."}
				},
				"required": ["city"],
				"additionalProperties": false
			}`),
		},
	}

	// Turn 1: ask about weather — model should call get_weather.
	resp1, err := session.GenerateContent(ctx,
		[]llms.MessageContent{
			{Role: llms.ChatMessageTypeSystem, Parts: []llms.ContentPart{llms.TextContent{Text: "You are a helpful assistant. Use the get_weather tool to answer weather questions."}}},
			{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextContent{Text: "What's the weather in San Francisco?"}}},
		},
		llms.WithTools([]llms.Tool{weatherTool}),
	)
	require.NoError(t, err)
	require.Len(t, resp1.Choices, 1)
	require.NotEmpty(t, resp1.Choices[0].ToolCalls, "expected model to call get_weather")
	tc := resp1.Choices[0].ToolCalls[0]
	assert.Equal(t, "get_weather", tc.FunctionCall.Name)
	assert.Contains(t, tc.FunctionCall.Arguments, "San Francisco")
	t.Logf("Tool call: %s(%s)", tc.FunctionCall.Name, tc.FunctionCall.Arguments)

	// Turn 2: send tool result back with full history.
	resp2, err := session.GenerateContent(ctx,
		[]llms.MessageContent{
			{Role: llms.ChatMessageTypeSystem, Parts: []llms.ContentPart{llms.TextContent{Text: "You are a helpful assistant. Use the get_weather tool to answer weather questions."}}},
			{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextContent{Text: "What's the weather in San Francisco?"}}},
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{tc}},
			{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{llms.ToolCallResponse{
				ToolCallID: tc.ID,
				Name:       "get_weather",
				Content:    `{"temperature": 65, "condition": "sunny", "humidity": 55}`,
			}}},
		},
		llms.WithTools([]llms.Tool{weatherTool}),
	)
	require.NoError(t, err)
	require.Len(t, resp2.Choices, 1)
	assert.Empty(t, resp2.Choices[0].ToolCalls, "expected text response, not more tool calls")
	assert.Contains(t, resp2.Choices[0].Content, "sunny")
	t.Logf("Final response: %s", resp2.Choices[0].Content)

	// Verify token usage in GenerationInfo.
	info := resp2.Choices[0].GenerationInfo
	inputTok, _ := info["input_tokens"].(int)
	outputTok, _ := info["output_tokens"].(int)
	cachedTok, _ := info["cached_tokens"].(int)
	t.Logf("Usage: input=%d output=%d cached=%d", inputTok, outputTok, cachedTok)
	assert.Greater(t, inputTok, 0)
	assert.Greater(t, outputTok, 0)
}

// TestResponsesHTTPOpenRouter_FunctionCallMultiTurn tests multiple rounds of
// tool calls: the model calls a tool, gets a result, then calls another tool.
func TestResponsesHTTPOpenRouter_FunctionCallMultiTurn(t *testing.T) {
	llm := newOpenRouterResponsesTestClient(t, WithModel("openai/gpt-4.1-mini"))
	ctx := t.Context()

	session, err := llm.OpenResponsesHTTPSession(ctx)
	require.NoError(t, err)
	defer session.Close()

	tools := []llms.Tool{
		{
			Type: "function",
			Function: &llms.FunctionDefinition{
				Name:        "get_weather",
				Description: "Get the current weather for a city.",
				Parameters: json.RawMessage(`{
					"type": "object",
					"properties": {
						"city": {"type": "string", "description": "The city name."}
					},
					"required": ["city"],
					"additionalProperties": false
				}`),
			},
		},
		{
			Type: "function",
			Function: &llms.FunctionDefinition{
				Name:        "get_population",
				Description: "Get the population of a city.",
				Parameters: json.RawMessage(`{
					"type": "object",
					"properties": {
						"city": {"type": "string", "description": "The city name."}
					},
					"required": ["city"],
					"additionalProperties": false
				}`),
			},
		},
	}

	messages := []llms.MessageContent{
		{Role: llms.ChatMessageTypeSystem, Parts: []llms.ContentPart{llms.TextContent{Text: "You are a helpful assistant. Always use the provided tools to answer questions. Call each tool separately."}}},
		{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextContent{Text: "What's the weather and population of Tokyo?"}}},
	}

	// Agent loop: keep going until we get a text response with no tool calls.
	maxIterations := 5
	for i := range maxIterations {
		resp, err := session.GenerateContent(ctx, messages, llms.WithTools(tools))
		require.NoError(t, err)
		require.Len(t, resp.Choices, 1)
		choice := resp.Choices[0]

		if len(choice.ToolCalls) == 0 {
			// Final text response.
			t.Logf("Final response (after %d tool rounds): %s", i, choice.Content)
			assert.Contains(t, choice.Content, "Tokyo")
			return
		}

		// Add assistant tool calls to history.
		var aiParts []llms.ContentPart
		if choice.Content != "" {
			aiParts = append(aiParts, llms.TextContent{Text: choice.Content})
		}
		for _, tc := range choice.ToolCalls {
			aiParts = append(aiParts, tc)
		}
		messages = append(messages, llms.MessageContent{Role: llms.ChatMessageTypeAI, Parts: aiParts})

		// Execute and add tool results.
		var toolParts []llms.ContentPart
		for _, tc := range choice.ToolCalls {
			t.Logf("Tool call %d: %s(%s)", i, tc.FunctionCall.Name, tc.FunctionCall.Arguments)
			output := fakeToolResult(tc.FunctionCall.Name)
			toolParts = append(toolParts, llms.ToolCallResponse{
				ToolCallID: tc.ID,
				Name:       tc.FunctionCall.Name,
				Content:    output,
			})
		}
		messages = append(messages, llms.MessageContent{Role: llms.ChatMessageTypeTool, Parts: toolParts})
	}

	t.Fatal("agent did not produce a final text response within max iterations")
}

func fakeToolResult(name string) string {
	switch name {
	case "get_weather":
		return `{"temperature": 22, "condition": "partly cloudy", "humidity": 60}`
	case "get_population":
		return `{"population": 13960000, "year": 2024}`
	default:
		return `{"error": "unknown tool"}`
	}
}

func TestResponsesHTTPOpenRouter_WebSearch(t *testing.T) {
	// Use the :online suffix to enable web search via OpenRouter plugin.
	llm := newOpenRouterResponsesTestClient(t, WithModel("minimax/minimax-m2.5:online"))
	ctx := t.Context()

	session, err := llm.OpenResponsesHTTPSession(ctx)
	require.NoError(t, err)
	defer session.Close()

	resp, err := session.GenerateContent(ctx,
		[]llms.MessageContent{
			{Role: llms.ChatMessageTypeSystem, Parts: []llms.ContentPart{llms.TextContent{Text: "You are a helpful assistant. Be concise. Use web search to answer."}}},
			{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextContent{Text: "What is the current population of Tokyo?"}}},
		},
	)
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)
	// Should contain some population figure.
	assert.NotEmpty(t, resp.Choices[0].Content)
	t.Logf("Web search response: %s", resp.Choices[0].Content)
}
