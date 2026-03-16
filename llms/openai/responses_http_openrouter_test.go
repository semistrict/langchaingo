package openai

import (
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
