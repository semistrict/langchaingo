package openai

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/internal/httprr"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai/chatgptauth"
)

// newChatGPTTestClient creates an LLM configured with ChatGPT auth and httprr.
// When recording, it uses real ChatGPT OAuth credentials.
// When replaying, it uses a fake token with the recorded responses.
func newChatGPTTestClient(t *testing.T, opts ...Option) *LLM {
	t.Helper()

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	// Remove ChatGPT-Account-ID so recording matches replay (replay has no token provider).
	rr.ScrubReq(func(req *http.Request) error {
		req.Header.Del("ChatGPT-Account-ID")
		return nil
	})

	if !rr.Recording() {
		t.Parallel()
	}

	clientOpts := []Option{
		WithHTTPClient(rr.Client()),
		WithBaseURL(chatgptauth.DefaultBaseURL),
	}

	if rr.Recording() {
		// Use real ChatGPT auth when recording.
		clientOpts = append(clientOpts, WithChatGPTAuth(""))
	} else {
		// Use fake token matching what the scrubber produces.
		clientOpts = append(clientOpts, WithToken("test-api-key"))
	}

	clientOpts = append(clientOpts, opts...)
	llm, err := New(clientOpts...)
	require.NoError(t, err)
	return llm
}

func TestResponsesHTTP_BasicCompletion(t *testing.T) {
	llm := newChatGPTTestClient(t, WithModel("gpt-5.4"))
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
	assert.NotEmpty(t, resp.Choices[0].ID, "response should have an ID")
}

func TestResponsesHTTP_MultiTurn(t *testing.T) {
	llm := newChatGPTTestClient(t, WithModel("gpt-5.4"))
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
	assert.NotEmpty(t, resp1.Choices[0].ID)

	// Turn 2: replay full history (ChatGPT backend doesn't support previous_response_id).
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

func TestResponsesHTTP_Call(t *testing.T) {
	llm := newChatGPTTestClient(t, WithModel("gpt-5.4"))
	ctx := t.Context()

	session, err := llm.OpenResponsesHTTPSession(ctx)
	require.NoError(t, err)
	defer session.Close()

	// Call requires instructions, so use GenerateContent with a system message.
	resp, err := session.GenerateContent(ctx,
		[]llms.MessageContent{
			{Role: llms.ChatMessageTypeSystem, Parts: []llms.ContentPart{llms.TextContent{Text: "Answer in exactly one word."}}},
			{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextContent{Text: "What is 2+2?"}}},
		},
	)
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)
	assert.Contains(t, resp.Choices[0].Content, "4")
}
