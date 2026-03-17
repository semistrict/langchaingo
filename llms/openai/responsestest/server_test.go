package responsestest_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/llms/openai/responsestest"
)

func TestServer_Echo(t *testing.T) {
	srv := responsestest.NewServer(0)
	defer srv.Close()

	llm, err := openai.New(
		openai.WithToken("test-token"),
		openai.WithModel("gpt-4o"),
		openai.WithBaseURL(srv.URL),
	)
	require.NoError(t, err)

	session, err := llm.OpenResponsesHTTPSession(t.Context())
	require.NoError(t, err)
	defer session.Close()

	resp, err := session.GenerateContent(t.Context(),
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "Hello, world!"),
		},
	)
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)
	assert.Equal(t, "Hello, world!", resp.Choices[0].Content)
	assert.Equal(t, int64(1), srv.RequestCount())
}

func TestServer_Streaming(t *testing.T) {
	srv := responsestest.NewServer(time.Millisecond)
	defer srv.Close()

	llm, err := openai.New(
		openai.WithToken("test-token"),
		openai.WithModel("gpt-4o"),
		openai.WithBaseURL(srv.URL),
	)
	require.NoError(t, err)

	session, err := llm.OpenResponsesHTTPSession(t.Context())
	require.NoError(t, err)
	defer session.Close()

	// Use a string longer than chunkSize (16) to get multiple chunks.
	input := "The quick brown fox jumps over the lazy dog near the river"

	var chunks []string
	resp, err := session.GenerateContent(t.Context(),
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, input),
		},
		llms.WithStreamingFunc(func(_ context.Context, chunk []byte) error {
			chunks = append(chunks, string(chunk))
			return nil
		}),
	)
	require.NoError(t, err)
	assert.Equal(t, input, resp.Choices[0].Content)
	assert.Greater(t, len(chunks), 1, "should receive multiple chunks")
	// Reassembled chunks must equal the original input.
	assert.Equal(t, input, strings.Join(chunks, ""))
}

func TestServer_CallTool(t *testing.T) {
	srv := responsestest.NewServer(0)
	defer srv.Close()

	llm, err := openai.New(
		openai.WithToken("test-token"),
		openai.WithModel("gpt-4o"),
		openai.WithBaseURL(srv.URL),
	)
	require.NoError(t, err)

	session, err := llm.OpenResponsesHTTPSession(t.Context())
	require.NoError(t, err)
	defer session.Close()

	resp, err := session.GenerateContent(t.Context(),
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, `calltool get_weather {"city": "SF"}`),
		},
	)
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)
	assert.Equal(t, "tool_calls", resp.Choices[0].StopReason)
	require.Len(t, resp.Choices[0].ToolCalls, 1)

	tc := resp.Choices[0].ToolCalls[0]
	assert.Equal(t, "get_weather", tc.FunctionCall.Name)
	assert.Equal(t, `{"city": "SF"}`, tc.FunctionCall.Arguments)
	assert.Contains(t, tc.ID, "call_get_weather_")
}

func TestServer_CallToolNoArgs(t *testing.T) {
	srv := responsestest.NewServer(0)
	defer srv.Close()

	llm, err := openai.New(
		openai.WithToken("test-token"),
		openai.WithModel("gpt-4o"),
		openai.WithBaseURL(srv.URL),
	)
	require.NoError(t, err)

	session, err := llm.OpenResponsesHTTPSession(t.Context())
	require.NoError(t, err)
	defer session.Close()

	resp, err := session.GenerateContent(t.Context(),
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "calltool list_items"),
		},
	)
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)
	assert.Equal(t, "tool_calls", resp.Choices[0].StopReason)
	require.Len(t, resp.Choices[0].ToolCalls, 1)

	tc := resp.Choices[0].ToolCalls[0]
	assert.Equal(t, "list_items", tc.FunctionCall.Name)
	assert.Equal(t, "{}", tc.FunctionCall.Arguments)
}

func TestServer_MultiTurn(t *testing.T) {
	srv := responsestest.NewServer(0)
	defer srv.Close()

	llm, err := openai.New(
		openai.WithToken("test-token"),
		openai.WithModel("gpt-4o"),
		openai.WithBaseURL(srv.URL),
	)
	require.NoError(t, err)

	session, err := llm.OpenResponsesHTTPSession(t.Context())
	require.NoError(t, err)
	defer session.Close()

	resp1, err := session.GenerateContent(t.Context(),
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "first"),
		},
	)
	require.NoError(t, err)
	assert.Equal(t, "first", resp1.Choices[0].Content)

	resp2, err := session.GenerateContent(t.Context(),
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "second"),
		},
	)
	require.NoError(t, err)
	assert.Equal(t, "second", resp2.Choices[0].Content)
	assert.Equal(t, int64(2), srv.RequestCount())
}
