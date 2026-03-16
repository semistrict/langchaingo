package openai

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/internal/wsrr"
	"github.com/tmc/langchaingo/llms"
)

func newResponsesTestClient(t *testing.T) (*LLM, *wsrr.RecordReplay) {
	t.Helper()

	wsrr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")

	llm, err := New()
	require.NoError(t, err)

	rr := wsrr.OpenForTest(t, "wss://api.openai.com/v1/responses")

	if !rr.Recording() {
		t.Parallel()
	}

	return llm, rr
}

func TestResponsesWebSearch(t *testing.T) {
	llm, rr := newResponsesTestClient(t)

	ctx := t.Context()

	var session *ResponsesSession
	var err error

	if rr.Recording() {
		// Recording: connect to real API.
		session, err = llm.OpenResponsesSession(ctx)
		require.NoError(t, err)
		session.SetRecorder(rr)
	} else {
		// Replay: connect to local server.
		replayLLM, err := New(
			WithToken("fake-token"),
			WithBaseURL(rr.DialURL("")),
		)
		require.NoError(t, err)
		session, err = replayLLM.OpenResponsesSession(ctx)
		require.NoError(t, err)
	}
	defer session.Close()

	resp, err := session.Send(ctx,
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "What is the current population of Tokyo? Be concise."),
		},
		llms.WithModel("gpt-5.4"),
		llms.WithTools([]llms.Tool{
			{Type: "web_search"},
		}),
	)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.Choices)
	c := resp.Choices[0]
	assert.Regexp(t, `(?i)tokyo`, c.Content)
	assert.Regexp(t, `\d`, c.Content) // should contain numbers
}

func TestResponsesMultiTurn(t *testing.T) {
	llm, rr := newResponsesTestClient(t)

	ctx := t.Context()

	var session *ResponsesSession
	var err error

	if rr.Recording() {
		session, err = llm.OpenResponsesSession(ctx)
		require.NoError(t, err)
		session.SetRecorder(rr)
	} else {
		replayLLM, err := New(
			WithToken("fake-token"),
			WithBaseURL(rr.DialURL("")),
		)
		require.NoError(t, err)
		session, err = replayLLM.OpenResponsesSession(ctx)
		require.NoError(t, err)
	}
	defer session.Close()

	// Turn 1
	resp1, err := session.Send(ctx,
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "What is 2+2? Answer with just the number."),
		},
		llms.WithModel("gpt-5.4"),
	)
	require.NoError(t, err)
	assert.Contains(t, resp1.Choices[0].Content, "4")

	// Turn 2 — should automatically chain via previous_response_id
	resp2, err := session.Send(ctx,
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "Multiply that by 3. Answer with just the number."),
		},
		llms.WithModel("gpt-5.4"),
	)
	require.NoError(t, err)
	assert.Contains(t, resp2.Choices[0].Content, "12")

	// Verify chaining happened.
	assert.NotEmpty(t, session.LastResponseID())
}

func TestResponsesStreamingLive(t *testing.T) {
	llm, rr := newResponsesTestClient(t)

	ctx := t.Context()

	var session *ResponsesSession
	var err error

	if rr.Recording() {
		session, err = llm.OpenResponsesSession(ctx)
		require.NoError(t, err)
		session.SetRecorder(rr)
	} else {
		replayLLM, err := New(
			WithToken("fake-token"),
			WithBaseURL(rr.DialURL("")),
		)
		require.NoError(t, err)
		session, err = replayLLM.OpenResponsesSession(ctx)
		require.NoError(t, err)
	}
	defer session.Close()

	var chunks []string
	resp, err := session.Send(ctx,
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "Say exactly: Hello World"),
		},
		llms.WithModel("gpt-5.4"),
		llms.WithStreamingFunc(func(_ context.Context, chunk []byte) error {
			chunks = append(chunks, string(chunk))
			return nil
		}),
	)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.Choices)
	fullContent := resp.Choices[0].Content
	assert.Regexp(t, `(?i)hello.*world`, fullContent)

	// Streaming should have produced multiple chunks.
	if rr.Recording() {
		// Only check chunk count when recording against real API.
		assert.Greater(t, len(chunks), 0)
	}

	// Verify chunks concatenate to full content.
	assert.Equal(t, fullContent, strings.Join(chunks, ""))
}
