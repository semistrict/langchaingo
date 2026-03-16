package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/llms"
	"nhooyr.io/websocket"
	"nhooyr.io/websocket/wsjson"
)

func mustJSON(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}

// newResponsesTestServer creates a WebSocket server that handles a sequence of turns.
// Each turn is a slice of events to send after reading a response.create request.
func newResponsesTestServer(t *testing.T, turns [][]json.RawMessage) *httptest.Server {
	t.Helper()
	turnIdx := 0
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Errorf("websocket accept: %v", err)
			return
		}
		defer conn.Close(websocket.StatusNormalClosure, "done")

		ctx := r.Context()
		for {
			var raw json.RawMessage
			if err := wsjson.Read(ctx, conn, &raw); err != nil {
				return // connection closed
			}

			if turnIdx >= len(turns) {
				t.Errorf("unexpected turn %d", turnIdx)
				return
			}

			for _, event := range turns[turnIdx] {
				if err := wsjson.Write(ctx, conn, event); err != nil {
					return
				}
			}
			turnIdx++
		}
	}))
}

// newResponsesTestServerWithCapture is like newResponsesTestServer but captures
// the last request received into *captured.
func newResponsesTestServerWithCapture(t *testing.T, turns [][]json.RawMessage, captured *json.RawMessage) *httptest.Server {
	t.Helper()
	turnIdx := 0
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Errorf("websocket accept: %v", err)
			return
		}
		defer conn.Close(websocket.StatusNormalClosure, "done")

		ctx := r.Context()
		for {
			var raw json.RawMessage
			if err := wsjson.Read(ctx, conn, &raw); err != nil {
				return
			}
			*captured = append(json.RawMessage(nil), raw...)

			if turnIdx >= len(turns) {
				t.Errorf("unexpected turn %d", turnIdx)
				return
			}

			for _, event := range turns[turnIdx] {
				if err := wsjson.Write(ctx, conn, event); err != nil {
					return
				}
			}
			turnIdx++
		}
	}))
}

func newTestLLMWithWSServer(t *testing.T, srv *httptest.Server) *LLM {
	t.Helper()
	// The internal client converts baseURL + "/responses" to a ws:// URL.
	// Our test server handles any path, so just pass srv.URL as the base.
	llm, err := New(
		WithToken("test-token"),
		WithModel("gpt-5.4"),
		WithBaseURL(srv.URL),
	)
	require.NoError(t, err)
	return llm
}

func TestResponsesSession_Send(t *testing.T) {
	turns := [][]json.RawMessage{
		{
			mustJSON(map[string]any{
				"type":     "response.created",
				"response": map[string]any{"id": "resp_1"},
			}),
			mustJSON(map[string]any{
				"type":  "response.output_text.delta",
				"delta": "Hello from Responses API!",
			}),
			mustJSON(map[string]any{
				"type": "response.completed",
				"response": map[string]any{
					"id":    "resp_1",
					"usage": map[string]any{"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
				},
			}),
		},
	}

	srv := newResponsesTestServer(t, turns)
	defer srv.Close()

	llm := newTestLLMWithWSServer(t, srv)
	ctx := t.Context()

	session, err := llm.OpenResponsesSession(ctx)
	require.NoError(t, err)
	defer session.Close()

	resp, err := session.Send(ctx,
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "Hello"),
		},
	)
	require.NoError(t, err)

	assert.Len(t, resp.Choices, 1)
	assert.Equal(t, "Hello from Responses API!", resp.Choices[0].Content)
	assert.Equal(t, "stop", resp.Choices[0].StopReason)
	assert.Equal(t, "resp_1", session.LastResponseID())
}

func TestResponsesSession_SendToolResults(t *testing.T) {
	turns := [][]json.RawMessage{
		// Turn 1: model requests a tool call.
		{
			mustJSON(map[string]any{
				"type":     "response.created",
				"response": map[string]any{"id": "resp_tool"},
			}),
			mustJSON(map[string]any{
				"type": "response.output_item.added",
			}),
			mustJSON(map[string]any{
				"type":  "response.function_call_arguments.delta",
				"delta": `{"city": "SF"}`,
			}),
			mustJSON(map[string]any{
				"type": "response.output_item.done",
				"item": map[string]any{
					"type":      "function_call",
					"id":        "fc_1",
					"call_id":   "call_abc",
					"name":      "get_weather",
					"arguments": `{"city": "SF"}`,
				},
			}),
			mustJSON(map[string]any{
				"type": "response.completed",
				"response": map[string]any{
					"id":    "resp_tool",
					"usage": map[string]any{"input_tokens": 15, "output_tokens": 8, "total_tokens": 23},
				},
			}),
		},
		// Turn 2: model responds with text after receiving tool result.
		{
			mustJSON(map[string]any{
				"type":     "response.created",
				"response": map[string]any{"id": "resp_final"},
			}),
			mustJSON(map[string]any{
				"type":  "response.output_text.delta",
				"delta": "It's sunny in SF!",
			}),
			mustJSON(map[string]any{
				"type": "response.completed",
				"response": map[string]any{
					"id":    "resp_final",
					"usage": map[string]any{"input_tokens": 25, "output_tokens": 6, "total_tokens": 31},
				},
			}),
		},
	}

	srv := newResponsesTestServer(t, turns)
	defer srv.Close()

	llm := newTestLLMWithWSServer(t, srv)
	ctx := t.Context()

	session, err := llm.OpenResponsesSession(ctx)
	require.NoError(t, err)
	defer session.Close()

	// Turn 1: get tool call.
	resp1, err := session.Send(ctx,
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "What's the weather in SF?"),
		},
	)
	require.NoError(t, err)
	assert.Equal(t, "tool_calls", resp1.Choices[0].StopReason)
	require.Len(t, resp1.Choices[0].ToolCalls, 1)
	assert.Equal(t, "get_weather", resp1.Choices[0].ToolCalls[0].FunctionCall.Name)

	// Turn 2: send tool result.
	resp2, err := session.SendToolResults(ctx,
		[]ToolResult{{CallID: "call_abc", Output: "sunny, 72F"}},
	)
	require.NoError(t, err)
	assert.Equal(t, "It's sunny in SF!", resp2.Choices[0].Content)
	assert.Equal(t, "resp_final", session.LastResponseID())
}

func TestResponsesSession_Streaming(t *testing.T) {
	turns := [][]json.RawMessage{
		{
			mustJSON(map[string]any{
				"type":     "response.created",
				"response": map[string]any{"id": "resp_s"},
			}),
			mustJSON(map[string]any{
				"type":  "response.output_text.delta",
				"delta": "A",
			}),
			mustJSON(map[string]any{
				"type":  "response.output_text.delta",
				"delta": "B",
			}),
			mustJSON(map[string]any{
				"type":  "response.output_text.delta",
				"delta": "C",
			}),
			mustJSON(map[string]any{
				"type": "response.completed",
				"response": map[string]any{
					"id":    "resp_s",
					"usage": map[string]any{"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
				},
			}),
		},
	}

	srv := newResponsesTestServer(t, turns)
	defer srv.Close()

	llm := newTestLLMWithWSServer(t, srv)
	ctx := t.Context()

	session, err := llm.OpenResponsesSession(ctx)
	require.NoError(t, err)
	defer session.Close()

	var chunks []string
	resp, err := session.Send(ctx,
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "stream test"),
		},
		llms.WithStreamingFunc(func(_ context.Context, chunk []byte) error {
			chunks = append(chunks, string(chunk))
			return nil
		}),
	)
	require.NoError(t, err)

	assert.Equal(t, "ABC", resp.Choices[0].Content)
	assert.Equal(t, []string{"A", "B", "C"}, chunks)
}

func TestResponsesSession_MultiTurnAutoChaining(t *testing.T) {
	turns := [][]json.RawMessage{
		{
			mustJSON(map[string]any{
				"type":     "response.created",
				"response": map[string]any{"id": "resp_a"},
			}),
			mustJSON(map[string]any{
				"type":  "response.output_text.delta",
				"delta": "First",
			}),
			mustJSON(map[string]any{
				"type": "response.completed",
				"response": map[string]any{
					"id":    "resp_a",
					"usage": map[string]any{"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
				},
			}),
		},
		{
			mustJSON(map[string]any{
				"type":     "response.created",
				"response": map[string]any{"id": "resp_b"},
			}),
			mustJSON(map[string]any{
				"type":  "response.output_text.delta",
				"delta": "Second",
			}),
			mustJSON(map[string]any{
				"type": "response.completed",
				"response": map[string]any{
					"id":    "resp_b",
					"usage": map[string]any{"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
				},
			}),
		},
	}

	srv := newResponsesTestServer(t, turns)
	defer srv.Close()

	llm := newTestLLMWithWSServer(t, srv)
	ctx := t.Context()

	session, err := llm.OpenResponsesSession(ctx)
	require.NoError(t, err)
	defer session.Close()

	r1, err := session.Send(ctx,
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "Turn 1")},
	)
	require.NoError(t, err)
	assert.Equal(t, "First", r1.Choices[0].Content)

	r2, err := session.Send(ctx,
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "Turn 2")},
	)
	require.NoError(t, err)
	assert.Equal(t, "Second", r2.Choices[0].Content)
	assert.Equal(t, "resp_b", session.LastResponseID())
}
