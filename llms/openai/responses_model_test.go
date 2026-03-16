package openai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/llms"
)

// TestResponsesSession_GenerateContent tests that GenerateContent works
// as an llms.Model implementation, returning a response with the ID set.
func TestResponsesSession_GenerateContent(t *testing.T) {
	turns := [][]json.RawMessage{
		{
			mustJSON(map[string]any{
				"type":     "response.created",
				"response": map[string]any{"id": "resp_gc1"},
			}),
			mustJSON(map[string]any{
				"type":  "response.output_text.delta",
				"delta": "Hello via GenerateContent!",
			}),
			mustJSON(map[string]any{
				"type": "response.completed",
				"response": map[string]any{
					"id":    "resp_gc1",
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

	// Use GenerateContent (the llms.Model interface).
	resp, err := session.GenerateContent(ctx,
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "Hello"),
		},
	)
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)
	assert.Equal(t, "Hello via GenerateContent!", resp.Choices[0].Content)
	assert.Equal(t, "resp_gc1", resp.Choices[0].ID, "response ID should be set on ContentChoice")
}

// TestResponsesSession_GenerateContentWithPreviousResponseID verifies that
// WithPreviousResponseID sets previous_response_id on the request.
func TestResponsesSession_GenerateContentWithPreviousResponseID(t *testing.T) {
	var capturedRequest json.RawMessage

	turns := [][]json.RawMessage{
		// Turn 1: normal response.
		{
			mustJSON(map[string]any{
				"type":     "response.created",
				"response": map[string]any{"id": "resp_first"},
			}),
			mustJSON(map[string]any{
				"type":  "response.output_text.delta",
				"delta": "First response",
			}),
			mustJSON(map[string]any{
				"type": "response.completed",
				"response": map[string]any{
					"id":    "resp_first",
					"usage": map[string]any{"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
				},
			}),
		},
		// Turn 2: response with previous_response_id.
		{
			mustJSON(map[string]any{
				"type":     "response.created",
				"response": map[string]any{"id": "resp_second"},
			}),
			mustJSON(map[string]any{
				"type":  "response.output_text.delta",
				"delta": "Second response",
			}),
			mustJSON(map[string]any{
				"type": "response.completed",
				"response": map[string]any{
					"id":    "resp_second",
					"usage": map[string]any{"input_tokens": 8, "output_tokens": 3, "total_tokens": 11},
				},
			}),
		},
	}

	srv := newResponsesTestServerWithCapture(t, turns, &capturedRequest)
	defer srv.Close()

	llm := newTestLLMWithWSServer(t, srv)
	ctx := t.Context()

	session, err := llm.OpenResponsesSession(ctx)
	require.NoError(t, err)
	defer session.Close()

	// Turn 1: send initial message, get response with ID.
	resp1, err := session.GenerateContent(ctx, []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "You are helpful."),
		llms.TextParts(llms.ChatMessageTypeHuman, "Hello"),
	})
	require.NoError(t, err)
	assert.Equal(t, "resp_first", resp1.Choices[0].ID)

	// Turn 2: explicitly pass previous_response_id and only new messages.
	resp2, err := session.GenerateContent(ctx, []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "Follow up"),
	}, llms.WithPreviousResponseID("resp_first"))
	require.NoError(t, err)
	assert.Equal(t, "Second response", resp2.Choices[0].Content)
	assert.Equal(t, "resp_second", resp2.Choices[0].ID)

	// Verify the captured request has previous_response_id set
	// and only contains the follow-up message.
	var req map[string]any
	require.NoError(t, json.Unmarshal(capturedRequest, &req))
	assert.Equal(t, "resp_first", req["previous_response_id"])
	input, ok := req["input"].([]any)
	require.True(t, ok)
	require.Len(t, input, 1, "should only send the new message")
	item, ok := input[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "message", item["type"])
	assert.Equal(t, "user", item["role"])
	content, ok := item["content"].([]any)
	require.True(t, ok)
	require.Len(t, content, 1)
	block, ok := content[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "Follow up", block["text"])
}

// TestResponsesSession_GenerateContentToolLoop tests the full tool call flow
// through GenerateContent, simulating how agents.Executor would use it.
func TestResponsesSession_GenerateContentToolLoop(t *testing.T) {
	turns := [][]json.RawMessage{
		// Turn 1: model requests a tool call.
		{
			mustJSON(map[string]any{
				"type":     "response.created",
				"response": map[string]any{"id": "resp_toolcall"},
			}),
			mustJSON(map[string]any{
				"type": "response.output_item.added",
			}),
			mustJSON(map[string]any{
				"type":  "response.function_call_arguments.delta",
				"delta": `{"__arg1": "2+2"}`,
			}),
			mustJSON(map[string]any{
				"type": "response.output_item.done",
				"item": map[string]any{
					"type":      "function_call",
					"id":        "fc_1",
					"call_id":   "call_calc",
					"name":      "calculator",
					"arguments": `{"__arg1": "2+2"}`,
				},
			}),
			mustJSON(map[string]any{
				"type": "response.completed",
				"response": map[string]any{
					"id":    "resp_toolcall",
					"usage": map[string]any{"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
				},
			}),
		},
		// Turn 2: model responds with final text after tool result.
		{
			mustJSON(map[string]any{
				"type":     "response.created",
				"response": map[string]any{"id": "resp_final"},
			}),
			mustJSON(map[string]any{
				"type":  "response.output_text.delta",
				"delta": "The answer is 4.",
			}),
			mustJSON(map[string]any{
				"type": "response.completed",
				"response": map[string]any{
					"id":    "resp_final",
					"usage": map[string]any{"input_tokens": 30, "output_tokens": 5, "total_tokens": 35},
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

	// Turn 1: initial message → tool call.
	resp1, err := session.GenerateContent(ctx,
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "What is 2+2?"),
		},
	)
	require.NoError(t, err)
	require.Len(t, resp1.Choices[0].ToolCalls, 1)
	assert.Equal(t, "calculator", resp1.Choices[0].ToolCalls[0].FunctionCall.Name)
	assert.Equal(t, "resp_toolcall", resp1.Choices[0].ID)

	// Turn 2: send tool result via GenerateContent (as the executor would,
	// rebuilding the full message history with the AI response ID threaded through).
	resp2, err := session.GenerateContent(ctx,
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "What is 2+2?"),
			// AI message with tool call and response ID.
			{
				Role: llms.ChatMessageTypeAI,
				Parts: []llms.ContentPart{
					llms.ToolCall{
						ID:   "call_calc",
						Type: "function",
						FunctionCall: &llms.FunctionCall{
							Name:      "calculator",
							Arguments: `{"__arg1": "2+2"}`,
						},
					},
				},
				ID: "resp_toolcall",
			},
			// Tool result.
			{
				Role: llms.ChatMessageTypeTool,
				Parts: []llms.ContentPart{
					llms.ToolCallResponse{
						ToolCallID: "call_calc",
						Content:    "4",
					},
				},
			},
		},
	)
	require.NoError(t, err)
	assert.Equal(t, "The answer is 4.", resp2.Choices[0].Content)
	assert.Equal(t, "resp_final", resp2.Choices[0].ID)
}

// TestResponsesSession_Call tests the simplified Call method.
func TestResponsesSession_Call(t *testing.T) {
	turns := [][]json.RawMessage{
		{
			mustJSON(map[string]any{
				"type":     "response.created",
				"response": map[string]any{"id": "resp_call"},
			}),
			mustJSON(map[string]any{
				"type":  "response.output_text.delta",
				"delta": "Hi there!",
			}),
			mustJSON(map[string]any{
				"type": "response.completed",
				"response": map[string]any{
					"id":    "resp_call",
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

	result, err := session.Call(ctx, "Hello")
	require.NoError(t, err)
	assert.Equal(t, "Hi there!", result)
}
