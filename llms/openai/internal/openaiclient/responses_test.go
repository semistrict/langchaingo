package openaiclient

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"nhooyr.io/websocket"
	"nhooyr.io/websocket/wsjson"
)

// newTestWSServer creates a test WebSocket server that sends the given events
// in sequence after receiving a response.create message.
func newTestWSServer(t *testing.T, events []json.RawMessage) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Errorf("websocket accept: %v", err)
			return
		}
		defer conn.Close(websocket.StatusNormalClosure, "done")

		ctx := r.Context()

		// Read the response.create request.
		var req ResponsesCreateRequest
		if err := wsjson.Read(ctx, conn, &req); err != nil {
			t.Errorf("read request: %v", err)
			return
		}

		// Send all events.
		for _, event := range events {
			if err := wsjson.Write(ctx, conn, event); err != nil {
				t.Errorf("write event: %v", err)
				return
			}
		}
	}))
}

func mustJSON(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}

func TestResponsesSession_BasicChat(t *testing.T) {
	events := []json.RawMessage{
		mustJSON(map[string]any{
			"type":     "response.created",
			"response": map[string]any{"id": "resp_abc123"},
		}),
		mustJSON(map[string]any{
			"type":  "response.output_text.delta",
			"delta": "Hello, ",
		}),
		mustJSON(map[string]any{
			"type":  "response.output_text.delta",
			"delta": "world!",
		}),
		mustJSON(map[string]any{
			"type": "response.completed",
			"response": map[string]any{
				"id": "resp_abc123",
				"usage": map[string]any{
					"input_tokens":  10,
					"output_tokens": 5,
					"total_tokens":  15,
				},
			},
		}),
	}

	srv := newTestWSServer(t, events)
	defer srv.Close()

	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")

	ctx := t.Context()
	session, err := DialResponsesSession(ctx, wsURL, "test-token", nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer session.Close()

	result, err := session.SendResponse(ctx, &ResponsesCreateRequest{
		Model: "gpt-5.4",
		Input: []ResponsesInputItem{
			{Type: "message", Role: "user", Content: []ResponsesContent{{Type: "input_text", Text: "Hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("send: %v", err)
	}

	if result.ResponseID != "resp_abc123" {
		t.Errorf("response ID = %q, want %q", result.ResponseID, "resp_abc123")
	}
	if result.Content != "Hello, world!" {
		t.Errorf("content = %q, want %q", result.Content, "Hello, world!")
	}
	if result.Usage.TotalTokens != 15 {
		t.Errorf("total tokens = %d, want %d", result.Usage.TotalTokens, 15)
	}
}

func TestResponsesSession_ToolCall(t *testing.T) {
	events := []json.RawMessage{
		mustJSON(map[string]any{
			"type":     "response.created",
			"response": map[string]any{"id": "resp_tool1"},
		}),
		mustJSON(map[string]any{
			"type": "response.output_item.added",
		}),
		mustJSON(map[string]any{
			"type":  "response.function_call_arguments.delta",
			"delta": `{"location":`,
		}),
		mustJSON(map[string]any{
			"type":  "response.function_call_arguments.delta",
			"delta": ` "Tokyo"}`,
		}),
		mustJSON(map[string]any{
			"type": "response.output_item.done",
			"item": map[string]any{
				"type":      "function_call",
				"id":        "fc_1",
				"call_id":   "call_xyz",
				"name":      "get_weather",
				"arguments": `{"location": "Tokyo"}`,
			},
		}),
		mustJSON(map[string]any{
			"type": "response.completed",
			"response": map[string]any{
				"id":    "resp_tool1",
				"usage": map[string]any{"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
			},
		}),
	}

	srv := newTestWSServer(t, events)
	defer srv.Close()

	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")

	ctx := t.Context()
	session, err := DialResponsesSession(ctx, wsURL, "test-token", nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer session.Close()

	result, err := session.SendResponse(ctx, &ResponsesCreateRequest{
		Model: "gpt-5.4",
		Input: []ResponsesInputItem{
			{Type: "message", Role: "user", Content: []ResponsesContent{{Type: "input_text", Text: "Weather in Tokyo?"}}},
		},
	})
	if err != nil {
		t.Fatalf("send: %v", err)
	}

	if len(result.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(result.ToolCalls))
	}
	tc := result.ToolCalls[0]
	if tc.Name != "get_weather" {
		t.Errorf("tool name = %q, want %q", tc.Name, "get_weather")
	}
	if tc.ID != "call_xyz" {
		t.Errorf("tool call ID = %q, want %q", tc.ID, "call_xyz")
	}
	if tc.Arguments != `{"location": "Tokyo"}` {
		t.Errorf("tool args = %q, want %q", tc.Arguments, `{"location": "Tokyo"}`)
	}
}

func TestResponsesSession_AutoChaining(t *testing.T) {
	// First turn events.
	turn1Events := []json.RawMessage{
		mustJSON(map[string]any{
			"type":     "response.created",
			"response": map[string]any{"id": "resp_turn1"},
		}),
		mustJSON(map[string]any{
			"type":  "response.output_text.delta",
			"delta": "Turn 1",
		}),
		mustJSON(map[string]any{
			"type": "response.completed",
			"response": map[string]any{
				"id":    "resp_turn1",
				"usage": map[string]any{"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
			},
		}),
	}

	// We need a server that handles two rounds. Use a stateful handler.
	turn := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Errorf("websocket accept: %v", err)
			return
		}
		defer conn.Close(websocket.StatusNormalClosure, "done")

		ctx := r.Context()

		for {
			var req ResponsesCreateRequest
			if err := wsjson.Read(ctx, conn, &req); err != nil {
				return // connection closed
			}

			var events []json.RawMessage
			if turn == 0 {
				events = turn1Events
				// Verify no previous_response_id on first turn.
				if req.PreviousResponseID != "" {
					t.Errorf("turn 0: expected empty previous_response_id, got %q", req.PreviousResponseID)
				}
			} else {
				// Verify previous_response_id is set on second turn.
				if req.PreviousResponseID != "resp_turn1" {
					t.Errorf("turn 1: expected previous_response_id=%q, got %q", "resp_turn1", req.PreviousResponseID)
				}
				events = []json.RawMessage{
					mustJSON(map[string]any{
						"type":     "response.created",
						"response": map[string]any{"id": "resp_turn2"},
					}),
					mustJSON(map[string]any{
						"type":  "response.output_text.delta",
						"delta": "Turn 2",
					}),
					mustJSON(map[string]any{
						"type": "response.completed",
						"response": map[string]any{
							"id":    "resp_turn2",
							"usage": map[string]any{"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
						},
					}),
				}
			}
			turn++

			for _, event := range events {
				if err := wsjson.Write(ctx, conn, event); err != nil {
					return
				}
			}
		}
	}))
	defer srv.Close()

	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")

	ctx := t.Context()
	session, err := DialResponsesSession(ctx, wsURL, "test-token", nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer session.Close()

	// Turn 1
	r1, err := session.SendResponse(ctx, &ResponsesCreateRequest{
		Model: "gpt-5.4",
		Input: []ResponsesInputItem{
			{Type: "message", Role: "user", Content: []ResponsesContent{{Type: "input_text", Text: "Hello"}}},
		},
	})
	if err != nil {
		t.Fatalf("turn 1: %v", err)
	}
	if r1.Content != "Turn 1" {
		t.Errorf("turn 1 content = %q, want %q", r1.Content, "Turn 1")
	}
	if session.LastResponseID() != "resp_turn1" {
		t.Errorf("last response ID = %q, want %q", session.LastResponseID(), "resp_turn1")
	}

	// Turn 2 — previous_response_id should be set automatically.
	r2, err := session.SendResponse(ctx, &ResponsesCreateRequest{
		Model: "gpt-5.4",
		Input: []ResponsesInputItem{
			{Type: "message", Role: "user", Content: []ResponsesContent{{Type: "input_text", Text: "Follow up"}}},
		},
	})
	if err != nil {
		t.Fatalf("turn 2: %v", err)
	}
	if r2.Content != "Turn 2" {
		t.Errorf("turn 2 content = %q, want %q", r2.Content, "Turn 2")
	}
}

func TestResponsesSession_Streaming(t *testing.T) {
	events := []json.RawMessage{
		mustJSON(map[string]any{
			"type":     "response.created",
			"response": map[string]any{"id": "resp_stream"},
		}),
		mustJSON(map[string]any{
			"type":  "response.output_text.delta",
			"delta": "chunk1",
		}),
		mustJSON(map[string]any{
			"type":  "response.output_text.delta",
			"delta": "chunk2",
		}),
		mustJSON(map[string]any{
			"type": "response.completed",
			"response": map[string]any{
				"id":    "resp_stream",
				"usage": map[string]any{"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
			},
		}),
	}

	srv := newTestWSServer(t, events)
	defer srv.Close()

	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")

	ctx := t.Context()
	session, err := DialResponsesSession(ctx, wsURL, "test-token", nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer session.Close()

	var chunks []string
	result, err := session.SendResponse(ctx, &ResponsesCreateRequest{
		Model: "gpt-5.4",
		Input: []ResponsesInputItem{
			{Type: "message", Role: "user", Content: []ResponsesContent{{Type: "input_text", Text: "Stream test"}}},
		},
		StreamingFunc: func(chunk []byte) error {
			chunks = append(chunks, string(chunk))
			return nil
		},
	})
	if err != nil {
		t.Fatalf("send: %v", err)
	}

	if result.Content != "chunk1chunk2" {
		t.Errorf("content = %q, want %q", result.Content, "chunk1chunk2")
	}
	if len(chunks) != 2 {
		t.Fatalf("chunks = %d, want 2", len(chunks))
	}
	if chunks[0] != "chunk1" || chunks[1] != "chunk2" {
		t.Errorf("chunks = %v, want [chunk1 chunk2]", chunks)
	}
}

func TestResponsesSession_ErrorEvent(t *testing.T) {
	events := []json.RawMessage{
		mustJSON(map[string]any{
			"type":   "error",
			"status": 400,
			"error": map[string]any{
				"type":    "invalid_request_error",
				"code":    "previous_response_not_found",
				"message": "Previous response with id 'resp_xyz' not found.",
			},
		}),
	}

	srv := newTestWSServer(t, events)
	defer srv.Close()

	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")

	ctx := t.Context()
	session, err := DialResponsesSession(ctx, wsURL, "test-token", nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer session.Close()

	_, err = session.SendResponse(ctx, &ResponsesCreateRequest{
		Model: "gpt-5.4",
		Input: []ResponsesInputItem{
			{Type: "message", Role: "user", Content: []ResponsesContent{{Type: "input_text", Text: "test"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "previous_response_not_found") {
		t.Errorf("error = %q, expected to contain 'previous_response_not_found'", err.Error())
	}
}

func TestResponsesSession_ResetChain(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Errorf("websocket accept: %v", err)
			return
		}
		defer conn.Close(websocket.StatusNormalClosure, "done")

		ctx := r.Context()
		reqNum := 0
		for {
			var req ResponsesCreateRequest
			if err := wsjson.Read(ctx, conn, &req); err != nil {
				return
			}

			if reqNum == 0 {
				// First request — no previous_response_id expected.
			} else if reqNum == 1 {
				// After ResetChain — should have no previous_response_id.
				if req.PreviousResponseID != "" {
					t.Errorf("after reset: expected empty previous_response_id, got %q", req.PreviousResponseID)
				}
			}
			reqNum++

			respID := "resp_" + strings.Repeat("x", reqNum)
			events := []json.RawMessage{
				mustJSON(map[string]any{
					"type":     "response.created",
					"response": map[string]any{"id": respID},
				}),
				mustJSON(map[string]any{
					"type":  "response.output_text.delta",
					"delta": "ok",
				}),
				mustJSON(map[string]any{
					"type": "response.completed",
					"response": map[string]any{
						"id":    respID,
						"usage": map[string]any{"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
					},
				}),
			}
			for _, event := range events {
				if err := wsjson.Write(ctx, conn, event); err != nil {
					return
				}
			}
		}
	}))
	defer srv.Close()

	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")

	ctx := t.Context()
	session, err := DialResponsesSession(ctx, wsURL, "test-token", nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer session.Close()

	// First request.
	_, err = session.SendResponse(ctx, &ResponsesCreateRequest{
		Model: "gpt-5.4",
		Input: []ResponsesInputItem{
			{Type: "message", Role: "user", Content: []ResponsesContent{{Type: "input_text", Text: "first"}}},
		},
	})
	if err != nil {
		t.Fatalf("first request: %v", err)
	}

	// Reset chain and send again.
	session.ResetChain()

	_, err = session.SendResponse(ctx, &ResponsesCreateRequest{
		Model: "gpt-5.4",
		Input: []ResponsesInputItem{
			{Type: "message", Role: "user", Content: []ResponsesContent{{Type: "input_text", Text: "fresh start"}}},
		},
	})
	if err != nil {
		t.Fatalf("after reset: %v", err)
	}
}
