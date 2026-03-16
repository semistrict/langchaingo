package openaiclient

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"nhooyr.io/websocket"
	"nhooyr.io/websocket/wsjson"
)

// MessageRecorder captures WebSocket messages for record/replay testing.
type MessageRecorder interface {
	RecordSend(msg json.RawMessage)
	RecordRecv(msg json.RawMessage)
}

// ResponsesSession manages a persistent WebSocket connection to the
// OpenAI Responses API. It processes one response at a time and
// automatically chains responses via previous_response_id.
type ResponsesSession struct {
	conn       *websocket.Conn
	lastRespID string
	recorder   MessageRecorder // optional, for recording WebSocket messages
	mu         sync.Mutex
}

// SetRecorder sets a message recorder on the session for testing.
func (s *ResponsesSession) SetRecorder(r MessageRecorder) {
	s.recorder = r
}

// DialResponsesSession opens a WebSocket connection to the Responses API.
func DialResponsesSession(ctx context.Context, wsURL, token string) (*ResponsesSession, error) {
	conn, _, err := websocket.Dial(ctx, wsURL, &websocket.DialOptions{
		HTTPHeader: http.Header{
			"Authorization": []string{"Bearer " + token},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("websocket dial: %w", err)
	}
	// Allow large messages for long responses.
	conn.SetReadLimit(10 * 1024 * 1024) // 10 MB
	return &ResponsesSession{conn: conn}, nil
}

// SendResponse sends a response.create message and reads streaming events
// until the response is complete. It returns the aggregated result.
//
// This method is not safe for concurrent use — the WebSocket protocol
// processes one response at a time.
func (s *ResponsesSession) SendResponse(ctx context.Context, req *ResponsesCreateRequest) (*ResponsesResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Set type and auto-chain from last response.
	req.Type = "response.create"
	if req.PreviousResponseID == "" && s.lastRespID != "" {
		req.PreviousResponseID = s.lastRespID
	}

	// Send the request.
	if s.recorder != nil {
		if b, err := json.Marshal(req); err == nil {
			s.recorder.RecordSend(b)
		}
	}
	if err := wsjson.Write(ctx, s.conn, req); err != nil {
		return nil, fmt.Errorf("websocket write: %w", err)
	}

	// Read events until response.completed or error.
	return s.readEvents(ctx, req.StreamingFunc)
}

// ResetChain clears the previous_response_id so the next SendResponse
// starts a fresh conversation on the same connection.
func (s *ResponsesSession) ResetChain() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastRespID = ""
}

// LastResponseID returns the ID of the most recent completed response.
func (s *ResponsesSession) LastResponseID() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastRespID
}

// Close closes the WebSocket connection.
func (s *ResponsesSession) Close() error {
	return s.conn.Close(websocket.StatusNormalClosure, "closing")
}

// readEvents reads WebSocket messages and accumulates the response.
func (s *ResponsesSession) readEvents(ctx context.Context, streamingFunc func([]byte) error) (*ResponsesResult, error) {
	var (
		result       ResponsesResult
		contentBuf   strings.Builder
		toolCallArgs = make(map[int]*strings.Builder) // index -> args builder
		toolCalls    []ResponsesToolCall
		toolCallIdx  int
	)

	for {
		// Read raw JSON message.
		var raw json.RawMessage
		if err := wsjson.Read(ctx, s.conn, &raw); err != nil {
			return nil, fmt.Errorf("websocket read: %w", err)
		}
		if s.recorder != nil {
			s.recorder.RecordRecv(json.RawMessage(append([]byte(nil), raw...)))
		}

		// Peek at the event type.
		var event responsesEvent
		if err := json.Unmarshal(raw, &event); err != nil {
			return nil, fmt.Errorf("unmarshal event type: %w", err)
		}

		switch event.Type {
		case "response.created":
			var created responsesCreatedEvent
			if err := json.Unmarshal(raw, &created); err != nil {
				return nil, fmt.Errorf("unmarshal response.created: %w", err)
			}
			result.ResponseID = created.Response.ID

		case "response.output_text.delta":
			var delta responsesOutputTextDelta
			if err := json.Unmarshal(raw, &delta); err != nil {
				return nil, fmt.Errorf("unmarshal text delta: %w", err)
			}
			contentBuf.WriteString(delta.Delta)
			if streamingFunc != nil {
				if err := streamingFunc([]byte(delta.Delta)); err != nil {
					return nil, err
				}
			}

		case "response.function_call_arguments.delta":
			var delta responsesFunctionCallArgsDelta
			if err := json.Unmarshal(raw, &delta); err != nil {
				return nil, fmt.Errorf("unmarshal function call args delta: %w", err)
			}
			builder, ok := toolCallArgs[toolCallIdx]
			if !ok {
				builder = &strings.Builder{}
				toolCallArgs[toolCallIdx] = builder
			}
			builder.WriteString(delta.Delta)

		case "response.output_item.added":
			// A new output item is starting. Track the index for
			// accumulating function call arguments.
			// We don't need to parse the full item here.

		case "response.output_item.done":
			var done responsesOutputItemDone
			if err := json.Unmarshal(raw, &done); err != nil {
				return nil, fmt.Errorf("unmarshal output item done: %w", err)
			}
			if done.Item.Type == "function_call" {
				args := done.Item.Arguments
				if builder, ok := toolCallArgs[toolCallIdx]; ok {
					args = builder.String()
				}
				toolCalls = append(toolCalls, ResponsesToolCall{
					ID:        done.Item.CallID,
					Name:      done.Item.Name,
					Arguments: args,
				})
				toolCallIdx++
			}

		case "response.completed":
			var completed responsesCompletedEvent
			if err := json.Unmarshal(raw, &completed); err != nil {
				return nil, fmt.Errorf("unmarshal response.completed: %w", err)
			}
			result.Content = contentBuf.String()
			result.ToolCalls = toolCalls
			result.Usage = completed.Response.Usage
			s.lastRespID = completed.Response.ID
			return &result, nil

		case "error":
			var errEvent responsesErrorEvent
			if err := json.Unmarshal(raw, &errEvent); err != nil {
				return nil, fmt.Errorf("unmarshal error event: %w", err)
			}
			return nil, fmt.Errorf("responses API error (status %d, code %s): %s",
				errEvent.Status, errEvent.Error.Code, errEvent.Error.Message)

		default:
			// Ignore unknown events for forward compatibility.
		}
	}
}
