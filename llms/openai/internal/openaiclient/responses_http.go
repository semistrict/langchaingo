package openaiclient

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// responsesHTTPRequest is the JSON body for the HTTP Responses API.
// It's similar to ResponsesCreateRequest but without the WebSocket "type" field
// and with stream control.
type responsesHTTPRequest struct {
	Model              string               `json:"model"`
	Input              []ResponsesInputItem `json:"input"`
	Tools              []ResponsesTool      `json:"tools,omitempty"`
	PreviousResponseID string               `json:"previous_response_id,omitempty"`
	Instructions       string               `json:"instructions"`          // always sent, even if empty
	Store              bool                 `json:"store"`                  // always sent; ChatGPT requires false
	Stream             bool                 `json:"stream"`
}

// SendResponseHTTP sends a Responses API request via HTTP POST with SSE streaming.
// This is used for backends that don't support WebSocket (e.g., ChatGPT backend).
func (c *Client) SendResponseHTTP(ctx context.Context, req *ResponsesCreateRequest) (*ResponsesResult, error) {
	httpReq := responsesHTTPRequest{
		Model:              req.Model,
		Input:              req.Input,
		Tools:              req.Tools,
		PreviousResponseID: req.PreviousResponseID,
		Instructions:       req.Instructions,
		Store:              false, // ChatGPT backend requires store=false
		Stream:             true,  // always stream to get SSE events
	}

	body, err := json.Marshal(httpReq)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.buildURL("/responses", req.Model), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	if err := c.setHeaders(httpRequest); err != nil {
		return nil, err
	}

	resp, err := c.httpClient.Do(httpRequest)
	if err != nil {
		return nil, sanitizeHTTPError(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("responses API returned %d: %s", resp.StatusCode, string(respBody))
	}

	return readSSEEvents(resp.Body, req.StreamingFunc)
}

// readSSEEvents reads Server-Sent Events from an HTTP response body and
// accumulates the result. The event JSON payloads are identical to the
// WebSocket events, so we reuse the same types.
func readSSEEvents(body io.Reader, streamingFunc func([]byte) error) (*ResponsesResult, error) {
	var (
		result       ResponsesResult
		contentBuf   strings.Builder
		toolCallArgs = make(map[int]*strings.Builder)
		toolCalls    []ResponsesToolCall
		toolCallIdx  int
	)

	scanner := bufio.NewScanner(body)
	// SSE can have large data lines.
	scanner.Buffer(make([]byte, 0, 64*1024), 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Text()

		// SSE format: "event: <type>\ndata: <json>\n\n"
		// We only care about "data:" lines.
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		// "[DONE]" signals end of stream (OpenAI convention).
		if data == "[DONE]" {
			break
		}

		var event responsesEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue // skip malformed events
		}

		raw := []byte(data)

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
			// New output item starting.

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
			return &result, nil

		case "error":
			var errEvent responsesErrorEvent
			if err := json.Unmarshal(raw, &errEvent); err != nil {
				return nil, fmt.Errorf("unmarshal error event: %w", err)
			}
			return nil, fmt.Errorf("responses API error (code %s): %s",
				errEvent.Error.Code, errEvent.Error.Message)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read SSE stream: %w", err)
	}

	// If we got content but never saw response.completed, return what we have.
	if contentBuf.Len() > 0 || len(toolCalls) > 0 {
		result.Content = contentBuf.String()
		result.ToolCalls = toolCalls
		return &result, nil
	}

	return nil, fmt.Errorf("SSE stream ended without response.completed")
}
