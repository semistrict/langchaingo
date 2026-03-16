package openai

import (
	"context"
	"encoding/json"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai/internal/openaiclient"
)

// ToolResult is the output of a locally-executed tool call,
// sent back to the model via SendToolResults.
type ToolResult struct {
	CallID string // matches ToolCall.ID from the response
	Output string // the tool's output as a string
}

// ResponsesSession is a persistent WebSocket connection to the OpenAI
// Responses API. It maintains connection state across multiple turns,
// enabling low-latency multi-turn conversations where only incremental
// inputs are sent per turn.
//
// A session automatically chains responses: each Send/SendToolResults call
// sets previous_response_id from the last successful response, so the server
// reuses cached context. Call ResetChain() to start a fresh conversation
// on the same connection.
//
// The session is safe for sequential use but not concurrent — the WebSocket
// protocol processes one response at a time. The 60-minute server-side
// connection limit is enforced by the server.
type ResponsesSession struct {
	session *openaiclient.ResponsesSession
	model   string // default model from the LLM
}

// OpenResponsesSession dials a WebSocket connection to the Responses API.
// The connection stays open until Close() is called or the 60-minute server
// limit is reached.
func (o *LLM) OpenResponsesSession(ctx context.Context) (*ResponsesSession, error) {
	session, err := o.client.OpenResponsesSession(ctx)
	if err != nil {
		return nil, err
	}
	return &ResponsesSession{
		session: session,
		model:   o.model,
	}, nil
}

// Send sends user messages and returns the model's response.
// Model and tools are per-message (set via CallOptions).
// Automatically sets previous_response_id from the last turn.
//
// Returns *llms.ContentResponse with the same shape as GenerateContent —
// text in Choices[0].Content, tool calls in Choices[0].ToolCalls.
func (s *ResponsesSession) Send(
	ctx context.Context,
	messages []llms.MessageContent,
	options ...llms.CallOption,
) (*llms.ContentResponse, error) {
	opts := llms.CallOptions{}
	for _, opt := range options {
		opt(&opts)
	}

	model := opts.Model
	if model == "" {
		model = s.model
	}

	// Convert llms.MessageContent to Responses API input items.
	input := messagesToResponsesInput(messages)

	// Convert llms tools to Responses API tools.
	tools := convertTools(opts.Tools)

	req := &openaiclient.ResponsesCreateRequest{
		Model: model,
		Input: input,
		Tools: tools,
	}

	if opts.StreamingFunc != nil {
		req.StreamingFunc = func(chunk []byte) error {
			return opts.StreamingFunc(ctx, chunk)
		}
	}

	result, err := s.session.SendResponse(ctx, req)
	if err != nil {
		return nil, err
	}

	return responsesResultToContentResponse(result), nil
}

// SendToolResults sends function call outputs back to the model.
// This is the natural continuation after receiving ToolCalls in a response.
// Each ToolResult maps to a function_call_output input item.
func (s *ResponsesSession) SendToolResults(
	ctx context.Context,
	results []ToolResult,
	options ...llms.CallOption,
) (*llms.ContentResponse, error) {
	opts := llms.CallOptions{}
	for _, opt := range options {
		opt(&opts)
	}

	model := opts.Model
	if model == "" {
		model = s.model
	}

	input := make([]openaiclient.ResponsesInputItem, len(results))
	for i, r := range results {
		input[i] = openaiclient.ResponsesInputItem{
			Type:   "function_call_output",
			CallID: r.CallID,
			Output: r.Output,
		}
	}

	tools := convertTools(opts.Tools)

	req := &openaiclient.ResponsesCreateRequest{
		Model: model,
		Input: input,
		Tools: tools,
	}

	if opts.StreamingFunc != nil {
		req.StreamingFunc = func(chunk []byte) error {
			return opts.StreamingFunc(ctx, chunk)
		}
	}

	result, err := s.session.SendResponse(ctx, req)
	if err != nil {
		return nil, err
	}

	return responsesResultToContentResponse(result), nil
}

// ResetChain clears the previous_response_id, so the next Send starts
// a fresh conversation on the same WebSocket connection.
func (s *ResponsesSession) ResetChain() {
	s.session.ResetChain()
}

// LastResponseID returns the ID of the most recent response, useful for
// manual chaining or debugging.
func (s *ResponsesSession) LastResponseID() string {
	return s.session.LastResponseID()
}

// Close closes the WebSocket connection.
func (s *ResponsesSession) Close() error {
	return s.session.Close()
}

// SetRecorder sets a message recorder on the session for testing.
func (s *ResponsesSession) SetRecorder(r openaiclient.MessageRecorder) {
	s.session.SetRecorder(r)
}

// messagesToResponsesInput converts llms.MessageContent to Responses API input items.
func messagesToResponsesInput(messages []llms.MessageContent) []openaiclient.ResponsesInputItem {
	items := make([]openaiclient.ResponsesInputItem, 0, len(messages))
	for _, msg := range messages {
		role := ""
		switch msg.Role {
		case llms.ChatMessageTypeHuman:
			role = "user"
		case llms.ChatMessageTypeAI:
			role = "assistant"
		case llms.ChatMessageTypeSystem:
			role = "system"
		default:
			role = string(msg.Role)
		}

		// Check if this is a tool call response.
		for _, part := range msg.Parts {
			if tcr, ok := part.(llms.ToolCallResponse); ok {
				items = append(items, openaiclient.ResponsesInputItem{
					Type:   "function_call_output",
					CallID: tcr.ToolCallID,
					Output: tcr.Content,
				})
				continue
			}
		}

		// Build text content.
		var content []openaiclient.ResponsesContent
		for _, part := range msg.Parts {
			switch p := part.(type) {
			case llms.TextContent:
				content = append(content, openaiclient.ResponsesContent{
					Type: "input_text",
					Text: p.Text,
				})
			}
		}
		if len(content) > 0 {
			items = append(items, openaiclient.ResponsesInputItem{
				Type:    "message",
				Role:    role,
				Content: content,
			})
		}
	}
	return items
}

// convertTools converts llms.Tool to Responses API tool format.
func convertTools(tools []llms.Tool) []openaiclient.ResponsesTool {
	if len(tools) == 0 {
		return nil
	}
	result := make([]openaiclient.ResponsesTool, len(tools))
	for i, t := range tools {
		rt := openaiclient.ResponsesTool{Type: t.Type}

		if t.Function != nil {
			rt.Name = t.Function.Name
			rt.Description = t.Function.Description
			if t.Function.Parameters != nil {
				switch p := t.Function.Parameters.(type) {
				case json.RawMessage:
					rt.Parameters = p
				case []byte:
					rt.Parameters = p
				default:
					if b, err := json.Marshal(p); err == nil {
						rt.Parameters = b
					}
				}
			}
		}

		result[i] = rt
	}
	return result
}

// responsesResultToContentResponse converts a ResponsesResult to llms.ContentResponse.
func responsesResultToContentResponse(result *openaiclient.ResponsesResult) *llms.ContentResponse {
	choice := &llms.ContentChoice{
		Content:    result.Content,
		StopReason: "stop",
		GenerationInfo: map[string]any{
			"response_id":  result.ResponseID,
			"input_tokens": result.Usage.InputTokens,
			"output_tokens": result.Usage.OutputTokens,
			"total_tokens": result.Usage.TotalTokens,
		},
	}

	if len(result.ToolCalls) > 0 {
		choice.StopReason = "tool_calls"
		choice.ToolCalls = make([]llms.ToolCall, len(result.ToolCalls))
		for i, tc := range result.ToolCalls {
			choice.ToolCalls[i] = llms.ToolCall{
				ID:   tc.ID,
				Type: "function",
				FunctionCall: &llms.FunctionCall{
					Name:      tc.Name,
					Arguments: tc.Arguments,
				},
			}
		}
		if len(result.ToolCalls) > 0 {
			choice.FuncCall = choice.ToolCalls[0].FunctionCall
		}
	}

	return &llms.ContentResponse{
		Choices: []*llms.ContentChoice{choice},
	}
}
