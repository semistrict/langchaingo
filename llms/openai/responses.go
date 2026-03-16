package openai

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai/internal/openaiclient"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

const tracerName = "github.com/tmc/langchaingo/llms/openai"

// ToolResult is the output of a locally-executed tool call,
// sent back to the model via SendToolResults.
type ToolResult struct {
	CallID string // matches ToolCall.ID from the response
	Output string // the tool's output as a string
}

// responseSender abstracts how responses are sent — via WebSocket or HTTP.
type responseSender interface {
	SendResponse(ctx context.Context, req *openaiclient.ResponsesCreateRequest) (*openaiclient.ResponsesResult, error)
	Close() error
}

// httpResponseSender sends responses via HTTP POST with SSE streaming.
// Unlike the WebSocket sender, this does NOT auto-chain responses —
// the caller must explicitly set PreviousResponseID on the request
// (via MessageContent.ID) if the backend supports it.
type httpResponseSender struct {
	client     *openaiclient.Client
	lastRespID string
}

func (h *httpResponseSender) SendResponse(ctx context.Context, req *openaiclient.ResponsesCreateRequest) (*openaiclient.ResponsesResult, error) {
	result, err := h.client.SendResponseHTTP(ctx, req)
	if err != nil {
		return nil, err
	}
	h.lastRespID = result.ResponseID
	return result, nil
}

func (h *httpResponseSender) Close() error { return nil }

// ResponsesSession is a connection to the OpenAI Responses API.
// It supports two transports:
//   - WebSocket (persistent connection, used with OpenAI API)
//   - HTTP POST with SSE (used with ChatGPT backend)
//
// A session automatically chains responses: each Send/SendToolResults call
// sets previous_response_id from the last successful response, so the server
// reuses cached context. Call ResetChain() to start a fresh conversation
// on the same connection.
type ResponsesSession struct {
	sender      responseSender
	session     *openaiclient.ResponsesSession // non-nil for WebSocket transport
	model       string                         // default model from the LLM
	sessionCtx  context.Context                // context carrying the session span
	sessionSpan trace.Span
}

// OpenResponsesSession dials a WebSocket connection to the Responses API.
// The connection stays open until Close() is called or the 60-minute server
// limit is reached.
func (o *LLM) OpenResponsesSession(ctx context.Context) (*ResponsesSession, error) {
	tracer := otel.Tracer(tracerName)
	ctx, sessionSpan := tracer.Start(ctx, "responses.session",
		trace.WithAttributes(
			attribute.String("gen_ai.system", "openai"),
			attribute.String("gen_ai.request.model", o.model),
			attribute.String("langsmith.span.kind", "chain"),
		),
	)

	session, err := o.client.OpenResponsesSession(ctx)
	if err != nil {
		sessionSpan.RecordError(err)
		sessionSpan.SetStatus(codes.Error, err.Error())
		sessionSpan.End()
		return nil, err
	}
	return &ResponsesSession{
		sender:      session,
		session:     session,
		model:       o.model,
		sessionCtx:  ctx,
		sessionSpan: sessionSpan,
	}, nil
}

// OpenResponsesHTTPSession creates a session that uses HTTP POST with SSE
// streaming for the Responses API. This is used for backends that don't
// support WebSocket (e.g., the ChatGPT backend at chatgpt.com).
func (o *LLM) OpenResponsesHTTPSession(ctx context.Context) (*ResponsesSession, error) {
	tracer := otel.Tracer(tracerName)
	ctx, sessionSpan := tracer.Start(ctx, "responses.session",
		trace.WithAttributes(
			attribute.String("gen_ai.system", "openai"),
			attribute.String("gen_ai.request.model", o.model),
			attribute.String("langsmith.span.kind", "chain"),
		),
	)

	return &ResponsesSession{
		sender:      &httpResponseSender{client: o.client},
		model:       o.model,
		sessionCtx:  ctx,
		sessionSpan: sessionSpan,
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

	tracer := otel.Tracer(tracerName)
	spanCtx := s.spanContext(ctx)
	spanCtx, span := tracer.Start(spanCtx, "responses.send",
		trace.WithAttributes(
			attribute.String("gen_ai.system", "openai"),
			attribute.String("gen_ai.operation.name", "chat"),
			attribute.String("gen_ai.request.model", model),
			attribute.String("langsmith.span.kind", "llm"),
		),
	)
	defer span.End()
	_ = spanCtx // span context used only for parenting

	// Record input messages as span attributes.
	for i, msg := range messages {
		for _, part := range msg.Parts {
			if tc, ok := part.(llms.TextContent); ok {
				span.SetAttributes(
					attribute.String(fmt.Sprintf("gen_ai.prompt.%d.role", i), string(msg.Role)),
					attribute.String(fmt.Sprintf("gen_ai.prompt.%d.content", i), tc.Text),
				)
			}
		}
	}

	// Extract system messages as instructions for the Responses API.
	instructions, filtered := extractInstructions(messages)
	input := messagesToResponsesInput(filtered)
	tools := convertTools(opts.Tools)

	req := &openaiclient.ResponsesCreateRequest{
		Model:        model,
		Input:        input,
		Tools:        tools,
		Instructions: instructions,
	}

	if opts.StreamingFunc != nil {
		req.StreamingFunc = func(chunk []byte) error {
			return opts.StreamingFunc(ctx, chunk)
		}
	}

	result, err := s.sender.SendResponse(ctx, req)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		return nil, err
	}

	span.SetAttributes(
		attribute.Int("gen_ai.usage.input_tokens", result.Usage.InputTokens),
		attribute.Int("gen_ai.usage.output_tokens", result.Usage.OutputTokens),
		attribute.String("gen_ai.completion.0.content", result.Content),
		attribute.String("gen_ai.response.id", result.ResponseID),
	)
	if len(result.ToolCalls) > 0 {
		for i, tc := range result.ToolCalls {
			span.SetAttributes(
				attribute.String(fmt.Sprintf("gen_ai.completion.tool_calls.%d.name", i), tc.Name),
				attribute.String(fmt.Sprintf("gen_ai.completion.tool_calls.%d.arguments", i), tc.Arguments),
			)
		}
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

	tracer := otel.Tracer(tracerName)
	spanCtx := s.spanContext(ctx)
	spanCtx, span := tracer.Start(spanCtx, "responses.send_tool_results",
		trace.WithAttributes(
			attribute.String("gen_ai.system", "openai"),
			attribute.String("gen_ai.operation.name", "chat"),
			attribute.String("gen_ai.request.model", model),
			attribute.String("langsmith.span.kind", "llm"),
		),
	)
	defer span.End()
	_ = spanCtx

	input := make([]openaiclient.ResponsesInputItem, len(results))
	for i, r := range results {
		input[i] = openaiclient.ResponsesInputItem{
			Type:   "function_call_output",
			CallID: r.CallID,
			Output: r.Output,
		}
		span.SetAttributes(
			attribute.String(fmt.Sprintf("gen_ai.tool_result.%d.call_id", i), r.CallID),
			attribute.String(fmt.Sprintf("gen_ai.tool_result.%d.output", i), r.Output),
		)
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

	result, err := s.sender.SendResponse(ctx, req)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		return nil, err
	}

	span.SetAttributes(
		attribute.Int("gen_ai.usage.input_tokens", result.Usage.InputTokens),
		attribute.Int("gen_ai.usage.output_tokens", result.Usage.OutputTokens),
		attribute.String("gen_ai.completion.0.content", result.Content),
		attribute.String("gen_ai.response.id", result.ResponseID),
	)

	return responsesResultToContentResponse(result), nil
}

// ResetChain clears the previous_response_id, so the next Send starts
// a fresh conversation on the same connection.
func (s *ResponsesSession) ResetChain() {
	if s.session != nil {
		s.session.ResetChain()
	}
	if h, ok := s.sender.(*httpResponseSender); ok {
		h.lastRespID = ""
	}
}

// LastResponseID returns the ID of the most recent response, useful for
// manual chaining or debugging.
func (s *ResponsesSession) LastResponseID() string {
	if s.session != nil {
		return s.session.LastResponseID()
	}
	if h, ok := s.sender.(*httpResponseSender); ok {
		return h.lastRespID
	}
	return ""
}

// Compile-time check that ResponsesSession implements llms.Model.
var _ llms.Model = (*ResponsesSession)(nil)

// GenerateContent implements llms.Model. It sends messages via the Responses
// API (WebSocket or HTTP). Use llms.WithPreviousResponseID to chain requests
// for server-side context caching on backends that support it.
func (s *ResponsesSession) GenerateContent(
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

	// Extract system messages as instructions for the Responses API.
	instructions, filtered := extractInstructions(messages)
	input := messagesToResponsesInput(filtered)
	tools := convertTools(opts.Tools)

	req := &openaiclient.ResponsesCreateRequest{
		Model:              model,
		Input:              input,
		Tools:              tools,
		Instructions:       instructions,
		PreviousResponseID: opts.PreviousResponseID,
	}

	if opts.StreamingFunc != nil {
		req.StreamingFunc = func(chunk []byte) error {
			return opts.StreamingFunc(ctx, chunk)
		}
	}

	// OTel tracing.
	tracer := otel.Tracer(tracerName)
	spanCtx := s.spanContext(ctx)
	spanCtx, span := tracer.Start(spanCtx, "responses.generate_content",
		trace.WithAttributes(
			attribute.String("gen_ai.system", "openai"),
			attribute.String("gen_ai.operation.name", "chat"),
			attribute.String("gen_ai.request.model", model),
			attribute.String("langsmith.span.kind", "llm"),
		),
	)
	defer span.End()
	_ = spanCtx

	result, err := s.sender.SendResponse(ctx, req)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		return nil, err
	}

	span.SetAttributes(
		attribute.Int("gen_ai.usage.input_tokens", result.Usage.InputTokens),
		attribute.Int("gen_ai.usage.output_tokens", result.Usage.OutputTokens),
		attribute.String("gen_ai.response.id", result.ResponseID),
	)

	resp := responsesResultToContentResponse(result)
	if len(resp.Choices) > 0 {
		resp.Choices[0].ID = result.ResponseID
	}
	return resp, nil
}

// Call implements llms.Model. It sends a single text prompt and returns
// the model's text response.
func (s *ResponsesSession) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	resp, err := s.GenerateContent(ctx, []llms.MessageContent{
		{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextContent{Text: prompt}}},
	}, options...)
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("empty response from model")
	}
	return resp.Choices[0].Content, nil
}

// Close closes the connection and ends the session trace span.
func (s *ResponsesSession) Close() error {
	if s.sessionSpan != nil {
		s.sessionSpan.End()
	}
	return s.sender.Close()
}

// spanContext returns a context that carries the session span as parent,
// so child spans (Send, SendToolResults) are nested under the session.
func (s *ResponsesSession) spanContext(ctx context.Context) context.Context {
	if s.sessionSpan != nil {
		return trace.ContextWithSpan(ctx, s.sessionSpan)
	}
	return ctx
}

// SetRecorder sets a message recorder on the session for testing.
// Only works with WebSocket-based sessions.
func (s *ResponsesSession) SetRecorder(r openaiclient.MessageRecorder) {
	if s.session != nil {
		s.session.SetRecorder(r)
	}
}

// extractInstructions pulls system messages out of the message list and
// concatenates them into a single instructions string. The remaining non-system
// messages are returned. The Responses API uses an "instructions" field
// instead of system messages in the input array.
func extractInstructions(messages []llms.MessageContent) (string, []llms.MessageContent) {
	var instructions []string
	var remaining []llms.MessageContent
	for _, msg := range messages {
		if msg.Role == llms.ChatMessageTypeSystem {
			for _, part := range msg.Parts {
				if tc, ok := part.(llms.TextContent); ok {
					instructions = append(instructions, tc.Text)
				}
			}
		} else {
			remaining = append(remaining, msg)
		}
	}
	if len(instructions) == 0 {
		return "", messages
	}
	joined := instructions[0]
	for _, s := range instructions[1:] {
		joined += "\n" + s
	}
	return joined, remaining
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
		// Assistant messages use "output_text"; user/system use "input_text".
		contentType := "input_text"
		if msg.Role == llms.ChatMessageTypeAI {
			contentType = "output_text"
		}
		var content []openaiclient.ResponsesContent
		for _, part := range msg.Parts {
			switch p := part.(type) {
			case llms.TextContent:
				content = append(content, openaiclient.ResponsesContent{
					Type: contentType,
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
