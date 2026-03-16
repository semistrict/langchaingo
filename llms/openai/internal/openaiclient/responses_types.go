package openaiclient

import "encoding/json"

// ResponsesCreateRequest is sent to the server as a WebSocket message.
// It maps to the "response.create" event in the Responses API.
type ResponsesCreateRequest struct {
	Type               string               `json:"type"` // always "response.create"
	Model              string               `json:"model"`
	Input              []ResponsesInputItem  `json:"input"`
	Tools              []ResponsesTool       `json:"tools,omitempty"`
	PreviousResponseID string               `json:"previous_response_id,omitempty"`
	Store              *bool                `json:"store,omitempty"`
	ReasoningEffort    string               `json:"reasoning_effort,omitempty"`
	StreamingFunc      func([]byte) error   `json:"-"`
}

// ResponsesInputItem is a union type for input items in a Responses API request.
// It can be a user/system message or a function_call_output.
type ResponsesInputItem struct {
	Type    string             `json:"type"` // "message" or "function_call_output"
	Role    string             `json:"role,omitempty"`
	Content []ResponsesContent `json:"content,omitempty"`
	CallID  string             `json:"call_id,omitempty"`
	Output  string             `json:"output,omitempty"`
}

// ResponsesContent is a content block within a message input item.
type ResponsesContent struct {
	Type string `json:"type"` // "input_text"
	Text string `json:"text,omitempty"`
}

// ResponsesTool defines a tool available in the Responses API.
type ResponsesTool struct {
	Type        string          `json:"type"`                  // "function", "web_search"
	Name        string          `json:"name,omitempty"`        // for function tools
	Description string          `json:"description,omitempty"` // for function tools
	Parameters  json.RawMessage `json:"parameters,omitempty"`  // JSON Schema for function tools
	Strict      bool            `json:"strict,omitempty"`      // for function tools
}

// responsesEvent is the generic envelope for all server-sent events.
type responsesEvent struct {
	Type string `json:"type"`
}

// responsesCreatedEvent is sent when the server creates a new response.
type responsesCreatedEvent struct {
	Type     string `json:"type"`
	Response struct {
		ID string `json:"id"`
	} `json:"response"`
}

// responsesOutputTextDelta carries incremental text content.
type responsesOutputTextDelta struct {
	Type  string `json:"type"`
	Delta string `json:"delta"`
}

// responsesFunctionCallArgsDelta carries incremental function call arguments.
type responsesFunctionCallArgsDelta struct {
	Type  string `json:"type"`
	Delta string `json:"delta"`
}

// responsesOutputItemDone is sent when an output item is fully generated.
type responsesOutputItemDone struct {
	Type string              `json:"type"`
	Item responsesOutputItem `json:"item"`
}

// responsesOutputItem represents a completed output item.
type responsesOutputItem struct {
	Type      string `json:"type"` // "message", "function_call"
	ID        string `json:"id"`
	Name      string `json:"name,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// responsesCompletedEvent is sent when the response is fully generated.
type responsesCompletedEvent struct {
	Type     string `json:"type"`
	Response struct {
		ID    string         `json:"id"`
		Usage ResponsesUsage `json:"usage"`
	} `json:"response"`
}

// ResponsesUsage contains token usage statistics.
type ResponsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// responsesErrorEvent is sent when an error occurs.
type responsesErrorEvent struct {
	Type   string `json:"type"`
	Status int    `json:"status"`
	Error  struct {
		Type    string `json:"type"`
		Code    string `json:"code"`
		Message string `json:"message"`
		Param   string `json:"param,omitempty"`
	} `json:"error"`
}

// ResponsesResult is the aggregated result of a Responses API call.
type ResponsesResult struct {
	ResponseID string
	Content    string
	ToolCalls  []ResponsesToolCall
	Usage      ResponsesUsage
}

// ResponsesToolCall represents a tool call requested by the model.
type ResponsesToolCall struct {
	ID        string // unique call ID
	Name      string // function name
	Arguments string // JSON arguments
}
