package openai

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/llms"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"nhooyr.io/websocket"
	"nhooyr.io/websocket/wsjson"
)

func TestResponsesOtel_SendCreatesSpans(t *testing.T) {
	// Set up in-memory OTel exporter.
	exporter := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	defer tp.Shutdown(t.Context())

	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(prev)

	// Create a test WebSocket server.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close(websocket.StatusNormalClosure, "done")
		ctx := r.Context()

		// Read request, send response.
		var raw json.RawMessage
		if err := wsjson.Read(ctx, conn, &raw); err != nil {
			return
		}
		events := []any{
			map[string]any{
				"type":     "response.created",
				"response": map[string]any{"id": "resp_otel1"},
			},
			map[string]any{
				"type":  "response.output_text.delta",
				"delta": "Hello from OTel test!",
			},
			map[string]any{
				"type": "response.completed",
				"response": map[string]any{
					"id":    "resp_otel1",
					"usage": map[string]any{"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
				},
			},
		}
		for _, e := range events {
			if err := wsjson.Write(ctx, conn, e); err != nil {
				return
			}
		}
	}))
	defer srv.Close()

	llm, err := New(
		WithToken("test-token"),
		WithModel("gpt-5.4"),
		WithBaseURL(srv.URL),
	)
	require.NoError(t, err)

	ctx := t.Context()
	session, err := llm.OpenResponsesSession(ctx)
	require.NoError(t, err)

	resp, err := session.Send(ctx,
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "Hello"),
		},
	)
	require.NoError(t, err)
	assert.Equal(t, "Hello from OTel test!", resp.Choices[0].Content)

	session.Close()

	// Force flush and check spans.
	tp.ForceFlush(ctx)
	spans := exporter.GetSpans()

	// Should have 2 spans: session + send.
	require.GreaterOrEqual(t, len(spans), 2)

	// Find the spans by name.
	var sessionSpan, sendSpan tracetest.SpanStub
	for _, s := range spans {
		switch s.Name {
		case "responses.session":
			sessionSpan = s
		case "responses.send":
			sendSpan = s
		}
	}

	require.NotEmpty(t, sessionSpan.Name, "missing responses.session span")
	require.NotEmpty(t, sendSpan.Name, "missing responses.send span")

	// Verify session span attributes.
	assertHasAttr(t, sessionSpan.Attributes, "gen_ai.system", "openai")
	assertHasAttr(t, sessionSpan.Attributes, "langsmith.span.kind", "chain")

	// Verify send span attributes.
	assertHasAttr(t, sendSpan.Attributes, "gen_ai.system", "openai")
	assertHasAttr(t, sendSpan.Attributes, "gen_ai.operation.name", "chat")
	assertHasAttr(t, sendSpan.Attributes, "gen_ai.request.model", "gpt-5.4")
	assertHasAttr(t, sendSpan.Attributes, "langsmith.span.kind", "llm")
	assertHasAttr(t, sendSpan.Attributes, "gen_ai.completion.0.content", "Hello from OTel test!")
	assertHasAttr(t, sendSpan.Attributes, "gen_ai.response.id", "resp_otel1")

	// Verify token usage.
	assertHasIntAttr(t, sendSpan.Attributes, "gen_ai.usage.input_tokens", 10)
	assertHasIntAttr(t, sendSpan.Attributes, "gen_ai.usage.output_tokens", 5)

	// Verify send span is a child of session span.
	assert.Equal(t, sessionSpan.SpanContext.TraceID(), sendSpan.SpanContext.TraceID(),
		"send span should be in the same trace as session span")
}

func TestResponsesOtel_ToolCallSpans(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	defer tp.Shutdown(t.Context())

	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(prev)

	turn := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close(websocket.StatusNormalClosure, "done")
		ctx := r.Context()

		for {
			var raw json.RawMessage
			if err := wsjson.Read(ctx, conn, &raw); err != nil {
				return
			}

			var events []any
			if turn == 0 {
				events = []any{
					map[string]any{"type": "response.created", "response": map[string]any{"id": "resp_t1"}},
					map[string]any{"type": "response.output_item.added"},
					map[string]any{"type": "response.function_call_arguments.delta", "delta": `{"q":"test"}`},
					map[string]any{"type": "response.output_item.done", "item": map[string]any{
						"type": "function_call", "id": "fc_1", "call_id": "call_1",
						"name": "search", "arguments": `{"q":"test"}`,
					}},
					map[string]any{"type": "response.completed", "response": map[string]any{
						"id": "resp_t1", "usage": map[string]any{"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
					}},
				}
			} else {
				events = []any{
					map[string]any{"type": "response.created", "response": map[string]any{"id": "resp_t2"}},
					map[string]any{"type": "response.output_text.delta", "delta": "Done"},
					map[string]any{"type": "response.completed", "response": map[string]any{
						"id": "resp_t2", "usage": map[string]any{"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
					}},
				}
			}
			turn++
			for _, e := range events {
				if err := wsjson.Write(ctx, conn, e); err != nil {
					return
				}
			}
		}
	}))
	defer srv.Close()

	llm, err := New(WithToken("test"), WithModel("gpt-5.4"), WithBaseURL(srv.URL))
	require.NoError(t, err)

	ctx := t.Context()
	session, err := llm.OpenResponsesSession(ctx)
	require.NoError(t, err)

	// Turn 1: get tool call.
	resp1, err := session.Send(ctx,
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "search")},
	)
	require.NoError(t, err)
	require.Len(t, resp1.Choices[0].ToolCalls, 1)

	// Turn 2: send tool result.
	_, err = session.SendToolResults(ctx,
		[]ToolResult{{CallID: "call_1", Output: "result"}},
	)
	require.NoError(t, err)

	session.Close()
	tp.ForceFlush(ctx)
	spans := exporter.GetSpans()

	// Should have: session, send, send_tool_results = 3 spans.
	require.GreaterOrEqual(t, len(spans), 3)

	var toolResultSpan tracetest.SpanStub
	for _, s := range spans {
		if s.Name == "responses.send_tool_results" {
			toolResultSpan = s
		}
	}

	require.NotEmpty(t, toolResultSpan.Name, "missing responses.send_tool_results span")
	assertHasAttr(t, toolResultSpan.Attributes, "gen_ai.tool_result.0.call_id", "call_1")
	assertHasAttr(t, toolResultSpan.Attributes, "gen_ai.tool_result.0.output", "result")
}

func TestResponsesOtel_ErrorRecordsSpan(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	defer tp.Shutdown(t.Context())

	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(prev)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close(websocket.StatusNormalClosure, "done")
		ctx := r.Context()

		var raw json.RawMessage
		if err := wsjson.Read(ctx, conn, &raw); err != nil {
			return
		}
		wsjson.Write(ctx, conn, map[string]any{
			"type": "error", "status": 400,
			"error": map[string]any{"type": "invalid_request_error", "code": "bad_request", "message": "bad"},
		})
	}))
	defer srv.Close()

	llm, err := New(WithToken("test"), WithModel("gpt-5.4"), WithBaseURL(srv.URL))
	require.NoError(t, err)

	ctx := t.Context()
	session, err := llm.OpenResponsesSession(ctx)
	require.NoError(t, err)

	_, err = session.Send(ctx,
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "fail")},
	)
	require.Error(t, err)

	session.Close()
	tp.ForceFlush(ctx)
	spans := exporter.GetSpans()

	var sendSpan tracetest.SpanStub
	for _, s := range spans {
		if s.Name == "responses.send" {
			sendSpan = s
		}
	}
	require.NotEmpty(t, sendSpan.Name)
	assert.Equal(t, "Error", sendSpan.Status.Code.String())
}

// assertHasAttr checks that the attribute list contains the given key-value pair.
func assertHasAttr(t *testing.T, attrs []attribute.KeyValue, key, value string) {
	t.Helper()
	for _, a := range attrs {
		if string(a.Key) == key {
			assert.Equal(t, value, a.Value.AsString(), "attribute %s", key)
			return
		}
	}
	t.Errorf("missing attribute %s", key)
}

// assertHasIntAttr checks that the attribute list contains the given int key-value pair.
func assertHasIntAttr(t *testing.T, attrs []attribute.KeyValue, key string, value int) {
	t.Helper()
	for _, a := range attrs {
		if string(a.Key) == key {
			assert.Equal(t, int64(value), a.Value.AsInt64(), "attribute %s", key)
			return
		}
	}
	t.Errorf("missing attribute %s", key)
}
