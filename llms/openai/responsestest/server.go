// Package responsestest provides a mock HTTP server for the OpenAI Responses API.
// It echoes back the user's input text as a streaming SSE response with a
// configurable delay between chunks.
package responsestest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"time"
)

// chunkSize is the target number of characters per streaming delta.
const chunkSize = 16

// Server is a mock HTTP server for the OpenAI Responses API.
// It echoes back the last user message as a streaming SSE response,
// splitting the text into chunks of roughly 16 characters.
type Server struct {
	*httptest.Server
	chunkDelay atomic.Int64 // nanoseconds between SSE chunks
	reqCount   atomic.Int64
}

// NewServer creates a new mock Responses API server backed by httptest.
// The chunkDelay controls the pause between SSE delta events.
func NewServer(chunkDelay time.Duration) *Server {
	s := &Server{}
	s.chunkDelay.Store(int64(chunkDelay))
	s.Server = httptest.NewServer(http.HandlerFunc(s.handle))
	return s
}

// Handler returns an http.Handler that can be used with a custom listener
// (e.g. http.ListenAndServe). Use this instead of NewServer when you need
// control over the listen address.
func Handler(chunkDelay time.Duration) http.Handler {
	s := &Server{}
	s.chunkDelay.Store(int64(chunkDelay))
	return http.HandlerFunc(s.handle)
}

// SetChunkDelay changes the delay between chunks at runtime.
func (s *Server) SetChunkDelay(d time.Duration) {
	s.chunkDelay.Store(int64(d))
}

// RequestCount returns the number of requests handled.
func (s *Server) RequestCount() int64 {
	return s.reqCount.Load()
}

func (s *Server) handle(w http.ResponseWriter, r *http.Request) {
	s.reqCount.Add(1)

	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Input []struct {
			Type    string `json:"type"`
			Role    string `json:"role"`
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"input"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Find the last user message to echo.
	var userText string
	for _, item := range req.Input {
		if item.Role == "user" && item.Type == "message" {
			for _, c := range item.Content {
				userText = c.Text
			}
		}
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	respID := fmt.Sprintf("resp_echo_%d", s.reqCount.Load())

	// response.created
	writeSSE(w, flusher, map[string]any{
		"type":     "response.created",
		"response": map[string]any{"id": respID},
	})

	// Check for "calltool NAME {args}" pattern.
	if name, args, ok := parseCallTool(userText); ok {
		s.writeToolCall(w, flusher, respID, name, args)
	} else {
		s.writeTextEcho(w, flusher, respID, userText)
	}
}

// parseCallTool checks if text matches "calltool NAME {json args}".
// Returns the tool name, JSON args string, and whether the pattern matched.
func parseCallTool(text string) (name, args string, ok bool) {
	text = strings.TrimSpace(text)
	if !strings.HasPrefix(text, "calltool ") {
		return "", "", false
	}
	rest := strings.TrimPrefix(text, "calltool ")
	// Split into NAME and the rest ({args}).
	idx := strings.IndexByte(rest, ' ')
	if idx < 0 {
		// "calltool NAME" with no args.
		return rest, "{}", true
	}
	name = rest[:idx]
	args = strings.TrimSpace(rest[idx+1:])
	if args == "" {
		args = "{}"
	}
	return name, args, true
}

// writeTextEcho streams the user text back as chunked deltas.
func (s *Server) writeTextEcho(w http.ResponseWriter, flusher http.Flusher, respID, userText string) {
	// response.output_item.added
	writeSSE(w, flusher, map[string]any{
		"type": "response.output_item.added",
		"item": map[string]any{"type": "message", "id": "item_0"},
	})

	// response.content_part.added
	writeSSE(w, flusher, map[string]any{
		"type": "response.content_part.added",
		"part": map[string]any{"type": "output_text", "text": ""},
	})

	delay := time.Duration(s.chunkDelay.Load())
	for _, chunk := range splitChunks(userText, chunkSize) {
		if delay > 0 {
			time.Sleep(delay)
		}
		writeSSE(w, flusher, map[string]any{
			"type":  "response.output_text.delta",
			"delta": chunk,
		})
	}

	// response.output_text.done
	writeSSE(w, flusher, map[string]any{
		"type": "response.output_text.done",
		"text": userText,
	})

	// response.content_part.done
	writeSSE(w, flusher, map[string]any{
		"type": "response.content_part.done",
		"part": map[string]any{"type": "output_text", "text": userText},
	})

	// response.output_item.done
	writeSSE(w, flusher, map[string]any{
		"type": "response.output_item.done",
		"item": map[string]any{"type": "message", "id": "item_0"},
	})

	// response.completed
	writeSSE(w, flusher, map[string]any{
		"type": "response.completed",
		"response": map[string]any{
			"id": respID,
			"usage": map[string]any{
				"input_tokens":  len([]rune(userText)),
				"output_tokens": len([]rune(userText)),
				"total_tokens":  len([]rune(userText)) * 2,
			},
		},
	})

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// writeToolCall streams a function_call response for the given tool name and args.
func (s *Server) writeToolCall(w http.ResponseWriter, flusher http.Flusher, respID, name, args string) {
	callID := fmt.Sprintf("call_%s_%s", name, respID)

	// response.output_item.added
	writeSSE(w, flusher, map[string]any{
		"type": "response.output_item.added",
		"item": map[string]any{"type": "function_call", "id": "fc_0"},
	})

	// Stream the arguments as chunked deltas.
	delay := time.Duration(s.chunkDelay.Load())
	for _, chunk := range splitChunks(args, chunkSize) {
		if delay > 0 {
			time.Sleep(delay)
		}
		writeSSE(w, flusher, map[string]any{
			"type":  "response.function_call_arguments.delta",
			"delta": chunk,
		})
	}

	// response.output_item.done with the complete function call.
	writeSSE(w, flusher, map[string]any{
		"type": "response.output_item.done",
		"item": map[string]any{
			"type":      "function_call",
			"id":        "fc_0",
			"call_id":   callID,
			"name":      name,
			"arguments": args,
		},
	})

	// response.completed
	writeSSE(w, flusher, map[string]any{
		"type": "response.completed",
		"response": map[string]any{
			"id": respID,
			"usage": map[string]any{
				"input_tokens":  len([]rune(args)),
				"output_tokens": len([]rune(args)),
				"total_tokens":  len([]rune(args)) * 2,
			},
		},
	})

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// splitChunks splits s into pieces of up to size runes.
// Short strings (≤ size) are returned as a single chunk.
func splitChunks(s string, size int) []string {
	runes := []rune(s)
	if len(runes) == 0 {
		return nil
	}
	var chunks []string
	for len(runes) > 0 {
		end := size
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[:end]))
		runes = runes[end:]
	}
	return chunks
}

func writeSSE(w http.ResponseWriter, f http.Flusher, event any) {
	data, _ := json.Marshal(event)
	fmt.Fprintf(w, "data: %s\n\n", data)
	f.Flush()
}
