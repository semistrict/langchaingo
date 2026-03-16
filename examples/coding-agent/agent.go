package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tmc/langchaingo/examples/coding-agent/sandbox"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

// AgentEvent is emitted by the agent loop to notify the TUI of progress.
type AgentEvent struct {
	Type string // "text", "tool_call", "tool_result", "error", "done"

	Text       string // "text" events
	ToolName   string // "tool_call" / "tool_result" events
	ToolArgs   string // "tool_call" events
	ToolOutput string // "tool_result" events
	Error      error  // "error" events

	InputTokens  int
	OutputTokens int
	CachedTokens int
}

// Conversation holds the persistent message history across turns.
type Conversation struct {
	messages []llms.MessageContent
	tracker  *conversationTracker
}

// NewConversation creates a new conversation with the given system prompt.
func NewConversation(sysPrompt string, sb sandbox.Sandbox) *Conversation {
	return &Conversation{
		messages: []llms.MessageContent{
			{Role: llms.ChatMessageTypeSystem, Parts: []llms.ContentPart{llms.TextContent{Text: sysPrompt}}},
		},
		tracker: newConversationTracker(sb),
	}
}

// RunTurn sends a user message and runs the agent loop (tool calls until text response).
func (c *Conversation) RunTurn(
	ctx context.Context,
	session *openai.ResponsesSession,
	sb sandbox.Sandbox,
	userPrompt string,
	onEvent func(AgentEvent),
) error {
	// Append user message.
	c.messages = append(c.messages, llms.MessageContent{
		Role:  llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{llms.TextContent{Text: userPrompt}},
	})
	c.tracker.Track("user", userPrompt)

	for {
		resp, err := session.GenerateContent(ctx, c.messages,
			llms.WithTools(codexTools),
			llms.WithStreamingFunc(func(_ context.Context, chunk []byte) error {
				onEvent(AgentEvent{Type: "text", Text: string(chunk)})
				return nil
			}),
		)
		if err != nil {
			onEvent(AgentEvent{Type: "error", Error: err})
			return err
		}

		if len(resp.Choices) == 0 {
			break
		}
		choice := resp.Choices[0]

		if len(choice.ToolCalls) == 0 {
			// Append assistant text to history.
			if choice.Content != "" {
				c.messages = append(c.messages, llms.MessageContent{
					Role:  llms.ChatMessageTypeAI,
					Parts: []llms.ContentPart{llms.TextContent{Text: choice.Content}},
				})
				c.tracker.Track("assistant", choice.Content)
			}
			inputTokens, _ := choice.GenerationInfo["input_tokens"].(int)
			outputTokens, _ := choice.GenerationInfo["output_tokens"].(int)
			cachedTokens, _ := choice.GenerationInfo["cached_tokens"].(int)
			onEvent(AgentEvent{Type: "done", InputTokens: inputTokens, OutputTokens: outputTokens, CachedTokens: cachedTokens})
			return nil
		}

		// Add assistant message with tool calls to history.
		var assistantParts []llms.ContentPart
		if choice.Content != "" {
			assistantParts = append(assistantParts, llms.TextContent{Text: choice.Content})
		}
		for _, tc := range choice.ToolCalls {
			assistantParts = append(assistantParts, tc)
		}
		c.messages = append(c.messages, llms.MessageContent{
			Role:  llms.ChatMessageTypeAI,
			Parts: assistantParts,
		})

		// Execute tool calls, add results to history.
		var toolParts []llms.ContentPart
		for _, tc := range choice.ToolCalls {
			onEvent(AgentEvent{Type: "tool_call", ToolName: tc.FunctionCall.Name, ToolArgs: tc.FunctionCall.Arguments})

			output, execErr := executeTool(ctx, sb, tc.FunctionCall.Name, tc.FunctionCall.Arguments)
			if execErr != nil {
				output = fmt.Sprintf("Error: %v", execErr)
			}
			onEvent(AgentEvent{Type: "tool_result", ToolName: tc.FunctionCall.Name, ToolOutput: truncate(output, 4000)})

			toolParts = append(toolParts, llms.ToolCallResponse{
				ToolCallID: tc.ID,
				Name:       tc.FunctionCall.Name,
				Content:    output,
			})
			c.tracker.Track("tool:"+tc.FunctionCall.Name, output)
		}
		c.messages = append(c.messages, llms.MessageContent{
			Role:  llms.ChatMessageTypeTool,
			Parts: toolParts,
		})
	}

	onEvent(AgentEvent{Type: "done"})
	return nil
}

// CompactAndResume writes history to a file, replaces messages with a resume
// instruction, then runs a turn with that instruction so the agent reads the
// file and picks up where it left off. Returns the history file path.
func (c *Conversation) CompactAndResume(
	ctx context.Context,
	session *openai.ResponsesSession,
	sb sandbox.Sandbox,
	onEvent func(AgentEvent),
) (string, error) {
	resumeMsgs, err := c.tracker.Compact(ctx)
	if err != nil {
		return "", err
	}

	// Keep system prompt, replace everything else.
	var sysMsg llms.MessageContent
	if len(c.messages) > 0 && c.messages[0].Role == llms.ChatMessageTypeSystem {
		sysMsg = c.messages[0]
	}
	c.messages = append([]llms.MessageContent{sysMsg}, resumeMsgs...)

	path := fmt.Sprintf("%s/%s.md", compactionDir, c.tracker.session)

	// Extract the resume text and run it as a normal turn.
	resumeText := ""
	for _, part := range resumeMsgs[0].Parts {
		if tc, ok := part.(llms.TextContent); ok {
			resumeText = tc.Text
			break
		}
	}
	// Remove the message we just added — RunTurn will re-add it.
	c.messages = c.messages[:len(c.messages)-1]

	err = c.RunTurn(ctx, session, sb, resumeText, onEvent)
	return path, err
}

// AddSystemContext adds a human message to the conversation that the agent
// will see on its next turn. Used for shell mode results.
func (c *Conversation) AddSystemContext(text string) {
	c.messages = append(c.messages, llms.MessageContent{
		Role:  llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{llms.TextContent{Text: text}},
	})
	c.tracker.Track("user (shell)", text)
}

// MessageCount returns the number of messages in the conversation.
func (c *Conversation) MessageCount() int {
	return len(c.messages)
}

// executeTool dispatches a tool call to the sandbox.
func executeTool(ctx context.Context, sb sandbox.Sandbox, name, argsJSON string) (string, error) {
	switch name {
	case "shell":
		var args struct {
			Command   []string `json:"command"`
			Workdir   string   `json:"workdir"`
			TimeoutMs int      `json:"timeout_ms"`
		}
		if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
			return "", fmt.Errorf("parse shell args: %w", err)
		}
		result, err := sb.Shell(ctx, args.Command, args.Workdir, args.TimeoutMs)
		if err != nil {
			return "", err
		}
		return formatShellResult(result), nil

	case "write_stdin":
		var args struct {
			SessionID   int    `json:"session_id"`
			Chars       string `json:"chars"`
			YieldTimeMs int    `json:"yield_time_ms"`
		}
		if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
			return "", fmt.Errorf("parse write_stdin args: %w", err)
		}
		result, err := sb.WriteStdin(ctx, args.SessionID, args.Chars, args.YieldTimeMs)
		if err != nil {
			return "", err
		}
		return formatShellResult(result), nil

	case "read_file":
		var args struct {
			FilePath  string `json:"file_path"`
			StartLine int    `json:"start_line"`
			EndLine   int    `json:"end_line"`
		}
		if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
			return "", fmt.Errorf("parse read_file args: %w", err)
		}
		return sb.ReadFile(ctx, args.FilePath, args.StartLine, args.EndLine)

	case "list_dir":
		var args struct {
			DirPath string `json:"dir_path"`
			Depth   int    `json:"depth"`
		}
		if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
			return "", fmt.Errorf("parse list_dir args: %w", err)
		}
		entries, err := sb.ListDir(ctx, args.DirPath, args.Depth)
		if err != nil {
			return "", err
		}
		var buf strings.Builder
		for _, e := range entries {
			fmt.Fprintf(&buf, "%s\t%s\n", e.Type, e.Name)
		}
		return buf.String(), nil

	case "grep_files":
		var args struct {
			Pattern string `json:"pattern"`
			Path    string `json:"path"`
			Include string `json:"include"`
		}
		if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
			return "", fmt.Errorf("parse grep_files args: %w", err)
		}
		matches, err := sb.GrepFiles(ctx, args.Pattern, args.Path, args.Include)
		if err != nil {
			return "", err
		}
		var buf strings.Builder
		for _, m := range matches {
			fmt.Fprintf(&buf, "%s:%d:%s\n", m.File, m.Line, m.Text)
		}
		return buf.String(), nil

	case "apply_patch":
		var args struct {
			Patch string `json:"patch"`
		}
		if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
			return "", fmt.Errorf("parse apply_patch args: %w", err)
		}
		if err := applyPatch(ctx, sb, args.Patch); err != nil {
			return "", err
		}
		return "Patch applied successfully.", nil

	default:
		return "", fmt.Errorf("unknown tool: %s", name)
	}
}

func formatShellResult(r sandbox.ShellResult) string {
	var buf strings.Builder
	buf.WriteString(r.Output)
	if r.SessionID > 0 {
		fmt.Fprintf(&buf, "\n[session_id: %d, still running]", r.SessionID)
	} else {
		fmt.Fprintf(&buf, "\n[exit code: %d]", r.ExitCode)
	}
	return buf.String()
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "\n... (truncated)"
}
