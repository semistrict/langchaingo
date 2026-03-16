package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/tmc/langchaingo/examples/coding-agent/sandbox"
	"github.com/tmc/langchaingo/llms"
)

const (
	compactionThreshold = 40
	compactionDir       = ".agent/conversation_history"
)

type conversationTracker struct {
	messages []trackedMessage
	sb       sandbox.Sandbox
	session  string
}

type trackedMessage struct {
	role    string
	content string
}

func newConversationTracker(sb sandbox.Sandbox) *conversationTracker {
	return &conversationTracker{
		sb:      sb,
		session: fmt.Sprintf("%d", time.Now().UnixNano()),
	}
}

func (ct *conversationTracker) Track(role, content string) {
	ct.messages = append(ct.messages, trackedMessage{role: role, content: content})
}

func (ct *conversationTracker) NeedsCompaction() bool {
	return len(ct.messages) >= compactionThreshold
}

// Compact writes the full conversation history to a file via the sandbox
// and returns replacement messages that instruct the agent to read the file
// and resume automatically — no user interaction needed.
func (ct *conversationTracker) Compact(ctx context.Context) ([]llms.MessageContent, error) {
	var md strings.Builder
	md.WriteString("# Conversation History\n\n")
	for _, msg := range ct.messages {
		fmt.Fprintf(&md, "## %s\n\n%s\n\n", msg.role, msg.content)
	}

	path := fmt.Sprintf("%s/%s.md", compactionDir, ct.session)
	if err := ct.sb.WriteFile(ctx, path, md.String()); err != nil {
		return nil, fmt.Errorf("write history file: %w", err)
	}

	ct.messages = nil

	// Instruct the agent to read the history and resume autonomously.
	// This follows the ccmd/deepagentsjs pattern: the agent gets the file
	// path, reads it with read_file, and picks up where it left off.
	resumeText := fmt.Sprintf(
		"This session is being continued from a previous conversation that ran out of context. "+
			"The full conversation history has been saved to %s.\n\n"+
			"Read that file now with read_file to understand what was being worked on. "+
			"Then continue the conversation from where it left off without asking the user any further questions. "+
			"Resume directly — do not acknowledge the summary, do not recap what was happening, "+
			"do not preface with \"I'll continue\" or similar. "+
			"Pick up the last task as if the break never happened.",
		path,
	)

	return []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{llms.TextContent{Text: resumeText}},
		},
	}, nil
}
