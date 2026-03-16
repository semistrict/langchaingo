package openai

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/llms"
)

// TestResponsesHTTPOpenRouter_MultiTurnMemory tests that the model can recall
// information from earlier turns when full history is replayed — the same
// pattern the coding agent uses.
func TestResponsesHTTPOpenRouter_MultiTurnMemory(t *testing.T) {
	llm := newOpenRouterResponsesTestClient(t, WithModel("openai/gpt-4.1-mini"))
	ctx := t.Context()

	session, err := llm.OpenResponsesHTTPSession(ctx)
	require.NoError(t, err)
	defer session.Close()

	sys := llms.MessageContent{
		Role:  llms.ChatMessageTypeSystem,
		Parts: []llms.ContentPart{llms.TextContent{Text: "You are a helpful assistant. Be concise. Answer in one sentence."}},
	}

	// Accumulate messages like Conversation does.
	messages := []llms.MessageContent{sys}

	// Turn 1: tell the model a fact.
	messages = append(messages, llms.MessageContent{
		Role:  llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{llms.TextContent{Text: "My favorite color is purple. Remember that."}},
	})
	resp1, err := session.GenerateContent(ctx, messages)
	require.NoError(t, err)
	require.Len(t, resp1.Choices, 1)
	t.Logf("Turn 1: %s", resp1.Choices[0].Content)

	// Add assistant response to history.
	messages = append(messages, llms.MessageContent{
		Role:  llms.ChatMessageTypeAI,
		Parts: []llms.ContentPart{llms.TextContent{Text: resp1.Choices[0].Content}},
	})

	// Turn 2: ask something unrelated.
	messages = append(messages, llms.MessageContent{
		Role:  llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{llms.TextContent{Text: "What is 2+2?"}},
	})
	resp2, err := session.GenerateContent(ctx, messages)
	require.NoError(t, err)
	require.Len(t, resp2.Choices, 1)
	t.Logf("Turn 2: %s", resp2.Choices[0].Content)

	messages = append(messages, llms.MessageContent{
		Role:  llms.ChatMessageTypeAI,
		Parts: []llms.ContentPart{llms.TextContent{Text: resp2.Choices[0].Content}},
	})

	// Turn 3: ask about the fact from turn 1.
	messages = append(messages, llms.MessageContent{
		Role:  llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{llms.TextContent{Text: "What is my favorite color?"}},
	})
	resp3, err := session.GenerateContent(ctx, messages)
	require.NoError(t, err)
	require.Len(t, resp3.Choices, 1)
	t.Logf("Turn 3 (%d messages sent): %s", len(messages), resp3.Choices[0].Content)

	assert.Contains(t, resp3.Choices[0].Content, "purple")

	// Verify GenerationInfo contains token usage including cached_tokens.
	info := resp3.Choices[0].GenerationInfo
	assert.NotNil(t, info)
	inputTok, _ := info["input_tokens"].(int)
	outputTok, _ := info["output_tokens"].(int)
	cachedTok, _ := info["cached_tokens"].(int)
	t.Logf("Turn 3 usage: input=%d output=%d cached=%d", inputTok, outputTok, cachedTok)
	assert.Greater(t, inputTok, 0, "input_tokens should be > 0")
	assert.Greater(t, outputTok, 0, "output_tokens should be > 0")
}
