// Command tinygo-check verifies that core langchaingo packages compile with tinygo.
// It is not meant to be run directly.
package main

import (
	_ "github.com/tmc/langchaingo/callbacks"
	_ "github.com/tmc/langchaingo/chains"
	_ "github.com/tmc/langchaingo/httputil"
	_ "github.com/tmc/langchaingo/llms"
	_ "github.com/tmc/langchaingo/llms/openai"
	_ "github.com/tmc/langchaingo/memory"
	_ "github.com/tmc/langchaingo/prompts"
	_ "github.com/tmc/langchaingo/schema"
	_ "github.com/tmc/langchaingo/textsplitter"
	_ "github.com/tmc/langchaingo/tools"
)

func main() {}
