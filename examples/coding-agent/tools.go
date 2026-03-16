package main

import (
	"encoding/json"

	"github.com/tmc/langchaingo/llms"
)

// codexTools defines the tool definitions matching Codex tool names.
// These are sent to the model so it can invoke them via function calling.
var codexTools = []llms.Tool{
	{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:        "shell",
			Description: "Execute a shell command. Returns stdout and stderr. If the command doesn't finish within timeout_ms, returns a session_id that can be used with write_stdin to send more input or poll for output.",
			Parameters: jsonSchema(map[string]any{
				"type": "object",
				"properties": map[string]any{
					"command": map[string]any{
						"type":        "array",
						"items":       map[string]any{"type": "string"},
						"description": "The command and arguments to execute.",
					},
					"workdir": map[string]any{
						"type":        "string",
						"description": "Working directory for the command. Defaults to the current directory.",
					},
					"timeout_ms": map[string]any{
						"type":        "integer",
						"description": "Maximum time in milliseconds to wait for the command to finish. Default 30000.",
					},
				},
				"required":             []string{"command"},
				"additionalProperties": false,
			}),
			Strict: true,
		},
	},
	{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:        "write_stdin",
			Description: "Send input to a running shell session (started by a shell command that timed out). Pass empty chars to just poll for new output.",
			Parameters: jsonSchema(map[string]any{
				"type": "object",
				"properties": map[string]any{
					"session_id": map[string]any{
						"type":        "integer",
						"description": "The session ID returned by the shell tool.",
					},
					"chars": map[string]any{
						"type":        "string",
						"description": "Characters to send to the process stdin.",
					},
					"yield_time_ms": map[string]any{
						"type":        "integer",
						"description": "Time in milliseconds to wait for output after sending input. Default 500.",
					},
				},
				"required":             []string{"session_id"},
				"additionalProperties": false,
			}),
			Strict: true,
		},
	},
	{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:        "read_file",
			Description: "Read the contents of a file. Optionally specify a line range (1-indexed).",
			Parameters: jsonSchema(map[string]any{
				"type": "object",
				"properties": map[string]any{
					"file_path": map[string]any{
						"type":        "string",
						"description": "Path to the file to read.",
					},
					"start_line": map[string]any{
						"type":        "integer",
						"description": "First line to read (1-indexed). 0 or omitted means start of file.",
					},
					"end_line": map[string]any{
						"type":        "integer",
						"description": "Last line to read (1-indexed, inclusive). 0 or omitted means end of file.",
					},
				},
				"required":             []string{"file_path"},
				"additionalProperties": false,
			}),
			Strict: true,
		},
	},
	{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:        "list_dir",
			Description: "List the contents of a directory.",
			Parameters: jsonSchema(map[string]any{
				"type": "object",
				"properties": map[string]any{
					"dir_path": map[string]any{
						"type":        "string",
						"description": "Path to the directory to list.",
					},
					"depth": map[string]any{
						"type":        "integer",
						"description": "How many levels deep to list. Default 1.",
					},
				},
				"required":             []string{"dir_path"},
				"additionalProperties": false,
			}),
			Strict: true,
		},
	},
	{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:        "grep_files",
			Description: "Search for a pattern in files using grep. Returns matching lines with file paths and line numbers.",
			Parameters: jsonSchema(map[string]any{
				"type": "object",
				"properties": map[string]any{
					"pattern": map[string]any{
						"type":        "string",
						"description": "The regex pattern to search for.",
					},
					"path": map[string]any{
						"type":        "string",
						"description": "The file or directory to search in.",
					},
					"include": map[string]any{
						"type":        "string",
						"description": "File glob pattern to filter which files to search (e.g. '*.go').",
					},
				},
				"required":             []string{"pattern", "path"},
				"additionalProperties": false,
			}),
			Strict: true,
		},
	},
	{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:        "apply_patch",
			Description: "Apply a unified diff patch to files. Supports creating, modifying, and deleting files.",
			Parameters: jsonSchema(map[string]any{
				"type": "object",
				"properties": map[string]any{
					"patch": map[string]any{
						"type":        "string",
						"description": "The patch in unified diff format.",
					},
				},
				"required":             []string{"patch"},
				"additionalProperties": false,
			}),
			Strict: true,
		},
	},
}

func jsonSchema(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}
