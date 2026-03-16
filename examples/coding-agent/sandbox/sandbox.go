// Package sandbox defines the interface for executing coding agent tool
// operations (shell commands, file I/O, directory listing, text search).
// Implementations can target local filesystem/bash or remote sandboxes.
package sandbox

import "context"

// Sandbox handles I/O primitives for the coding agent.
type Sandbox interface {
	// Shell executes a command and returns stdout+stderr.
	// If the command is still running when timeoutMs expires, returns a
	// ShellResult with a SessionID for use with WriteStdin.
	Shell(ctx context.Context, command []string, workdir string, timeoutMs int) (ShellResult, error)

	// WriteStdin sends input to a running shell session and returns new output.
	// Pass empty chars to just poll for output.
	WriteStdin(ctx context.Context, sessionID int, chars string, yieldTimeMs int) (ShellResult, error)

	// ReadFile reads a file, optionally a line range (1-indexed, 0 means no limit).
	ReadFile(ctx context.Context, path string, startLine, endLine int) (string, error)

	// WriteFile writes content to a file, creating it if needed.
	WriteFile(ctx context.Context, path string, content string) error

	// ListDir lists directory entries up to the given depth (0 = immediate children).
	ListDir(ctx context.Context, path string, depth int) ([]DirEntry, error)

	// GrepFiles searches for a pattern across files.
	GrepFiles(ctx context.Context, pattern string, path string, include string) ([]GrepMatch, error)
}

// ShellResult is the result of a shell command execution.
type ShellResult struct {
	Output    string
	ExitCode  int // -1 if still running
	SessionID int // non-zero if process is still running
}

// DirEntry is a single directory listing entry.
type DirEntry struct {
	Name string
	Type string // "file", "dir", "symlink"
}

// GrepMatch is a single search match.
type GrepMatch struct {
	File string
	Line int
	Text string
}
