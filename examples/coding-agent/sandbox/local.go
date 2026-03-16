package sandbox

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	pty "github.com/creack/pty/v2"
	"github.com/vito/midterm"
)

// Local implements Sandbox using local filesystem and bash.
type Local struct {
	mu         sync.Mutex
	sessions   map[int]*session
	nextSessID int
}

type session struct {
	cmd  *exec.Cmd
	ptyF *os.File           // PTY file (read/write)
	term *midterm.Terminal   // virtual terminal tracking screen state
	mu   sync.Mutex         // protects term reads
	done chan struct{}
}

// screenText returns the current visible text from the midterm terminal,
// stripping trailing empty lines.
func (s *session) screenText() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var buf strings.Builder
	h := s.term.UsedHeight()
	for row := range h {
		if row < len(s.term.Content) {
			buf.WriteString(string(s.term.Content[row]))
		}
		buf.WriteByte('\n')
	}
	return strings.TrimRight(buf.String(), "\n")
}

// snapshotAndReset returns the current screen text and clears the terminal
// so subsequent reads only return new output.
func (s *session) snapshotAndReset() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var buf strings.Builder
	h := s.term.UsedHeight()
	for row := range h {
		if row < len(s.term.Content) {
			buf.WriteString(string(s.term.Content[row]))
		}
		buf.WriteByte('\n')
	}
	s.term.Reset()
	return strings.TrimRight(buf.String(), "\n")
}

// NewLocal creates a new local sandbox.
func NewLocal() *Local {
	return &Local{
		sessions:   make(map[int]*session),
		nextSessID: 1,
	}
}

func (l *Local) Shell(ctx context.Context, command []string, workdir string, timeoutMs int) (ShellResult, error) {
	if len(command) == 0 {
		return ShellResult{}, fmt.Errorf("empty command")
	}

	timeout := time.Duration(timeoutMs) * time.Millisecond
	if timeout <= 0 {
		timeout = 30 * time.Second
	}

	cmdCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Run the command directly with pipes (not PTY) for simple commands.
	// PTY is only used for persistent interactive sessions.
	cmd := exec.CommandContext(cmdCtx, command[0], command[1:]...)
	if workdir != "" {
		cmd.Dir = workdir
	}

	var outBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &outBuf

	err := cmd.Run()

	if cmdCtx.Err() != nil {
		// Timeout — start a persistent PTY session.
		cmdStr := strings.Join(command, " ")
		return l.startPersistentSession(ctx, cmdStr, workdir)
	}

	exitCode := 0
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else {
			exitCode = 1
		}
	}
	return ShellResult{Output: outBuf.String(), ExitCode: exitCode}, nil
}

func screenText(term *midterm.Terminal) string {
	var buf strings.Builder
	h := term.UsedHeight()
	for row := range h {
		if row < len(term.Content) {
			buf.WriteString(string(term.Content[row]))
		}
		buf.WriteByte('\n')
	}
	return strings.TrimRight(buf.String(), "\n")
}

func (l *Local) startPersistentSession(ctx context.Context, cmdStr string, workdir string) (ShellResult, error) {
	cmd := exec.CommandContext(ctx, "bash", "-l")
	if workdir != "" {
		cmd.Dir = workdir
	}

	term := midterm.NewTerminal(24, 120)
	ptmx, err := pty.StartWithSize(cmd, &pty.Winsize{Rows: 24, Cols: 120})
	if err != nil {
		return ShellResult{}, fmt.Errorf("start persistent session: %w", err)
	}

	done := make(chan struct{})
	go func() {
		io.Copy(term, ptmx)
		close(done)
	}()

	sess := &session{
		cmd:  cmd,
		ptyF: ptmx,
		term: term,
		done: done,
	}

	// Send the original command.
	fmt.Fprintf(ptmx, "%s\n", cmdStr)

	// Give it a moment to produce some output.
	time.Sleep(500 * time.Millisecond)

	l.mu.Lock()
	id := l.nextSessID
	l.nextSessID++
	l.sessions[id] = sess
	l.mu.Unlock()

	return ShellResult{
		Output:    sess.snapshotAndReset(),
		ExitCode:  -1,
		SessionID: id,
	}, nil
}

func (l *Local) WriteStdin(ctx context.Context, sessionID int, chars string, yieldTimeMs int) (ShellResult, error) {
	l.mu.Lock()
	sess, ok := l.sessions[sessionID]
	l.mu.Unlock()
	if !ok {
		return ShellResult{}, fmt.Errorf("session %d not found", sessionID)
	}

	if chars != "" {
		if _, err := sess.ptyF.WriteString(chars); err != nil {
			return ShellResult{}, fmt.Errorf("write stdin: %w", err)
		}
	}

	yieldTime := time.Duration(yieldTimeMs) * time.Millisecond
	if yieldTime <= 0 {
		yieldTime = 500 * time.Millisecond
	}

	select {
	case <-sess.done:
		l.mu.Lock()
		delete(l.sessions, sessionID)
		l.mu.Unlock()
		sess.ptyF.Close()
		exitCode := 0
		if sess.cmd.ProcessState != nil {
			exitCode = sess.cmd.ProcessState.ExitCode()
		}
		return ShellResult{
			Output:   sess.screenText(),
			ExitCode: exitCode,
		}, nil
	case <-time.After(yieldTime):
		return ShellResult{
			Output:    sess.snapshotAndReset(),
			ExitCode:  -1,
			SessionID: sessionID,
		}, nil
	case <-ctx.Done():
		return ShellResult{}, ctx.Err()
	}
}

func (l *Local) ReadFile(_ context.Context, path string, startLine, endLine int) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}

	if startLine <= 0 && endLine <= 0 {
		return string(data), nil
	}

	lines := strings.Split(string(data), "\n")

	start := 0
	if startLine > 0 {
		start = startLine - 1
	}
	end := len(lines)
	if endLine > 0 && endLine < len(lines) {
		end = endLine
	}

	if start >= len(lines) {
		return "", nil
	}
	if start < 0 {
		start = 0
	}

	var buf strings.Builder
	for i := start; i < end; i++ {
		fmt.Fprintf(&buf, "%d\t%s\n", i+1, lines[i])
	}
	return buf.String(), nil
}

func (l *Local) WriteFile(_ context.Context, path string, content string) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(content), 0o644)
}

func (l *Local) ListDir(_ context.Context, path string, depth int) ([]DirEntry, error) {
	if depth <= 0 {
		depth = 1
	}
	var entries []DirEntry
	return entries, l.listDirRecursive(path, "", depth, &entries)
}

func (l *Local) listDirRecursive(basePath, prefix string, depth int, entries *[]DirEntry) error {
	if depth <= 0 {
		return nil
	}
	dirEntries, err := os.ReadDir(filepath.Join(basePath, prefix))
	if err != nil {
		return err
	}
	for _, e := range dirEntries {
		name := filepath.Join(prefix, e.Name())
		typ := "file"
		if e.IsDir() {
			typ = "dir"
		} else if e.Type()&os.ModeSymlink != 0 {
			typ = "symlink"
		}
		*entries = append(*entries, DirEntry{Name: name, Type: typ})
		if e.IsDir() && depth > 1 {
			if err := l.listDirRecursive(basePath, name, depth-1, entries); err != nil {
				continue
			}
		}
	}
	return nil
}

func (l *Local) GrepFiles(_ context.Context, pattern string, path string, include string) ([]GrepMatch, error) {
	args := []string{"-rn", pattern}
	if include != "" {
		args = append(args, "--include="+include)
	}
	args = append(args, path)

	cmd := exec.Command("grep", args...)
	out, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() == 1 {
			return nil, nil
		}
		return nil, err
	}

	var matches []GrepMatch
	scanner := bufio.NewScanner(bytes.NewReader(out))
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.SplitN(line, ":", 3)
		if len(parts) < 3 {
			continue
		}
		lineNum, _ := strconv.Atoi(parts[1])
		matches = append(matches, GrepMatch{
			File: parts[0],
			Line: lineNum,
			Text: parts[2],
		})
	}
	return matches, nil
}
