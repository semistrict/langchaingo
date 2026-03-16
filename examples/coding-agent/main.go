// Coding agent example using ChatGPT OAuth auth with the HTTP Responses API.
// Uses Codex-compatible tool names (shell, read_file, list_dir, apply_patch, etc.)
// with a BubbleTea TUI for interactive chat.
//
// Usage:
//
//	go run ./examples/coding-agent/ "list the files in the current directory"
//	go run ./examples/coding-agent/  # interactive mode
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/atotto/clipboard"
	"github.com/charmbracelet/bubbles/textarea"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
	"github.com/tmc/langchaingo/examples/coding-agent/sandbox"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

const defaultModel = "gpt-5.4"

func buildSystemPrompt() string {
	now := time.Now()
	zone, _ := now.Zone()
	return fmt.Sprintf(`You are a coding agent. You have access to tools for reading files, listing directories, running shell commands, searching files, and applying patches.

Current date: %s
Timezone: %s

When the user asks you to perform a task:
1. Understand the request
2. Use the available tools to explore the codebase and make changes
3. Verify your changes work by running tests or checking output

Be concise in your responses. Show your work by describing what you're doing.`,
		now.Format("2006-01-02 (Monday)"), zone)
}

func main() {
	modelFlag := flag.String("model", "", "model name (default: gpt-5.4 for chatgpt, gemini-3.1-flash-lite for openrouter)")
	backend := flag.String("backend", "chatgpt", "backend: chatgpt or openrouter")
	resumeID := flag.String("r", "", "resume a previous session by UUID")
	flag.Parse()

	model := *modelFlag
	if model == "" {
		if m := os.Getenv("MODEL"); m != "" {
			model = m
		}
	}

	var opts []openai.Option
	switch *backend {
	case "openrouter":
		key := os.Getenv("OPENROUTER_API_KEY")
		if key == "" {
			log.Fatal("OPENROUTER_API_KEY not set")
		}
		opts = append(opts,
			openai.WithToken(key),
			openai.WithBaseURL("https://openrouter.ai/api/v1"),
		)
		if model == "" {
			model = "google/gemini-3.1-flash-lite-preview"
		}
	case "chatgpt", "":
		opts = append(opts, openai.WithChatGPTAuth(""))
		if model == "" {
			model = defaultModel
		}
	default:
		log.Fatalf("unknown backend: %s (use chatgpt or openrouter)", *backend)
	}
	opts = append(opts, openai.WithModel(model))

	llm, err := openai.New(opts...)
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	session, err := llm.OpenResponsesHTTPSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	sb := sandbox.NewLocal()

	store, err := OpenStore()
	if err != nil {
		log.Fatal(err)
	}
	defer store.Close()

	// Create or resume a session.
	var conv *Conversation
	if *resumeID != "" {
		exists, err := store.SessionExists(*resumeID)
		if err != nil || !exists {
			log.Fatalf("session %s not found", *resumeID)
		}
		// Load backend/model from saved session.
		savedBackend, savedModel, err := store.GetSessionInfo(*resumeID)
		if err != nil {
			log.Fatal(err)
		}
		if model == "" || model == defaultModel {
			model = savedModel
		}
		_ = savedBackend // backend already set from flags or could override
		conv, err = ResumeConversation(sb, store, *resumeID)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Fprintf(os.Stderr, "Resumed session %s (%d messages)\n", *resumeID, conv.MessageCount())
	}

	// If args remain after flags, run non-interactively.
	if flag.NArg() > 0 {
		prompt := strings.Join(flag.Args(), " ")
		if conv == nil {
			sessionID, err := store.NewSession(*backend, model)
			if err != nil {
				log.Fatal(err)
			}
			conv = NewConversation(buildSystemPrompt(), sb, store, sessionID)
			fmt.Fprintf(os.Stderr, "Session: %s\n", sessionID)
		}
		renderer, _ := glamour.NewTermRenderer(glamour.WithAutoStyle(), glamour.WithWordWrap(100))
		var textBuf strings.Builder
		err := conv.RunTurn(ctx, session, sb, prompt, func(ev AgentEvent) {
			switch ev.Type {
			case "text":
				textBuf.WriteString(ev.Text)
			case "tool_call":
				if s := strings.TrimSpace(textBuf.String()); s != "" {
					if rendered, err := renderer.Render(s); err == nil {
						fmt.Print(rendered)
					} else {
						fmt.Print(s)
					}
					textBuf.Reset()
				}
				fmt.Printf("\n%s\n", formatToolCallLine(ev.ToolName, ev.ToolArgs))
			case "tool_result":
				fmt.Printf("%s\n", formatToolOutput(ev.ToolOutput, ev.ToolName))
			case "error":
				fmt.Fprintf(os.Stderr, "%s %v\n", errorBullet(), ev.Error)
			case "done":
				if s := strings.TrimSpace(textBuf.String()); s != "" {
					if rendered, err := renderer.Render(s); err == nil {
						fmt.Print(rendered)
					} else {
						fmt.Print(s)
					}
					textBuf.Reset()
				}
			}
		})
		if err != nil {
			log.Fatal(err)
		}
		return
	}

	// Interactive TUI mode.
	if conv == nil {
		sessionID, err := store.NewSession(*backend, model)
		if err != nil {
			log.Fatal(err)
		}
		conv = NewConversation(buildSystemPrompt(), sb, store, sessionID)
	}
	m := newTUIModel(session, sb, conv, *backend, model)
	p := tea.NewProgram(&m, tea.WithAltScreen(), tea.WithMouseCellMotion())
	m.program = p
	if _, err := p.Run(); err != nil {
		log.Fatal(err)
	}
	fmt.Fprintf(os.Stderr, "To resume: %s -r %s\n", os.Args[0], conv.SessionID())
}

// --- Styles matching Codex TUI ---

var (
	boldStyle = lipgloss.NewStyle().Bold(true)
	faintStyle = lipgloss.NewStyle().Faint(true)
	errorStyle = lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#B91C1C", Dark: "#F87171"})
	statusStyle = lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#6B7280", Dark: "#94A3B8"})
	userBgStyle = lipgloss.NewStyle().Background(lipgloss.AdaptiveColor{Light: "#F3F4F6", Dark: "#1F2329"})
	greenBulletStyle = lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#15803D", Dark: "#4ADE80"}).Bold(true)
	redBulletStyle = lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#B91C1C", Dark: "#F87171"}).Bold(true)
	titleColorFile = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#A7F3D0"))
	titleColorSearch = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#E2E8F0"))
	titleColorPatch = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#C4B5FD"))
	shellColor      = lipgloss.AdaptiveColor{Light: "#7C3AED", Dark: "#C4B5FD"}
)

func formatTokens(n int) string {
	if n >= 1_000_000 {
		return fmt.Sprintf("%.1fm", float64(n)/1_000_000)
	}
	if n >= 1000 {
		return fmt.Sprintf("%.1fk", float64(n)/1000)
	}
	return fmt.Sprintf("%d", n)
}

func runningBullet() string { return "◐" }
func successBullet() string { return greenBulletStyle.Render("•") }
func failBullet() string   { return redBulletStyle.Render("•") }
func errorBullet() string  { return redBulletStyle.Render("✗") }

// stripBashLC removes common shell prefixes from a command string for display.
func stripBashLC(command string) string {
	command = strings.TrimSpace(command)
	for _, prefix := range []string{"bash -lc ", "/bin/bash -lc ", "zsh -lc ", "/bin/zsh -lc "} {
		if strings.HasPrefix(command, prefix) {
			rest := strings.TrimSpace(command[len(prefix):])
			// Strip surrounding quotes.
			if len(rest) >= 2 && rest[0] == '\'' && rest[len(rest)-1] == '\'' {
				rest = rest[1 : len(rest)-1]
			}
			return rest
		}
	}
	return command
}

// formatToolTitle produces the title line for a tool call (the command or tool name + key args).
func formatToolTitle(toolName, argsJSON string) string {
	switch toolName {
	case "shell":
		var args struct {
			Command []string `json:"command"`
		}
		if json.Unmarshal([]byte(argsJSON), &args) == nil && len(args.Command) > 0 {
			return stripBashLC(strings.Join(args.Command, " "))
		}
		return toolName

	case "read_file":
		var args struct {
			FilePath  string `json:"file_path"`
			StartLine int    `json:"start_line"`
			EndLine   int    `json:"end_line"`
		}
		if json.Unmarshal([]byte(argsJSON), &args) == nil {
			if args.StartLine > 0 || args.EndLine > 0 {
				return fmt.Sprintf("Read %s (lines %d-%d)", args.FilePath, args.StartLine, args.EndLine)
			}
			return "Read " + args.FilePath
		}
		return "read_file"

	case "list_dir":
		var args struct {
			DirPath string `json:"dir_path"`
		}
		if json.Unmarshal([]byte(argsJSON), &args) == nil {
			return "List " + args.DirPath
		}
		return "list_dir"

	case "grep_files":
		var args struct {
			Pattern string `json:"pattern"`
			Path    string `json:"path"`
		}
		if json.Unmarshal([]byte(argsJSON), &args) == nil {
			return fmt.Sprintf("Grep '%s' in %s", args.Pattern, args.Path)
		}
		return "grep_files"

	case "apply_patch":
		return "Apply patch"

	case "write_stdin":
		return "write_stdin"

	default:
		return toolName
	}
}

// toolTitleStyle returns the appropriate style for a tool name.
func toolTitleStyle(toolName string) lipgloss.Style {
	switch toolName {
	case "read_file", "list_dir":
		return titleColorFile
	case "grep_files":
		return titleColorSearch
	case "apply_patch":
		return titleColorPatch
	default:
		return boldStyle
	}
}

// formatToolOutput renders tool output in the Codex style: head/tail truncation with └ prefix.
func formatToolOutput(output, toolName string) string {
	output = strings.TrimRight(output, "\n")
	if output == "" {
		return faintStyle.Render("  └ (no output)")
	}

	lines := strings.Split(output, "\n")

	// For shell, show head/tail if output is long.
	if len(lines) > 6 {
		head := lines[:3]
		tail := lines[len(lines)-3:]
		var display []string
		display = append(display, head...)
		display = append(display, faintStyle.Render(fmt.Sprintf("    … +%d lines", len(lines)-6)))
		display = append(display, tail...)
		lines = display
	}

	var result []string
	for i, line := range lines {
		prefix := "    "
		if i == 0 {
			prefix = "  └ "
		}
		result = append(result, faintStyle.Render(prefix)+line)
	}
	return strings.Join(result, "\n")
}

// formatToolCallLine renders the full tool call entry (running state).
func formatToolCallLine(toolName, argsJSON string) string {
	title := formatToolTitle(toolName, argsJSON)
	switch toolName {
	case "shell":
		return fmt.Sprintf("%s %s %s", runningBullet(), boldStyle.Render("Running"), title)
	default:
		return fmt.Sprintf("%s %s", runningBullet(), toolTitleStyle(toolName).Render(title))
	}
}

// formatToolResultLine renders the completed tool call entry.
func formatToolResultLine(toolName, argsJSON, output string) string {
	title := formatToolTitle(toolName, argsJSON)
	outputRendered := formatToolOutput(output, toolName)
	if toolName == "shell" {
		// Parse exit code from output.
		bullet := successBullet()
		verb := "Ran"
		if strings.Contains(output, "Error:") || strings.HasSuffix(strings.TrimSpace(output), "[exit code: 1]") {
			bullet = failBullet()
		}
		return fmt.Sprintf("%s %s %s\n%s", bullet, boldStyle.Render(verb), title, outputRendered)
	}
	return fmt.Sprintf("%s %s\n%s", successBullet(), toolTitleStyle(toolName).Render(title), outputRendered)
}

// --- TUI ---

type agentEventMsg AgentEvent
type agentDoneMsg struct{}
type compactDoneMsg struct {
	path string
	err  error
}
type shellResultMsg struct {
	command string
	output  string
	exit    int
}

type tuiModel struct {
	session  *openai.ResponsesSession
	sb       sandbox.Sandbox
	conv     *Conversation
	textarea textarea.Model
	// sections holds rendered sections (tool calls, user messages, etc.)
	sections []string
	// mdBuf accumulates the current assistant text stream for live markdown rendering.
	mdBuf    strings.Builder
	// lines is the flattened line buffer for scrolling.
	lines    []string
	scroll   int  // scroll offset from bottom (0 = pinned to bottom)
	follow   bool // auto-scroll to bottom on new content
	running  bool
	width    int
	height   int
	status   string
	program  *tea.Program
	lastToolName   string
	lastToolArgs   string
	mdRenderer     *glamour.TermRenderer
	totalInputTok  int
	totalOutputTok int
	totalCachedTok int
	shellMode        bool
	cancelRun        context.CancelFunc // cancels the current agent run
	pendingToolCalls []string           // tool call IDs awaiting results
	backendName      string             // "chatgpt" or "openrouter"
	defaultModel     string             // model from startup flags
	// Text selection state.
	selecting bool
	hasSelection bool
	selStart  [2]int // [row, col] in screen coordinates
	selEnd    [2]int
}

func newTUIModel(session *openai.ResponsesSession, sb sandbox.Sandbox, conv *Conversation, backendName, model string) tuiModel {
	ta := textarea.New()
	ta.Prompt = ""
	ta.Placeholder = "Type a message... (ctrl+c to quit)"
	ta.Focus()
	ta.SetHeight(1)
	ta.ShowLineNumbers = false
	ta.CharLimit = 0
	// Clear all default textarea styling to avoid black background artifacts.
	clear := lipgloss.NewStyle()
	ta.FocusedStyle.Base = clear
	ta.FocusedStyle.CursorLine = clear
	ta.FocusedStyle.Placeholder = lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#A1A1AA", Dark: "#6B7280"})
	ta.FocusedStyle.EndOfBuffer = clear
	ta.FocusedStyle.Text = clear
	ta.BlurredStyle.Base = clear
	ta.BlurredStyle.CursorLine = clear
	ta.BlurredStyle.EndOfBuffer = clear
	ta.BlurredStyle.Text = clear

	renderer, _ := glamour.NewTermRenderer(
		glamour.WithAutoStyle(),
		glamour.WithWordWrap(100),
	)

	m := tuiModel{
		session:      session,
		sb:           sb,
		conv:         conv,
		textarea:     ta,
		follow:       true,
		status:       "ready",
		mdRenderer:   renderer,
		backendName:  backendName,
		defaultModel: model,
	}

	// Replay messages from the loaded conversation into the transcript.
	for _, msg := range conv.MessagesAfterLastCompaction() {
		switch msg.Role {
		case llms.ChatMessageTypeHuman:
			for _, part := range msg.Parts {
				if tc, ok := part.(llms.TextContent); ok {
					rendered := userBgStyle.Padding(0, 1).Render("› " + tc.Text)
					m.sections = append(m.sections, rendered)
				}
			}
		case llms.ChatMessageTypeAI:
			for _, part := range msg.Parts {
				if tc, ok := part.(llms.TextContent); ok && tc.Text != "" {
					m.sections = append(m.sections, m.renderMarkdown(tc.Text))
				}
			}
		}
	}

	return m
}

func (m *tuiModel) Init() tea.Cmd {
	return textarea.Blink
}

func (m *tuiModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.MouseMsg:
		switch {
		case msg.Button == tea.MouseButtonWheelUp:
			m.scrollUp(3)
		case msg.Button == tea.MouseButtonWheelDown:
			m.scrollDown(3)
		case msg.Button == tea.MouseButtonLeft && msg.Action == tea.MouseActionPress:
			if msg.Y < m.transcriptHeight() {
				m.selecting = true
				m.hasSelection = false
				m.selStart = [2]int{msg.Y, msg.X}
				m.selEnd = m.selStart
			} else {
				// Click outside transcript clears selection.
				m.hasSelection = false
			}
		case msg.Action == tea.MouseActionMotion && m.selecting:
			m.selEnd = [2]int{msg.Y, msg.X}
			m.hasSelection = true
		case msg.Action == tea.MouseActionRelease && m.selecting:
			m.selEnd = [2]int{msg.Y, msg.X}
			m.selecting = false
			if text := m.selectedText(); text != "" {
				m.hasSelection = true
				clipboard.WriteAll(text)
				m.status = "copied to clipboard"
			} else {
				m.hasSelection = false
			}
		}

	case tea.KeyMsg:
		// Any keystroke clears text selection.
		m.hasSelection = false

		// Esc interrupts the current agent run.
		if msg.Type == tea.KeyEsc && m.running && m.cancelRun != nil {
			m.cancelRun()
			m.cancelRun = nil
			m.running = false
			m.flushMarkdown()
			// Send failure results for any pending tool calls.
			m.conv.CancelPendingToolCalls(m.pendingToolCalls)
			m.pendingToolCalls = nil
			m.sections = append(m.sections, faintStyle.Render("(interrupted)"))
			m.rebuildLines()
			return m, nil
		}

		// Backspace at empty input exits shell mode (as if deleting the !).
		if msg.Type == tea.KeyBackspace && m.shellMode && strings.TrimSpace(m.textarea.Value()) == "" {
			m.shellMode = false
			return m, nil
		}

		// '!' at empty input enters shell mode — return early so the '!'
		// character is not passed to the textarea.
		if msg.Type == tea.KeyRunes && string(msg.Runes) == "!" && !m.shellMode && strings.TrimSpace(m.textarea.Value()) == "" {
			m.shellMode = true
			return m, nil
		}

		switch msg.Type {
		case tea.KeyCtrlC:
			return m, tea.Quit
		case tea.KeyUp, tea.KeyPgUp:
			m.scrollUp(5)
		case tea.KeyDown, tea.KeyPgDown:
			m.scrollDown(5)
		case tea.KeyEnter:
			if m.running {
				break
			}
			text := strings.TrimSpace(m.textarea.Value())
			if text == "" {
				break
			}
			m.textarea.Reset()

			// Shell mode: execute command directly.
			if m.shellMode {
				m.sections = append(m.sections, lipgloss.NewStyle().Foreground(shellColor).Render("! ")+text)
				m.rebuildLines()
				m.running = true
				m.status = "running..."
				sb := m.sb
				ctx, cancel := context.WithCancel(context.Background())
				m.cancelRun = cancel
				cmds = append(cmds, func() tea.Msg {
					result, err := sb.Shell(ctx, []string{"bash", "-lc", text}, "", 30000)
					output := result.Output
					if err != nil {
						output = fmt.Sprintf("Error: %v", err)
					}
					return shellResultMsg{command: text, output: output, exit: result.ExitCode}
				})
				break
			}

			// Handle /model command.
			if strings.HasPrefix(text, "/model") {
				newModel := strings.TrimSpace(strings.TrimPrefix(text, "/model"))
				if newModel == "" {
					current := m.conv.Model
					if current == "" {
						current = "(session default)"
					}
					m.sections = append(m.sections, faintStyle.Render("Current model: "+current))
				} else {
					m.conv.Model = newModel
					m.sections = append(m.sections, faintStyle.Render("Switched to model: "+newModel))
				}
				m.rebuildLines()
				break
			}

			// Handle /compact command.
			if text == "/compact" {
				m.sections = append(m.sections, faintStyle.Render("Compacting conversation..."))
				m.rebuildLines()
				m.running = true
				m.status = "compacting..."
				p := m.program
				session := m.session
				sb := m.sb
				conv := m.conv
				ctx, cancel := context.WithCancel(context.Background())
				m.cancelRun = cancel
				cmds = append(cmds, func() tea.Msg {
					path, err := conv.CompactAndResume(ctx, session, sb, func(ev AgentEvent) {
						if p != nil {
							p.Send(agentEventMsg(ev))
						}
					})
					return compactDoneMsg{path: path, err: err}
				})
				break
			}

			m.running = true
			m.status = "thinking..."

			// User message with surface background, › prefix.
			rendered := userBgStyle.Padding(0, 1).Render("› " + text)
			m.sections = append(m.sections, rendered)
			m.rebuildLines()

			p := m.program
			session := m.session
			sb := m.sb
			conv := m.conv
			ctx, cancel := context.WithCancel(context.Background())
			m.cancelRun = cancel
			cmds = append(cmds, func() tea.Msg {
				_ = conv.RunTurn(ctx, session, sb, text, func(ev AgentEvent) {
					if p != nil {
						p.Send(agentEventMsg(ev))
					}
				})
				return agentDoneMsg{}
			})
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.textarea.SetWidth(msg.Width)
		m.rebuildLines()

	case agentEventMsg:
		ev := AgentEvent(msg)
		switch ev.Type {
		case "text":
			// Accumulate text and re-render markdown live.
			m.mdBuf.WriteString(ev.Text)
			m.rebuildLines()
		case "tool_call":
			m.flushMarkdown()
			m.lastToolName = ev.ToolName
			m.lastToolArgs = ev.ToolArgs
			if ev.ToolCallID != "" {
				m.pendingToolCalls = append(m.pendingToolCalls, ev.ToolCallID)
			}
			m.sections = append(m.sections, formatToolCallLine(ev.ToolName, ev.ToolArgs))
			m.rebuildLines()
			m.status = fmt.Sprintf("running %s...", ev.ToolName)
		case "tool_result":
			// Remove from pending.
			m.pendingToolCalls = nil
			m.sections = append(m.sections, formatToolOutput(ev.ToolOutput, m.lastToolName))
			m.rebuildLines()
			m.status = "thinking..."
		case "error":
			m.flushMarkdown()
			m.sections = append(m.sections, errorStyle.Render(fmt.Sprintf("%s %v", errorBullet(), ev.Error)))
			m.rebuildLines()
		case "done":
			m.totalInputTok += ev.InputTokens
			m.totalOutputTok += ev.OutputTokens
			m.totalCachedTok += ev.CachedTokens
		}

	case agentDoneMsg:
		m.flushMarkdown()
		m.running = false
		m.rebuildLines()

	case shellResultMsg:
		// Show result in transcript.
		bullet := successBullet()
		if msg.exit != 0 {
			bullet = failBullet()
		}
		m.sections = append(m.sections, fmt.Sprintf("%s %s %s\n%s",
			bullet, boldStyle.Render("Ran"), msg.command,
			formatToolOutput(msg.output, "shell")))
		m.rebuildLines()
		m.status = "shell (backspace to exit)"

		// Add to conversation history so the agent sees it on next turn.
		shellMsg := fmt.Sprintf("The user ran a shell command:\n$ %s\n\nOutput:\n%s\n\n[exit code: %d]",
			msg.command, msg.output, msg.exit)
		m.conv.AddSystemContext(shellMsg)

	case compactDoneMsg:
		m.flushMarkdown()
		m.running = false
		if msg.err != nil {
			m.sections = append(m.sections, errorStyle.Render(fmt.Sprintf("%s compact failed: %v", errorBullet(), msg.err)))
		}
		m.status = fmt.Sprintf("ready (%d msgs)", m.conv.MessageCount())
		m.rebuildLines()
	}

	var cmd tea.Cmd
	m.textarea, cmd = m.textarea.Update(msg)
	cmds = append(cmds, cmd)

	return m, tea.Batch(cmds...)
}

// renderComposer renders the textarea with a › or ! prefix.
func (m *tuiModel) renderComposer() string {
	view := m.textarea.View()
	lines := strings.Split(strings.TrimRight(view, "\n"), "\n")
	if len(lines) == 0 {
		lines = []string{""}
	}
	prefix := "› "
	indent := "  "
	if m.shellMode {
		prefix = lipgloss.NewStyle().Foreground(shellColor).Render("! ")
		indent = "  "
	}
	lines[0] = prefix + lines[0]
	for i := 1; i < len(lines); i++ {
		lines[i] = indent + lines[i]
	}
	return strings.Join(lines, "\n")
}

// flushMarkdown renders any buffered markdown text and appends it as a section.
func (m *tuiModel) flushMarkdown() {
	raw := strings.TrimSpace(m.mdBuf.String())
	if raw == "" {
		return
	}
	m.mdBuf.Reset()
	rendered := m.renderMarkdown(raw)
	m.sections = append(m.sections, rendered)
}

// renderMarkdown renders markdown text through glamour with • prefix.
func (m *tuiModel) renderMarkdown(text string) string {
	if m.mdRenderer == nil {
		return "• " + text
	}
	rendered, err := m.mdRenderer.Render(text)
	if err != nil {
		return "• " + text
	}
	rendered = strings.TrimSpace(rendered)
	if rendered == "" {
		return ""
	}
	// Add • prefix to first line, indent rest.
	lines := strings.Split(rendered, "\n")
	for i, line := range lines {
		if i == 0 {
			lines[i] = "• " + line
		} else {
			lines[i] = "  " + line
		}
	}
	return strings.Join(lines, "\n")
}

// transcriptHeight returns the number of lines available for the transcript.
func (m *tuiModel) transcriptHeight() int {
	// footer: top border(1) + input(1) + bottom border(1) + status(1)
	h := m.height - 4
	if h < 1 {
		h = 1
	}
	return h
}

// rebuildLines flattens sections + live markdown into the line buffer.
func (m *tuiModel) rebuildLines() {
	var parts []string
	for _, s := range m.sections {
		if s != "" {
			parts = append(parts, s)
		}
	}

	raw := strings.TrimSpace(m.mdBuf.String())
	if raw != "" {
		if rendered := m.renderMarkdown(raw); rendered != "" {
			parts = append(parts, rendered)
		}
	}

	content := strings.Join(parts, "\n\n")
	if content == "" {
		m.lines = nil
	} else {
		m.lines = strings.Split(content, "\n")
	}

	if m.follow {
		m.scroll = 0
	}
}

// selectedText extracts the text between selStart and selEnd from visible lines.
func (m *tuiModel) selectedText() string {
	visible := m.visibleLines()
	startRow, startCol := m.selStart[0], m.selStart[1]
	endRow, endCol := m.selEnd[0], m.selEnd[1]

	// Normalize so start <= end.
	if startRow > endRow || (startRow == endRow && startCol > endCol) {
		startRow, endRow = endRow, startRow
		startCol, endCol = endCol, startCol
	}

	var buf strings.Builder
	for row := startRow; row <= endRow && row < len(visible); row++ {
		// Strip ANSI for text extraction.
		line := stripAnsi(visible[row])
		sc, ec := 0, len(line)
		if row == startRow {
			sc = startCol
		}
		if row == endRow {
			ec = endCol + 1
		}
		if sc > len(line) {
			sc = len(line)
		}
		if ec > len(line) {
			ec = len(line)
		}
		if sc < ec {
			buf.WriteString(line[sc:ec])
		}
		if row < endRow {
			buf.WriteByte('\n')
		}
	}
	return buf.String()
}

// stripAnsi removes ANSI escape sequences from a string.
func stripAnsi(s string) string {
	var buf strings.Builder
	i := 0
	for i < len(s) {
		if s[i] == '\x1b' && i+1 < len(s) && s[i+1] == '[' {
			// Skip until we hit a letter.
			j := i + 2
			for j < len(s) && !((s[j] >= 'A' && s[j] <= 'Z') || (s[j] >= 'a' && s[j] <= 'z')) {
				j++
			}
			if j < len(s) {
				j++ // skip the final letter
			}
			i = j
			continue
		}
		buf.WriteByte(s[i])
		i++
	}
	return buf.String()
}

func (m *tuiModel) scrollUp(n int) {
	m.follow = false
	maxScroll := len(m.lines) - m.transcriptHeight()
	if maxScroll < 0 {
		maxScroll = 0
	}
	m.scroll += n
	if m.scroll > maxScroll {
		m.scroll = maxScroll
	}
}

func (m *tuiModel) scrollDown(n int) {
	m.scroll -= n
	if m.scroll <= 0 {
		m.scroll = 0
		m.follow = true
	}
}

// visibleLines returns the slice of lines to display in the transcript area.
func (m *tuiModel) visibleLines() []string {
	h := m.transcriptHeight()
	total := len(m.lines)
	if total <= h {
		return m.lines
	}

	// scroll is offset from bottom
	end := total - m.scroll
	start := end - h
	if start < 0 {
		start = 0
	}
	if end > total {
		end = total
	}
	return m.lines[start:end]
}

func (m *tuiModel) View() string {
	if m.width == 0 || m.height == 0 {
		return "Loading..."
	}

	h := m.transcriptHeight()

	// Render transcript lines, highlight selection, pad to fill height.
	visible := m.visibleLines()
	selStyle := lipgloss.NewStyle().Reverse(true)

	startRow, startCol := m.selStart[0], m.selStart[1]
	endRow, endCol := m.selEnd[0], m.selEnd[1]
	if startRow > endRow || (startRow == endRow && startCol > endCol) {
		startRow, endRow = endRow, startRow
		startCol, endCol = endCol, startCol
	}

	var transcript strings.Builder
	for i, line := range visible {
		if (m.selecting || m.hasSelection) && i >= startRow && i <= endRow {
			runes := []rune(stripAnsi(line))
			sc, ec := 0, len(runes)
			if i == startRow {
				sc = startCol
			}
			if i == endRow {
				ec = endCol + 1
			}
			if sc > len(runes) {
				sc = len(runes)
			}
			if ec > len(runes) {
				ec = len(runes)
			}
			before := string(runes[:sc])
			sel := string(runes[sc:ec])
			after := string(runes[ec:])
			transcript.WriteString(before + selStyle.Render(sel) + after)
		} else {
			transcript.WriteString(line)
		}
		if i < len(visible)-1 {
			transcript.WriteByte('\n')
		}
	}
	// Pad remaining lines.
	for i := len(visible); i < h; i++ {
		transcript.WriteByte('\n')
	}

	borderColor := lipgloss.AdaptiveColor{Light: "#A1A1AA", Dark: "#6B7280"}
	if m.shellMode {
		borderColor = shellColor
	}
	inputStyle := lipgloss.NewStyle().
		BorderTop(true).
		BorderBottom(true).
		BorderStyle(lipgloss.NormalBorder()).
		BorderForeground(borderColor).
		Padding(0, 1).
		Width(m.width)

	composer := m.renderComposer()

	statusLeft := m.status
	if !m.running {
		if m.shellMode {
			statusLeft = "shell (backspace to exit)"
		} else {
			statusLeft = "ready"
		}
	}
	activeModel := m.conv.Model
	if activeModel == "" {
		activeModel = m.defaultModel
	}
	sid := m.conv.SessionID()
	if len(sid) > 8 {
		sid = sid[:8]
	}
	statusRight := fmt.Sprintf("%s %s:%s %d msgs", sid, m.backendName, activeModel, m.conv.MessageCount())
	if m.totalInputTok > 0 || m.totalOutputTok > 0 {
		tok := fmt.Sprintf("%s in / %s out", formatTokens(m.totalInputTok), formatTokens(m.totalOutputTok))
		if m.totalCachedTok > 0 {
			tok += fmt.Sprintf(" (%s cached)", formatTokens(m.totalCachedTok))
		}
		statusRight += " | " + tok
	}
	gap := m.width - len(statusLeft) - len(statusRight) - 2
	if gap < 1 {
		gap = 1
	}
	status := statusStyle.Render(" " + statusLeft + strings.Repeat(" ", gap) + statusRight + " ")

	return transcript.String() + "\n" + inputStyle.Render(composer) + "\n" + status
}
