package main

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/tmc/langchaingo/examples/coding-agent/sandbox"
)

// applyPatch parses a unified diff and applies it using the sandbox's
// ReadFile/WriteFile methods. This keeps the logic sandbox-agnostic.
func applyPatch(ctx context.Context, sb sandbox.Sandbox, patch string) error {
	files, err := parseUnifiedDiff(patch)
	if err != nil {
		return err
	}
	for _, f := range files {
		if err := applyFilePatch(ctx, sb, f); err != nil {
			return fmt.Errorf("patch %s: %w", f.path, err)
		}
	}
	return nil
}

type filePatch struct {
	path   string
	create bool
	delete bool
	hunks  []hunk
}

type hunk struct {
	oldStart int
	oldCount int
	newStart int
	newCount int
	lines    []diffLine
}

type diffLine struct {
	op   byte // ' ', '+', '-'
	text string
}

func parseUnifiedDiff(patch string) ([]filePatch, error) {
	lines := strings.Split(patch, "\n")
	var files []filePatch
	i := 0

	for i < len(lines) {
		// Skip until we find a --- line.
		if !strings.HasPrefix(lines[i], "--- ") {
			i++
			continue
		}

		oldPath := stripPrefix(lines[i])
		i++
		if i >= len(lines) || !strings.HasPrefix(lines[i], "+++ ") {
			return nil, fmt.Errorf("expected +++ after --- at line %d", i)
		}
		newPath := stripPrefix(lines[i])
		i++

		fp := filePatch{}
		if oldPath == "/dev/null" {
			fp.create = true
			fp.path = newPath
		} else if newPath == "/dev/null" {
			fp.delete = true
			fp.path = oldPath
		} else {
			fp.path = newPath
		}

		// Parse hunks.
		for i < len(lines) && strings.HasPrefix(lines[i], "@@") {
			h, nextI, err := parseHunk(lines, i)
			if err != nil {
				return nil, err
			}
			fp.hunks = append(fp.hunks, h)
			i = nextI
		}

		files = append(files, fp)
	}

	return files, nil
}

func stripPrefix(s string) string {
	// Remove "--- a/" or "+++ b/" or "--- " or "+++ " prefixes.
	s = strings.TrimPrefix(s, "--- ")
	s = strings.TrimPrefix(s, "+++ ")
	s = strings.TrimPrefix(s, "a/")
	s = strings.TrimPrefix(s, "b/")
	return s
}

func parseHunk(lines []string, i int) (hunk, int, error) {
	// Parse @@ -old,count +new,count @@
	header := lines[i]
	i++

	h := hunk{}
	parts := strings.SplitN(header, "@@", 3)
	if len(parts) < 3 {
		return h, i, fmt.Errorf("invalid hunk header: %s", header)
	}

	rangePart := strings.TrimSpace(parts[1])
	ranges := strings.Fields(rangePart)
	if len(ranges) < 2 {
		return h, i, fmt.Errorf("invalid hunk ranges: %s", rangePart)
	}

	oldRange := strings.TrimPrefix(ranges[0], "-")
	newRange := strings.TrimPrefix(ranges[1], "+")

	oldParts := strings.SplitN(oldRange, ",", 2)
	h.oldStart, _ = strconv.Atoi(oldParts[0])
	h.oldCount = 1
	if len(oldParts) > 1 {
		h.oldCount, _ = strconv.Atoi(oldParts[1])
	}

	newParts := strings.SplitN(newRange, ",", 2)
	h.newStart, _ = strconv.Atoi(newParts[0])
	h.newCount = 1
	if len(newParts) > 1 {
		h.newCount, _ = strconv.Atoi(newParts[1])
	}

	// Read diff lines until next hunk, next file, or end.
	for i < len(lines) {
		line := lines[i]
		if strings.HasPrefix(line, "@@") || strings.HasPrefix(line, "--- ") {
			break
		}
		if len(line) == 0 {
			// Empty line in diff = context line with empty content.
			h.lines = append(h.lines, diffLine{op: ' ', text: ""})
			i++
			continue
		}
		op := line[0]
		text := ""
		if len(line) > 1 {
			text = line[1:]
		}
		switch op {
		case '+', '-', ' ':
			h.lines = append(h.lines, diffLine{op: op, text: text})
		default:
			// Treat as context line (e.g. "\ No newline at end of file").
			h.lines = append(h.lines, diffLine{op: ' ', text: line})
		}
		i++
	}

	return h, i, nil
}

func applyFilePatch(ctx context.Context, sb sandbox.Sandbox, fp filePatch) error {
	if fp.delete {
		// Write empty content signals deletion; or just skip.
		// For now, we don't have a Delete method, so we'll write empty.
		return sb.WriteFile(ctx, fp.path, "")
	}

	if fp.create {
		var content strings.Builder
		for _, h := range fp.hunks {
			for _, dl := range h.lines {
				if dl.op == '+' || dl.op == ' ' {
					content.WriteString(dl.text)
					content.WriteByte('\n')
				}
			}
		}
		return sb.WriteFile(ctx, fp.path, content.String())
	}

	// Modify existing file.
	existing, err := sb.ReadFile(ctx, fp.path, 0, 0)
	if err != nil {
		return err
	}

	lines := strings.Split(existing, "\n")

	// Apply hunks in reverse order so line numbers stay valid.
	for hi := len(fp.hunks) - 1; hi >= 0; hi-- {
		h := fp.hunks[hi]
		start := h.oldStart - 1 // 0-indexed
		if start < 0 {
			start = 0
		}

		// Collect new lines for this hunk.
		var newLines []string
		for _, dl := range h.lines {
			switch dl.op {
			case ' ':
				newLines = append(newLines, dl.text)
			case '+':
				newLines = append(newLines, dl.text)
			case '-':
				// skip (removed from old)
			}
		}

		end := start + h.oldCount
		if end > len(lines) {
			end = len(lines)
		}

		// Replace old lines with new lines.
		result := make([]string, 0, len(lines)-h.oldCount+len(newLines))
		result = append(result, lines[:start]...)
		result = append(result, newLines...)
		result = append(result, lines[end:]...)
		lines = result
	}

	return sb.WriteFile(ctx, fp.path, strings.Join(lines, "\n"))
}
