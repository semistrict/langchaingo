package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/google/uuid"
	_ "github.com/mattn/go-sqlite3"
	"github.com/tmc/langchaingo/llms"
)

const dbDir = ".langchaingoexampleagent"

// Store persists conversation messages in SQLite.
type Store struct {
	db *sql.DB
}

// OpenStore opens (or creates) the SQLite database at ~/.langchaingoexampleagent/sessions.db.
func OpenStore() (*Store, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("get home dir: %w", err)
	}
	dir := filepath.Join(home, dbDir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create db dir: %w", err)
	}
	dbPath := filepath.Join(dir, "sessions.db")
	db, err := sql.Open("sqlite3", dbPath+"?_journal_mode=WAL")
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}

	if _, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS sessions (
			id TEXT PRIMARY KEY,
			backend TEXT NOT NULL,
			model TEXT NOT NULL,
			created_at TEXT NOT NULL,
			updated_at TEXT NOT NULL
		);
		CREATE TABLE IF NOT EXISTS messages (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			session_id TEXT NOT NULL REFERENCES sessions(id),
			seq INTEGER NOT NULL,
			role TEXT NOT NULL,
			parts_json TEXT NOT NULL,
			created_at TEXT NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, seq);
	`); err != nil {
		db.Close()
		return nil, fmt.Errorf("create tables: %w", err)
	}

	return &Store{db: db}, nil
}

// Close closes the database.
func (s *Store) Close() error {
	return s.db.Close()
}

// NewSession creates a new session and returns its UUID.
func (s *Store) NewSession(backend, model string) (string, error) {
	id := uuid.New().String()
	now := time.Now().UTC().Format(time.RFC3339)
	_, err := s.db.Exec(
		"INSERT INTO sessions (id, backend, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
		id, backend, model, now, now,
	)
	if err != nil {
		return "", fmt.Errorf("insert session: %w", err)
	}
	return id, nil
}

// SaveMessage persists a single MessageContent to the session.
func (s *Store) SaveMessage(sessionID string, seq int, msg llms.MessageContent) error {
	partsJSON, err := marshalParts(msg.Parts)
	if err != nil {
		return fmt.Errorf("marshal parts: %w", err)
	}
	now := time.Now().UTC().Format(time.RFC3339)
	_, err = s.db.Exec(
		"INSERT INTO messages (session_id, seq, role, parts_json, created_at) VALUES (?, ?, ?, ?, ?)",
		sessionID, seq, string(msg.Role), partsJSON, now,
	)
	if err != nil {
		return fmt.Errorf("insert message: %w", err)
	}
	_, err = s.db.Exec("UPDATE sessions SET updated_at = ? WHERE id = ?", now, sessionID)
	return err
}

// LoadMessages loads all messages for a session, ordered by seq.
func (s *Store) LoadMessages(sessionID string) ([]llms.MessageContent, error) {
	rows, err := s.db.Query(
		"SELECT role, parts_json FROM messages WHERE session_id = ? ORDER BY seq",
		sessionID,
	)
	if err != nil {
		return nil, fmt.Errorf("query messages: %w", err)
	}
	defer rows.Close()

	var messages []llms.MessageContent
	for rows.Next() {
		var role, partsJSON string
		if err := rows.Scan(&role, &partsJSON); err != nil {
			return nil, fmt.Errorf("scan message: %w", err)
		}
		parts, err := unmarshalParts(partsJSON)
		if err != nil {
			return nil, fmt.Errorf("unmarshal parts: %w", err)
		}
		messages = append(messages, llms.MessageContent{
			Role:  llms.ChatMessageType(role),
			Parts: parts,
		})
	}
	return messages, rows.Err()
}

// SessionExists checks if a session ID exists.
func (s *Store) SessionExists(id string) (bool, error) {
	var count int
	err := s.db.QueryRow("SELECT COUNT(*) FROM sessions WHERE id = ?", id).Scan(&count)
	return count > 0, err
}

// GetSessionInfo returns backend and model for a session.
func (s *Store) GetSessionInfo(id string) (backend, model string, err error) {
	err = s.db.QueryRow("SELECT backend, model FROM sessions WHERE id = ?", id).Scan(&backend, &model)
	return
}

// MessageCount returns the number of messages in a session.
func (s *Store) MessageCount(sessionID string) (int, error) {
	var count int
	err := s.db.QueryRow("SELECT COUNT(*) FROM messages WHERE session_id = ?", sessionID).Scan(&count)
	return count, err
}

// --- JSON serialization for parts ---

// partEnvelope wraps a ContentPart with a type discriminator for JSON.
type partEnvelope struct {
	Type string          `json:"type"`
	Data json.RawMessage `json:"data"`
}

func marshalParts(parts []llms.ContentPart) (string, error) {
	var envelopes []partEnvelope
	for _, p := range parts {
		var typ string
		var data any
		switch v := p.(type) {
		case llms.TextContent:
			typ = "text"
			data = v
		case llms.ToolCall:
			typ = "tool_call"
			data = v
		case llms.ToolCallResponse:
			typ = "tool_call_response"
			data = v
		default:
			typ = "unknown"
			data = struct{}{}
		}
		b, err := json.Marshal(data)
		if err != nil {
			return "", err
		}
		envelopes = append(envelopes, partEnvelope{Type: typ, Data: b})
	}
	b, err := json.Marshal(envelopes)
	return string(b), err
}

func unmarshalParts(s string) ([]llms.ContentPart, error) {
	var envelopes []partEnvelope
	if err := json.Unmarshal([]byte(s), &envelopes); err != nil {
		return nil, err
	}
	var parts []llms.ContentPart
	for _, env := range envelopes {
		switch env.Type {
		case "text":
			var v llms.TextContent
			if err := json.Unmarshal(env.Data, &v); err != nil {
				return nil, err
			}
			parts = append(parts, v)
		case "tool_call":
			var v llms.ToolCall
			if err := json.Unmarshal(env.Data, &v); err != nil {
				return nil, err
			}
			parts = append(parts, v)
		case "tool_call_response":
			var v llms.ToolCallResponse
			if err := json.Unmarshal(env.Data, &v); err != nil {
				return nil, err
			}
			parts = append(parts, v)
		}
	}
	return parts, nil
}
