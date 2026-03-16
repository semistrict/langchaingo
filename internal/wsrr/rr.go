// Package wsrr implements WebSocket record and replay for testing.
//
// It records WebSocket messages exchanged between client and server,
// and replays them in tests using a local WebSocket server.
//
// Recording is controlled by the -wsrecord flag (only in test binaries).
// File format is JSON lines with direction and message data.
package wsrr

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"

	"nhooyr.io/websocket"
	"nhooyr.io/websocket/wsjson"
)

var (
	record   = new(string)
	recordMu sync.Mutex
)

func init() {
	if testing.Testing() {
		record = flag.String("wsrecord", "", "re-record WebSocket traces for files matching `regexp`")
	}
}

// entry is a single recorded WebSocket message.
type entry struct {
	Dir  string          `json:"dir"`  // "send" or "recv"
	Data json.RawMessage `json:"data"` // the JSON message
}

// RecordReplay can record or replay WebSocket interactions.
type RecordReplay struct {
	file      string
	recording bool
	entries   []entry // loaded for replay

	// Recording state
	mu       sync.Mutex
	recorded []entry

	// Replay server
	server *httptest.Server
}

// recording checks if the given file should be recorded based on the -wsrecord flag.
func isRecording(file string) bool {
	recordMu.Lock()
	defer recordMu.Unlock()
	if *record == "" {
		return false
	}
	re, err := regexp.Compile(*record)
	if err != nil {
		return false
	}
	return re.MatchString(file)
}

// OpenForTest creates a RecordReplay for the current test.
// In recording mode, use DialURL() to get the real WebSocket URL.
// In replay mode, DialURL() returns a local server URL.
func OpenForTest(t *testing.T, realURL string) *RecordReplay {
	t.Helper()

	testName := cleanFileName(t.Name())
	filename := filepath.Join("testdata", testName+".wsrr")

	if err := os.MkdirAll("testdata", 0o755); err != nil {
		t.Fatalf("wsrr: failed to create testdata directory: %v", err)
	}

	rr := &RecordReplay{
		file:      filename,
		recording: isRecording(filename),
	}

	if rr.recording {
		// Recording mode: client will connect to real URL directly.
		// We record via WrapConn.
		t.Logf("wsrr: recording to %s", filename)
	} else {
		// Replay mode: load entries and start local server.
		if err := rr.loadEntries(); err != nil {
			t.Fatalf("wsrr: failed to load %s: %v", filename, err)
		}
		rr.startReplayServer(t)
		t.Logf("wsrr: replaying from %s (%d messages)", filename, len(rr.entries))
	}

	t.Cleanup(func() {
		if rr.recording {
			if err := rr.save(); err != nil {
				t.Errorf("wsrr: failed to save recording: %v", err)
			}
		}
		if rr.server != nil {
			rr.server.Close()
		}
	})

	return rr
}

// Recording returns true if this is in recording mode.
func (rr *RecordReplay) Recording() bool {
	return rr.recording
}

// DialURL returns the base HTTP URL for the replay server.
// In replay mode, this returns the local server URL (http://...) which
// the OpenAI client will convert to ws:// and append /responses.
// The replay server handles any path, so this works transparently.
// In recording mode, returns the realURL unchanged.
func (rr *RecordReplay) DialURL(realURL string) string {
	if rr.recording {
		return realURL
	}
	return rr.server.URL
}

// RecordSend records a message sent by the client.
func (rr *RecordReplay) RecordSend(msg json.RawMessage) {
	rr.mu.Lock()
	defer rr.mu.Unlock()
	rr.recorded = append(rr.recorded, entry{Dir: "send", Data: msg})
}

// RecordRecv records a message received from the server.
func (rr *RecordReplay) RecordRecv(msg json.RawMessage) {
	rr.mu.Lock()
	defer rr.mu.Unlock()
	rr.recorded = append(rr.recorded, entry{Dir: "recv", Data: msg})
}

// ScrubEntries applies a scrubbing function to all recorded entries.
func (rr *RecordReplay) ScrubEntries(scrub func(e *entry) error) error {
	rr.mu.Lock()
	defer rr.mu.Unlock()
	for i := range rr.recorded {
		if err := scrub(&rr.recorded[i]); err != nil {
			return err
		}
	}
	return nil
}

// save writes recorded entries to the file.
func (rr *RecordReplay) save() error {
	rr.mu.Lock()
	defer rr.mu.Unlock()

	f, err := os.Create(rr.file)
	if err != nil {
		return fmt.Errorf("create %s: %w", rr.file, err)
	}
	defer f.Close()

	fmt.Fprintln(f, "wsrr trace v1")
	enc := json.NewEncoder(f)
	for _, e := range rr.recorded {
		if err := enc.Encode(e); err != nil {
			return fmt.Errorf("encode entry: %w", err)
		}
	}
	return nil
}

// loadEntries reads recorded entries from the file.
func (rr *RecordReplay) loadEntries() error {
	f, err := os.Open(rr.file)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024) // 10MB max line

	// Skip header line.
	if !scanner.Scan() {
		return fmt.Errorf("empty file")
	}
	header := scanner.Text()
	if !strings.HasPrefix(header, "wsrr trace v1") {
		return fmt.Errorf("unexpected header: %s", header)
	}

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var e entry
		if err := json.Unmarshal(line, &e); err != nil {
			return fmt.Errorf("unmarshal entry: %w", err)
		}
		rr.entries = append(rr.entries, e)
	}
	return scanner.Err()
}

// startReplayServer starts a local WebSocket server that replays recorded messages.
func (rr *RecordReplay) startReplayServer(t *testing.T) {
	t.Helper()

	rr.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Errorf("wsrr replay: websocket accept: %v", err)
			return
		}
		defer conn.Close(websocket.StatusNormalClosure, "done")
		conn.SetReadLimit(10 * 1024 * 1024)

		ctx := r.Context()
		idx := 0

		for idx < len(rr.entries) {
			e := rr.entries[idx]
			switch e.Dir {
			case "send":
				// Client should send a message. Read and discard it.
				var raw json.RawMessage
				if err := wsjson.Read(ctx, conn, &raw); err != nil {
					return // connection closed
				}
				idx++

			case "recv":
				// Send recorded message to client.
				if err := wsjson.Write(ctx, conn, e.Data); err != nil {
					return
				}
				idx++
			}
		}
	}))
}

// SkipIfNoCredentialsAndRecordingMissing skips the test if required environment
// variables are not set and no wsrr recording exists.
func SkipIfNoCredentialsAndRecordingMissing(t *testing.T, envVars ...string) {
	t.Helper()

	testName := cleanFileName(t.Name())
	filename := filepath.Join("testdata", testName+".wsrr")

	hasRecording := false
	if _, err := os.Stat(filename); err == nil {
		hasRecording = true
	}

	hasCredentials := false
	for _, envVar := range envVars {
		if os.Getenv(envVar) != "" {
			hasCredentials = true
			break
		}
	}

	if !hasRecording && !hasCredentials {
		missing := strings.Join(envVars, ", ")
		t.Skipf("%s not set and no wsrr recording at %s. Run with -wsrecord=. to record.", missing, filename)
	}
}

// cleanFileName converts a test name to a safe filename.
func cleanFileName(name string) string {
	name = strings.ReplaceAll(name, "/", "-")
	name = strings.ReplaceAll(name, " ", "-")
	return name
}
