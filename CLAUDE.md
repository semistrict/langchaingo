# CLAUDE.md

## Test Recordings

This project uses two record/replay systems for integration tests that hit external APIs.

### httprr (HTTP record/replay)

For REST API tests (Chat Completions, embeddings, etc.). Recordings are `.httprr` files in `testdata/` directories.

```bash
# Record all httprr tests in a package
OPENAI_API_KEY=sk-... go test ./llms/openai -httprecord=. -v

# Record a specific test
OPENAI_API_KEY=sk-... go test ./llms/openai -httprecord=. -run "TestWebSearch$" -v

# Replay (no API key needed)
go test ./llms/openai -run "TestWebSearch$" -v
```

The `-httprecord` flag takes a regex matched against filenames. Use `.` to match all.

### wsrr (WebSocket record/replay)

For WebSocket API tests (Responses API). Recordings are `.wsrr` files in `testdata/` directories.

```bash
# Record all wsrr tests
OPENAI_API_KEY=sk-... go test ./llms/openai -wsrecord=. -run "TestResponses(WebSearch|MultiTurn|StreamingLive)" -v

# Record a specific test
OPENAI_API_KEY=sk-... go test ./llms/openai -wsrecord=. -run "TestResponsesWebSearch$" -v

# Replay (no API key needed)
go test ./llms/openai -run "TestResponsesWebSearch$" -v
```

### Compressing httprr recordings

```bash
go run ./internal/devtools/rrtool pack -r
```

## OpenAI Provider

The OpenAI provider (`llms/openai/`) supports two APIs:

- **Chat Completions** (`GenerateContent`) — REST+SSE, the standard `llms.Model` interface
- **Responses API** (`OpenResponsesSession`) — persistent WebSocket sessions with auto-chaining via `previous_response_id`

### Web Search

- Chat Completions: use `llms.WithWebSearch()` with search-preview models (`gpt-4o-search-preview`, `gpt-4o-mini-search-preview`)
- Responses API: use `llms.WithTools([]llms.Tool{{Type: "web_search"}})` with any model (e.g. `gpt-5.4`)
