// Package chatgptauth provides OAuth token management for ChatGPT authentication.
//
// ChatGPT auth uses OAuth tokens from an auth.json file (produced by
// "npx @openai/codex login") instead of a static API key. Tokens are
// automatically refreshed when they expire.
//
// The auth.json file is searched in order:
//  1. Explicit path passed to [Load]
//  2. $CODEX_HOME/auth.json
//  3. ~/.codex/auth.json
package chatgptauth

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	// DefaultTokenURL is the OpenAI OAuth token endpoint.
	DefaultTokenURL = "https://auth.openai.com/oauth/token"
	// DefaultClientID is the default OAuth client ID for Codex.
	DefaultClientID = "app_EMoamEEZ73f0CkXaXp7hrann"
	// DefaultBaseURL is the ChatGPT backend API base URL.
	// The langchaingo client appends "/chat/completions" etc. to this.
	DefaultBaseURL = "https://chatgpt.com/backend-api/codex"

	refreshScope  = "openai profile email offline_access"
	refreshMargin = 5 * time.Minute // refresh 5min before expiry
)

// authFile is the JSON structure of the auth.json file.
type authFile struct {
	Tokens *tokens `json:"tokens,omitempty"`
}

type tokens struct {
	IDToken      string `json:"id_token,omitempty"`
	AccessToken  string `json:"access_token,omitempty"`
	RefreshToken string `json:"refresh_token,omitempty"`
	AccountID    string `json:"account_id,omitempty"`
}

// Auth holds ChatGPT OAuth credentials and handles token refresh.
type Auth struct {
	mu           sync.Mutex
	accessToken  string
	refreshToken string
	accountID    string
	expiry       time.Time
	tokenURL     string
	clientID     string
	authFilePath string
	httpClient   *http.Client
}

// Option configures the Auth.
type Option func(*Auth)

// WithTokenURL overrides the OAuth token endpoint.
func WithTokenURL(url string) Option {
	return func(a *Auth) { a.tokenURL = url }
}

// WithClientID overrides the OAuth client ID.
func WithClientID(id string) Option {
	return func(a *Auth) { a.clientID = id }
}

// WithHTTPClient sets the HTTP client used for token refresh.
func WithHTTPClient(c *http.Client) Option {
	return func(a *Auth) { a.httpClient = c }
}

// Load reads an auth.json file and returns an Auth that manages token refresh.
// If path is empty, it searches default locations.
func Load(path string, opts ...Option) (*Auth, error) {
	if path == "" {
		var err error
		path, err = findAuthFile()
		if err != nil {
			return nil, err
		}
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read auth file: %w", err)
	}

	var af authFile
	if err := json.Unmarshal(data, &af); err != nil {
		return nil, fmt.Errorf("parse auth file: %w", err)
	}
	if af.Tokens == nil {
		return nil, fmt.Errorf("auth file %s has no tokens", path)
	}
	if af.Tokens.AccessToken == "" {
		return nil, fmt.Errorf("auth file %s has no access_token", path)
	}
	if af.Tokens.RefreshToken == "" {
		return nil, fmt.Errorf("auth file %s has no refresh_token", path)
	}

	a := &Auth{
		accessToken:  af.Tokens.AccessToken,
		refreshToken: af.Tokens.RefreshToken,
		accountID:    af.Tokens.AccountID,
		tokenURL:     DefaultTokenURL,
		clientID:     DefaultClientID,
		authFilePath: path,
		httpClient:   http.DefaultClient,
	}

	for _, opt := range opts {
		opt(a)
	}

	// Parse expiry from the access token JWT.
	if exp, err := parseJWTExpiry(a.accessToken); err == nil {
		a.expiry = exp
	}

	// If no account_id in file, try to extract from id_token.
	if a.accountID == "" && af.Tokens.IDToken != "" {
		if id, err := extractAccountID(af.Tokens.IDToken); err == nil {
			a.accountID = id
		}
	}

	return a, nil
}

// Token returns the current access token and extra headers, refreshing if needed.
// This satisfies openaiclient.TokenProvider.
func (a *Auth) Token() (string, http.Header, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if time.Now().Add(refreshMargin).Before(a.expiry) {
		return a.accessToken, a.headers(), nil
	}

	if err := a.refresh(); err != nil {
		return "", nil, fmt.Errorf("refresh token: %w", err)
	}
	return a.accessToken, a.headers(), nil
}

// AccountID returns the ChatGPT account ID.
func (a *Auth) AccountID() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.accountID
}

func (a *Auth) headers() http.Header {
	h := http.Header{}
	if a.accountID != "" {
		h.Set("ChatGPT-Account-ID", a.accountID)
	}
	return h
}

func (a *Auth) refresh() error {
	form := url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {a.refreshToken},
		"client_id":     {a.clientID},
		"scope":         {refreshScope},
	}

	resp, err := a.httpClient.PostForm(a.tokenURL, form)
	if err != nil {
		return fmt.Errorf("POST %s: %w", a.tokenURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errResp struct {
			Error            string `json:"error"`
			ErrorDescription string `json:"error_description"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&errResp); err == nil && errResp.Error != "" {
			return fmt.Errorf("oauth error: %s: %s", errResp.Error, errResp.ErrorDescription)
		}
		return fmt.Errorf("oauth token endpoint returned %d", resp.StatusCode)
	}

	var tokenResp struct {
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
		IDToken      string `json:"id_token"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tokenResp); err != nil {
		return fmt.Errorf("decode token response: %w", err)
	}

	a.accessToken = tokenResp.AccessToken
	if tokenResp.RefreshToken != "" {
		a.refreshToken = tokenResp.RefreshToken
	}

	if exp, err := parseJWTExpiry(a.accessToken); err == nil {
		a.expiry = exp
	}

	if tokenResp.IDToken != "" {
		if id, err := extractAccountID(tokenResp.IDToken); err == nil {
			a.accountID = id
		}
	}

	// Persist refreshed tokens to auth.json.
	a.persistTokens(tokenResp.IDToken)

	return nil
}

func (a *Auth) persistTokens(idToken string) {
	data, err := os.ReadFile(a.authFilePath)
	if err != nil {
		return
	}

	var af authFile
	if err := json.Unmarshal(data, &af); err != nil || af.Tokens == nil {
		return
	}

	af.Tokens.AccessToken = a.accessToken
	af.Tokens.RefreshToken = a.refreshToken
	if idToken != "" {
		af.Tokens.IDToken = idToken
	}
	if a.accountID != "" {
		af.Tokens.AccountID = a.accountID
	}

	out, err := json.MarshalIndent(af, "", "  ")
	if err != nil {
		return
	}
	_ = os.WriteFile(a.authFilePath, out, 0o600)
}

// findAuthFile searches default locations for auth.json.
func findAuthFile() (string, error) {
	candidates := authFileCandidates()
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("no auth.json found; searched: %s (run 'npx @openai/codex login' first)", strings.Join(candidates, ", "))
}

func authFileCandidates() []string {
	var paths []string
	if v := os.Getenv("CODEX_HOME"); v != "" {
		paths = append(paths, filepath.Join(v, "auth.json"))
	}
	if home, err := os.UserHomeDir(); err == nil {
		paths = append(paths, filepath.Join(home, ".codex", "auth.json"))
		paths = append(paths, filepath.Join(home, ".chatgpt-local", "auth.json"))
	}
	return paths
}

// parseJWTExpiry decodes a JWT and extracts the "exp" claim.
func parseJWTExpiry(token string) (time.Time, error) {
	claims, err := parseJWTPayload(token)
	if err != nil {
		return time.Time{}, err
	}
	exp, ok := claims["exp"].(float64)
	if !ok {
		return time.Time{}, fmt.Errorf("no exp claim in JWT")
	}
	return time.Unix(int64(exp), 0), nil
}

// extractAccountID extracts the ChatGPT account ID from an id_token JWT.
func extractAccountID(idToken string) (string, error) {
	claims, err := parseJWTPayload(idToken)
	if err != nil {
		return "", err
	}
	// The account ID is nested under this claim key.
	if id, ok := claims["https://api.openai.com/auth"].(map[string]any); ok {
		if acct, ok := id["chatgpt_account_id"].(string); ok {
			return acct, nil
		}
	}
	// Flat claim path used by some token formats.
	if id, ok := claims["chatgpt_account_id"].(string); ok {
		return id, nil
	}
	return "", fmt.Errorf("no chatgpt_account_id in id_token")
}

// parseJWTPayload decodes the payload (second segment) of a JWT without
// verifying the signature — we trust the token from our own auth file.
func parseJWTPayload(token string) (map[string]any, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return nil, fmt.Errorf("invalid JWT: expected 3 parts, got %d", len(parts))
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, fmt.Errorf("decode JWT payload: %w", err)
	}
	var claims map[string]any
	if err := json.Unmarshal(payload, &claims); err != nil {
		return nil, fmt.Errorf("unmarshal JWT claims: %w", err)
	}
	return claims, nil
}
