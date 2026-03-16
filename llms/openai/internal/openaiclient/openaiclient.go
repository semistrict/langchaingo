package openaiclient

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strings"
)

const (
	defaultBaseURL              = "https://api.openai.com/v1"
	defaultFunctionCallBehavior = "auto"
)

// ErrEmptyResponse is returned when the OpenAI API returns an empty response.
var ErrEmptyResponse = errors.New("empty response")

type APIType string

const (
	APITypeOpenAI  APIType = "OPEN_AI"
	APITypeAzure   APIType = "AZURE"
	APITypeAzureAD APIType = "AZURE_AD"
)

// TokenProvider returns a fresh token and optional extra HTTP headers.
// It is called before each request, allowing token refresh for OAuth flows.
type TokenProvider func() (token string, extraHeaders http.Header, err error)

// Client is a client for the OpenAI API.
type Client struct {
	token         string
	Model         string
	baseURL       string
	organization  string
	apiType       APIType
	httpClient    Doer
	tokenProvider TokenProvider // optional; overrides static token when set

	EmbeddingModel      string
	EmbeddingDimensions int
	// required when APIType is APITypeAzure or APITypeAzureAD
	apiVersion string

	ResponseFormat *ResponseFormat
}

// Option is an option for the OpenAI client.
type Option func(*Client) error

// WithEmbeddingDimensions allows to setup specific dimensions for embedding's vector
func WithEmbeddingDimensions(dimensions int) Option {
	return func(c *Client) error {
		c.EmbeddingDimensions = dimensions
		return nil
	}
}

// WithTokenProvider sets a dynamic token provider that is called before each
// request. This is used for OAuth flows (e.g., ChatGPT auth) where the token
// may need refreshing and extra headers (like ChatGPT-Account-ID) are required.
func WithTokenProvider(tp TokenProvider) Option {
	return func(c *Client) error {
		c.tokenProvider = tp
		return nil
	}
}

// Doer performs a HTTP request.
type Doer interface {
	Do(req *http.Request) (*http.Response, error)
}

// New returns a new OpenAI client.
func New(token string, model string, baseURL string, organization string,
	apiType APIType, apiVersion string, httpClient Doer, embeddingModel string,
	responseFormat *ResponseFormat,
	opts ...Option,
) (*Client, error) {
	c := &Client{
		token:          token,
		Model:          model,
		EmbeddingModel: embeddingModel,
		baseURL:        strings.TrimSuffix(baseURL, "/"),
		organization:   organization,
		apiType:        apiType,
		apiVersion:     apiVersion,
		httpClient:     httpClient,
		ResponseFormat: responseFormat,
	}
	if c.baseURL == "" {
		c.baseURL = defaultBaseURL
	}

	for _, opt := range opts {
		if err := opt(c); err != nil {
			return nil, err
		}
	}

	return c, nil
}

// Completion is a completion.
type Completion struct {
	Text string `json:"text"`
}

// CreateCompletion creates a completion.
func (c *Client) CreateCompletion(ctx context.Context, r *CompletionRequest) (*Completion, error) {
	resp, err := c.createCompletion(ctx, r)
	if err != nil {
		return nil, err
	}
	if len(resp.Choices) == 0 {
		return nil, ErrEmptyResponse
	}
	return &Completion{
		Text: resp.Choices[0].Message.Content,
	}, nil
}

// EmbeddingRequest is a request to create an embedding.
type EmbeddingRequest struct {
	Model      string   `json:"model"`
	Input      []string `json:"input"`
	Dimensions int      `json:"dimensions"`
}

func (c *Client) makeEmbeddingPayload(r *EmbeddingRequest) *embeddingPayload {
	payload := &embeddingPayload{
		Model:      c.EmbeddingModel,
		Dimensions: c.EmbeddingDimensions,
		Input:      r.Input,
	}
	if r.Model != "" {
		payload.Model = r.Model
	}
	if payload.Model == "" {
		payload.Model = defaultEmbeddingModel
	}
	if r.Dimensions > 0 {
		payload.Dimensions = r.Dimensions
	}
	return payload
}

// CreateEmbedding creates embeddings.
func (c *Client) CreateEmbedding(ctx context.Context, r *EmbeddingRequest) ([][]float32, error) {
	if r.Model == "" {
		r.Model = defaultEmbeddingModel
	}

	resp, err := c.createEmbedding(ctx, c.makeEmbeddingPayload(r))
	if err != nil {
		return nil, err
	}

	if len(resp.Data) == 0 {
		return nil, ErrEmptyResponse
	}

	embeddings := make([][]float32, 0)
	for i := 0; i < len(resp.Data); i++ {
		embeddings = append(embeddings, resp.Data[i].Embedding)
	}

	return embeddings, nil
}

// CreateChat creates chat request.
func (c *Client) CreateChat(ctx context.Context, r *ChatRequest) (*ChatCompletionResponse, error) {
	if r.Model == "" {
		if c.Model == "" {
			r.Model = defaultChatModel
		} else {
			r.Model = c.Model
		}
	}
	resp, err := c.createChat(ctx, r)
	if err != nil {
		return nil, err
	}
	if len(resp.Choices) == 0 {
		return nil, ErrEmptyResponse
	}
	return resp, nil
}

func IsAzure(apiType APIType) bool {
	return apiType == APITypeAzure || apiType == APITypeAzureAD
}

func (c *Client) setHeaders(req *http.Request) error {
	req.Header.Set("Content-Type", "application/json")

	token := c.token
	if c.tokenProvider != nil {
		t, extra, err := c.tokenProvider()
		if err != nil {
			return fmt.Errorf("token provider: %w", err)
		}
		token = t
		for k, vs := range extra {
			for _, v := range vs {
				req.Header.Set(k, v)
			}
		}
	}

	if c.apiType == APITypeOpenAI || c.apiType == APITypeAzureAD {
		req.Header.Set("Authorization", "Bearer "+token)
	} else {
		req.Header.Set("api-key", token)
	}
	if c.organization != "" {
		req.Header.Set("OpenAI-Organization", c.organization)
	}
	return nil
}

// OpenResponsesSession opens a persistent WebSocket connection to the
// OpenAI Responses API.
func (c *Client) OpenResponsesSession(ctx context.Context) (*ResponsesSession, error) {
	// Convert REST base URL to WebSocket URL.
	// https://api.openai.com/v1 -> wss://api.openai.com/v1/responses
	wsURL := c.baseURL + "/responses"
	wsURL = strings.Replace(wsURL, "https://", "wss://", 1)
	wsURL = strings.Replace(wsURL, "http://", "ws://", 1)

	token := c.token
	var extraHeaders http.Header
	if c.tokenProvider != nil {
		t, extra, err := c.tokenProvider()
		if err != nil {
			return nil, fmt.Errorf("token provider: %w", err)
		}
		token = t
		extraHeaders = extra
	}
	return DialResponsesSession(ctx, wsURL, token, extraHeaders)
}

func (c *Client) buildURL(suffix string, model string) string {
	if IsAzure(c.apiType) {
		return c.buildAzureURL(suffix, model)
	}

	// open ai implement:
	return fmt.Sprintf("%s%s", c.baseURL, suffix)
}

func (c *Client) buildAzureURL(suffix string, model string) string {
	baseURL := c.baseURL
	baseURL = strings.TrimRight(baseURL, "/")

	// azure example url:
	// /openai/deployments/{model}/chat/completions?api-version={api_version}
	return fmt.Sprintf("%s/openai/deployments/%s%s?api-version=%s",
		baseURL, model, suffix, c.apiVersion,
	)
}

// sanitizeHTTPError sanitizes HTTP client errors to prevent leaking sensitive information.
// It checks for context deadline/cancellation errors and returns generic timeout messages
// instead of potentially exposing request details, headers, or other sensitive data.
func sanitizeHTTPError(err error) error {
	if err == nil {
		return nil
	}

	// Check for context deadline exceeded
	if errors.Is(err, context.DeadlineExceeded) {
		return errors.New("request timeout: API call exceeded deadline")
	}

	// Check for context cancellation
	if errors.Is(err, context.Canceled) {
		return context.Canceled
	}

	// Check for network timeout errors
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return errors.New("request timeout: network operation exceeded timeout")
	}

	// For other network errors, provide generic message without exposing details
	if _, ok := err.(net.Error); ok {
		return errors.New("network error: failed to reach API server")
	}

	// Return original error if it's not a sensitive type
	return err
}
