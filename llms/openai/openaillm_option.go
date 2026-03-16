package openai

import (
	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/llms/openai/chatgptauth"
	"github.com/tmc/langchaingo/llms/openai/internal/openaiclient"
)

const (
	tokenEnvVarName        = "OPENAI_API_KEY"      //nolint:gosec
	modelEnvVarName        = "OPENAI_MODEL"        //nolint:gosec
	baseURLEnvVarName      = "OPENAI_BASE_URL"     //nolint:gosec
	baseAPIBaseEnvVarName  = "OPENAI_API_BASE"     //nolint:gosec
	organizationEnvVarName = "OPENAI_ORGANIZATION" //nolint:gosec
)

type APIType openaiclient.APIType

const (
	APITypeOpenAI  APIType = APIType(openaiclient.APITypeOpenAI)
	APITypeAzure           = APIType(openaiclient.APITypeAzure)
	APITypeAzureAD         = APIType(openaiclient.APITypeAzureAD)
)

const (
	DefaultAPIVersion = "2023-05-15"
)

type options struct {
	token        string
	model        string
	baseURL      string
	organization string
	apiType      APIType
	httpClient   openaiclient.Doer

	responseFormat *ResponseFormat

	// required when APIType is APITypeAzure or APITypeAzureAD
	apiVersion          string
	embeddingModel      string
	embeddingDimensions int

	callbackHandler  callbacks.Handler
	chatGPTAuth      *chatgptauth.Auth // set by WithChatGPTAuth
	chatGPTAuthError error             // deferred error from WithChatGPTAuth
}

// Option is a functional option for the OpenAI client.
type Option func(*options)

// ResponseFormat is the response format for the OpenAI client.
type ResponseFormat = openaiclient.ResponseFormat

// ResponseFormatJSONSchema is the JSON Schema response format in structured output.
type ResponseFormatJSONSchema = openaiclient.ResponseFormatJSONSchema

// ResponseFormatJSONSchemaProperty is the JSON Schema property in structured output.
type ResponseFormatJSONSchemaProperty = openaiclient.ResponseFormatJSONSchemaProperty

// ResponseFormatJSON is the JSON response format.
var ResponseFormatJSON = &ResponseFormat{Type: "json_object"} //nolint:gochecknoglobals

// WithToken passes the OpenAI API token to the client. If not set, the token
// is read from the OPENAI_API_KEY environment variable.
func WithToken(token string) Option {
	return func(opts *options) {
		opts.token = token
	}
}

// WithModel passes the OpenAI model to the client. If not set, the model
// is read from the OPENAI_MODEL environment variable.
// Required when ApiType is Azure.
func WithModel(model string) Option {
	return func(opts *options) {
		opts.model = model
	}
}

// WithEmbeddingModel passes the OpenAI model to the client. Required when ApiType is Azure.
func WithEmbeddingModel(embeddingModel string) Option {
	return func(opts *options) {
		opts.embeddingModel = embeddingModel
	}
}

// WithEmbeddingDimensions passes the OpenAI embeddings dimensions to the client.
// Requires a compatible model, test-embedding-3 or later.
// For more info, please check openai doc
// https://platform.openai.com/docs/api-reference/embeddings/create#embeddings-create-dimensions
func WithEmbeddingDimensions(dimensions int) Option {
	return func(opts *options) {
		opts.embeddingDimensions = dimensions
	}
}

// WithBaseURL passes the OpenAI base url to the client. If not set, the base url
// is read from the OPENAI_BASE_URL environment variable. If still not set in ENV
// VAR OPENAI_BASE_URL, then the default value is https://api.openai.com/v1 is used.
func WithBaseURL(baseURL string) Option {
	return func(opts *options) {
		opts.baseURL = baseURL
	}
}

// WithOrganization passes the OpenAI organization to the client. If not set, the
// organization is read from the OPENAI_ORGANIZATION.
func WithOrganization(organization string) Option {
	return func(opts *options) {
		opts.organization = organization
	}
}

// WithAPIType passes the api type to the client. If not set, the default value
// is APITypeOpenAI.
func WithAPIType(apiType APIType) Option {
	return func(opts *options) {
		opts.apiType = apiType
	}
}

// WithAPIVersion passes the api version to the client. If not set, the default value
// is DefaultAPIVersion.
func WithAPIVersion(apiVersion string) Option {
	return func(opts *options) {
		opts.apiVersion = apiVersion
	}
}

// WithHTTPClient allows setting a custom HTTP client. If not set, the default value
// is http.DefaultClient.
func WithHTTPClient(client openaiclient.Doer) Option {
	return func(opts *options) {
		opts.httpClient = client
	}
}

// WithCallback allows setting a custom Callback Handler.
func WithCallback(callbackHandler callbacks.Handler) Option {
	return func(opts *options) {
		opts.callbackHandler = callbackHandler
	}
}

// WithResponseFormat allows setting a custom response format.
func WithResponseFormat(responseFormat *ResponseFormat) Option {
	return func(opts *options) {
		opts.responseFormat = responseFormat
	}
}

// WithChatGPTAuth configures the client to use ChatGPT OAuth authentication
// instead of an API key. The auth.json file (produced by "npx @openai/codex login")
// is loaded from the given path, or from default locations ($CODEX_HOME/auth.json,
// ~/.codex/auth.json) if path is empty.
//
// This automatically sets the base URL to the ChatGPT backend API and
// configures token refresh. No OPENAI_API_KEY is needed.
func WithChatGPTAuth(authFilePath string, authOpts ...chatgptauth.Option) Option {
	return func(opts *options) {
		auth, err := chatgptauth.Load(authFilePath, authOpts...)
		if err != nil {
			opts.chatGPTAuthError = err
			return
		}
		opts.chatGPTAuth = auth
		if opts.baseURL == "" {
			opts.baseURL = chatgptauth.DefaultBaseURL
		}
		// Set a placeholder token to pass the non-empty check in newClient.
		// The real token comes from the TokenProvider at request time.
		opts.token = "chatgpt-oauth"
	}
}
