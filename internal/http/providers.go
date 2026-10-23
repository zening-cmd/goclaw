package http

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/nextlevelbuilder/goclaw/internal/bus"
	"github.com/nextlevelbuilder/goclaw/internal/config"
	"github.com/nextlevelbuilder/goclaw/internal/i18n"
	"github.com/nextlevelbuilder/goclaw/internal/oauth"
	"github.com/nextlevelbuilder/goclaw/internal/permissions"
	"github.com/nextlevelbuilder/goclaw/internal/providers"
	"github.com/nextlevelbuilder/goclaw/internal/security"
	"github.com/nextlevelbuilder/goclaw/internal/store"
	"github.com/nextlevelbuilder/goclaw/internal/tools"
	usagecaps "github.com/nextlevelbuilder/goclaw/internal/usage/caps"
	"github.com/nextlevelbuilder/goclaw/pkg/protocol"
)

// ProvidersHandler handles LLM provider CRUD endpoints.
type ProvidersHandler struct {
	store           store.ProviderStore
	secretStore     store.ConfigSecretsStore
	providerReg     *providers.Registry
	gatewayAddr     string                           // for injecting MCP bridge into Claude CLI providers
	mcpLookup       providers.MCPServerLookup        // optional: resolves per-agent MCP servers
	shellDenyGroups func() map[string]bool           // optional: current global shell deny-group overrides
	apiBaseFallback func(providerType string) string // optional: config/env fallback for api_base
	cliMu           sync.Mutex                       // serializes Claude CLI provider create to prevent duplicates
	msgBus          *bus.MessageBus
	sysConfigStore  store.SystemConfigStore
	tracingStore    store.TracingStore      // optional: for provider-scoped pool activity
	agents          store.AgentCRUDStore    // optional: for provider pool activity agent lookup
	modelReg        providers.ModelRegistry // optional: forward-compat model resolver for Anthropic
	usageCaps       *usagecaps.Service
}

// NewProvidersHandler creates a handler for provider management endpoints.
func NewProvidersHandler(s store.ProviderStore, secretStore store.ConfigSecretsStore, providerReg *providers.Registry, gatewayAddr string) *ProvidersHandler {
	return &ProvidersHandler{store: s, secretStore: secretStore, providerReg: providerReg, gatewayAddr: gatewayAddr}
}

// SetMessageBus sets the message bus for audit event broadcasting.
// Must be called before serving requests (not thread-safe).
func (h *ProvidersHandler) SetMessageBus(msgBus *bus.MessageBus) {
	h.msgBus = msgBus
}

// SetSystemConfigStore sets the system config store for embedding status checks.
func (h *ProvidersHandler) SetSystemConfigStore(s store.SystemConfigStore) {
	h.sysConfigStore = s
}

// SetMCPServerLookup sets the per-agent MCP server lookup for Claude CLI providers.
// Must be called before serving requests (not thread-safe).
func (h *ProvidersHandler) SetMCPServerLookup(lookup providers.MCPServerLookup) {
	h.mcpLookup = lookup
}

// SetShellDenyGroupsSource sets the current global shell deny-group source for
// runtime provider registration. Must be called before serving requests.
func (h *ProvidersHandler) SetShellDenyGroupsSource(fn func() map[string]bool) {
	h.shellDenyGroups = fn
}

// SetAPIBaseFallback sets a function that returns config/env api_base by provider type.
// Used as fallback when DB providers have no api_base set.
func (h *ProvidersHandler) SetAPIBaseFallback(fn func(providerType string) string) {
	h.apiBaseFallback = fn
}

// SetTracingStore sets the tracing store for provider-scoped pool activity.
func (h *ProvidersHandler) SetTracingStore(ts store.TracingStore) {
	h.tracingStore = ts
}

// SetAgentStore sets the agent store for provider pool activity agent lookup.
func (h *ProvidersHandler) SetAgentStore(as store.AgentCRUDStore) {
	h.agents = as
}

// SetModelRegistry sets the forward-compat model registry used by Anthropic providers
// for model alias resolution and token counting. Must be called before serving requests.
func (h *ProvidersHandler) SetModelRegistry(r providers.ModelRegistry) {
	h.modelReg = r
}

func (h *ProvidersHandler) SetUsageCapService(s *usagecaps.Service) {
	h.usageCaps = s
}

func (h *ProvidersHandler) currentShellDenyPatterns() []*regexp.Regexp {
	if h.shellDenyGroups == nil {
		return tools.DefaultDenyPatterns()
	}
	return tools.ResolveDenyPatterns(h.shellDenyGroups())
}

// resolveAPIBase returns the provider's api_base, falling back to config/env if empty.
// For Ollama/OllamaCloud providers, applies a safety-net normalization: if the stored
// value is missing the /v1 suffix (pre-existing record before write-time normalization),
// the suffix is appended so all downstream call sites receive a ready-to-use URL.
func (h *ProvidersHandler) resolveAPIBase(p *store.LLMProviderData) string {
	base := ""
	if p.APIBase != "" {
		base = p.APIBase
	} else if h.apiBaseFallback != nil {
		base = h.apiBaseFallback(p.ProviderType)
	}
	// Safety net: normalize Ollama URLs missing /v1 (pre-existing DB records).
	if base != "" && (p.ProviderType == store.ProviderOllama || p.ProviderType == store.ProviderOllamaCloud) {
		base = strings.TrimRight(base, "/")
		if !strings.HasSuffix(base, "/v1") {
			base += "/v1"
		}
	}
	return base
}

// emitProviderCacheInvalidate broadcasts a provider cache invalidation event.
// Subscribers (e.g. ACP re-registration in gateway_managed.go) react to reload from DB.
func (h *ProvidersHandler) emitProviderCacheInvalidate(ctx context.Context, tenantID uuid.UUID, name string) bool {
	if h.msgBus == nil {
		return false
	}
	tenantID = providerCacheTenantID(ctx, tenantID)
	h.msgBus.Broadcast(bus.Event{
		Name:     protocol.EventCacheInvalidate,
		TenantID: tenantID,
		Payload:  bus.CacheInvalidatePayload{Kind: bus.CacheKindProvider, Key: name, TenantID: tenantID},
	})
	return true
}

func providerCacheTenantID(ctx context.Context, tenantID uuid.UUID) uuid.UUID {
	if tenantID != uuid.Nil {
		return tenantID
	}
	if tid := store.TenantIDFromContext(ctx); tid != uuid.Nil {
		return tid
	}
	return store.MasterTenantID
}

// RegisterRoutes registers all provider management routes on the given mux.
func (h *ProvidersHandler) RegisterRoutes(mux *http.ServeMux) {
	// Provider CRUD — reads are needed by /setup for browser-paired Operators
	// (issue #1075), so GET routes use read-level auth (GET→Viewer) while
	// mutations stay Admin-only.
	mux.HandleFunc("GET /v1/providers", h.readAuth(h.handleListProviders))
	mux.HandleFunc("POST /v1/providers", h.auth(h.handleCreateProvider))
	mux.HandleFunc("GET /v1/providers/{id}", h.readAuth(h.handleGetProvider))
	mux.HandleFunc("PUT /v1/providers/{id}", h.auth(h.handleUpdateProvider))
	mux.HandleFunc("DELETE /v1/providers/{id}", h.auth(h.handleDeleteProvider))

	// Model listing (proxied to upstream provider API)
	mux.HandleFunc("GET /v1/providers/{id}/models", h.readAuth(h.handleListProviderModels))

	// Provider + model verification (pre-flight check) — mutating actions, Admin.
	mux.HandleFunc("POST /v1/providers/{id}/reconnect", h.auth(h.handleReconnectProvider))
	mux.HandleFunc("POST /v1/providers/{id}/verify", h.auth(h.handleVerifyProvider))
	mux.HandleFunc("POST /v1/providers/{id}/verify-embedding", h.auth(h.handleVerifyEmbedding))

	// Provider-scoped Codex pool activity monitor (read-only status)
	mux.HandleFunc("GET /v1/providers/{id}/codex-pool-activity", h.readAuth(h.handleProviderCodexPoolActivity))

	// Embedding system status (read-only)
	mux.HandleFunc("GET /v1/embedding/status", h.readAuth(h.handleEmbeddingStatus))

	// Claude CLI auth status (global — not per-provider; read-only status)
	mux.HandleFunc("GET /v1/providers/claude-cli/auth-status", h.readAuth(h.handleClaudeCLIAuthStatus))
}

// auth gates provider mutations at Admin.
func (h *ProvidersHandler) auth(next http.HandlerFunc) http.HandlerFunc {
	return requireAuth(permissions.RoleAdmin, next)
}

// readAuth gates read-only provider endpoints at the method-derived minimum
// (GET→Viewer), so browser-paired Operators can complete /setup (issue #1075).
// Responses already mask API keys, and provider queries stay tenant-scoped.
func (h *ProvidersHandler) readAuth(next http.HandlerFunc) http.HandlerFunc {
	return requireAuth("", next)
}

// maskAPIKey replaces non-empty API keys with "***".
func maskAPIKey(p *store.LLMProviderData) {
	if p.APIKey != "" {
		p.APIKey = "***"
	}
}

type providerRuntimeRegistrationStatus string

const (
	providerRuntimeRegistered        providerRuntimeRegistrationStatus = "registered"
	providerRuntimeDisabled          providerRuntimeRegistrationStatus = "disabled"
	providerRuntimeSkipped           providerRuntimeRegistrationStatus = "skipped"
	providerRuntimeMissingCredential providerRuntimeRegistrationStatus = "missing_credential"
	providerRuntimeInvalidConfig     providerRuntimeRegistrationStatus = "invalid_config"
)

// registerInMemory adds (or replaces) a provider in the in-memory registry
// so it's immediately usable for verify/chat without a gateway restart.
func (h *ProvidersHandler) registerInMemory(p *store.LLMProviderData) providerRuntimeRegistrationStatus {
	if h.providerReg == nil || !p.Enabled {
		if p.Enabled {
			return providerRuntimeSkipped
		}
		return providerRuntimeDisabled
	}
	// ACP agents don't need an API key — skip in-memory registration
	// (ACP providers are registered via gateway_providers.go on startup or restart)
	if p.ProviderType == store.ProviderACP {
		return providerRuntimeSkipped
	}
	// Claude CLI doesn't need an API key — register immediately
	if p.ProviderType == store.ProviderClaudeCLI {
		cliPath := p.APIBase // reuse APIBase field for CLI path
		if cliPath == "" {
			cliPath = "claude"
		}
		// Validate: only accept "claude" or absolute path (mirrors startup path in cmd/gateway_providers.go).
		// Prevents DB-poisoning attacks where a relative path resolves against CWD.
		if cliPath != "claude" && !filepath.IsAbs(cliPath) {
			slog.Warn("security.claude_cli: invalid path, using default", "path", cliPath, "provider", p.Name)
			cliPath = "claude"
		}
		if _, err := exec.LookPath(cliPath); err != nil {
			slog.Warn("claude-cli: binary not found, skipping in-memory registration", "path", cliPath, "provider", p.Name, "error", err)
			return providerRuntimeInvalidConfig
		}
		cliOpts := []providers.ClaudeCLIOption{
			providers.WithClaudeCLIName(p.Name),
			providers.WithClaudeCLISecurityHooks("", true, h.currentShellDenyPatterns()),
		}
		if h.gatewayAddr != "" {
			mcpData := providers.BuildCLIMCPConfigData(nil, h.gatewayAddr, pkgGatewayToken)
			mcpData.AgentMCPLookup = h.mcpLookup
			cliOpts = append(cliOpts, providers.WithClaudeCLIMCPConfigData(mcpData))
		}
		h.providerReg.RegisterForTenant(p.TenantID, providers.NewClaudeCLIProvider(cliPath, cliOpts...))
		return providerRuntimeRegistered
	}
	// Ollama doesn't need an API key — handle before the key guard (same as startup).
	// In Docker, swap localhost → host.docker.internal so the container can reach the host.
	if p.ProviderType == store.ProviderOllama {
		host := p.APIBase
		if host == "" {
			host = "http://localhost:11434"
		}
		dockerHost := config.DockerLocalhost(host)
		numCtx := h.resolveOllamaNumCtx(p, dockerHost, "")
		prov := providers.NewOllamaProvider(p.Name, dockerHost, "llama3.3", numCtx, nil).
			WithThinkingEnabled(store.ParseThinkingEnabled(p.Settings))
		h.providerReg.RegisterForTenant(p.TenantID, prov)
		return providerRuntimeRegistered
	}
	// Vertex supports ADC (empty api_key) — handle before the generic key guard.
	if p.ProviderType == store.ProviderVertex {
		vsettings := store.ParseVertexProviderSettings(p.Settings)
		if vsettings == nil {
			slog.Warn("vertex: missing project_id/region in settings, cannot register", "name", p.Name)
			return providerRuntimeInvalidConfig
		}
		vcfg := providers.VertexConfig{
			Name:            p.Name,
			CredentialsJSON: p.APIKey,
			ProjectID:       vsettings.ProjectID,
			Region:          vsettings.Region,
			DefaultModel:    vsettings.Model,
			APIBaseOverride: p.APIBase,
		}
		prov, err := providers.NewVertexProviderWithTimeout(vcfg)
		if err != nil {
			slog.Warn("vertex: register in-memory failed", "name", p.Name, "error", err)
			return providerRuntimeInvalidConfig
		}
		h.providerReg.RegisterForTenant(p.TenantID, prov)
		return providerRuntimeRegistered
	}
	if p.APIKey == "" {
		return providerRuntimeMissingCredential
	}
	apiBase := h.resolveAPIBase(p)
	switch p.ProviderType {
	case store.ProviderAPIRoute:
		if apiBase == "" {
			apiBase = store.APIRouteDefaultAPIBase
		}
		h.providerReg.RegisterForTenant(p.TenantID, providers.NewOpenAIProvider(p.Name, p.APIKey, apiBase, store.APIRouteDefaultModel).WithProviderType(p.ProviderType))
	case store.ProviderChatGPTOAuth:
		ts := oauth.NewDBTokenSource(h.store, h.secretStore, p.Name).WithTenantID(p.TenantID)
		codex := providers.NewCodexProvider(p.Name, ts, apiBase, "")
		if oauthSettings := store.ParseChatGPTOAuthProviderSettings(p.Settings); oauthSettings != nil {
			codex.WithRoutingDefaults(oauthSettings.CodexPool.Strategy, oauthSettings.CodexPool.ExtraProviderNames)
		}
		h.providerReg.RegisterForTenant(p.TenantID, codex)
	case store.ProviderAnthropicNative:
		anthOpts := []providers.AnthropicOption{
			providers.WithAnthropicName(p.Name),
			providers.WithAnthropicBaseURL(apiBase),
		}
		if h.modelReg != nil {
			anthOpts = append(anthOpts, providers.WithAnthropicRegistry(h.modelReg))
		}
		h.providerReg.RegisterForTenant(p.TenantID, providers.NewAnthropicProvider(p.APIKey, anthOpts...))
	case store.ProviderDashScope:
		h.providerReg.RegisterForTenant(p.TenantID, providers.NewDashScopeProvider(p.Name, p.APIKey, apiBase, ""))
	case store.ProviderBailian:
		base := apiBase
		if base == "" {
			base = "https://coding-intl.dashscope.aliyuncs.com/v1"
		}
		h.providerReg.RegisterForTenant(p.TenantID, providers.NewOpenAIProvider(p.Name, p.APIKey, base, "qwen3.5-plus").
			WithProviderType(p.ProviderType))
	case store.ProviderZai:
		base := apiBase
		if base == "" {
			base = store.ZaiDefaultAPIBase
		}
		h.providerReg.RegisterForTenant(p.TenantID, providers.NewOpenAIProvider(p.Name, p.APIKey, base, store.ZaiDefaultModel))
	case store.ProviderZaiCoding:
		base := apiBase
		if base == "" {
			base = store.ZaiCodingDefaultAPIBase
		}
		h.providerReg.RegisterForTenant(p.TenantID, providers.NewOpenAIProvider(p.Name, p.APIKey, base, store.ZaiDefaultModel))
	case store.ProviderNovita:
		base := apiBase
		if base == "" {
			base = store.NovitaDefaultAPIBase
		}
		h.providerReg.RegisterForTenant(p.TenantID, providers.NewOpenAIProvider(p.Name, p.APIKey, base, store.NovitaDefaultModel))
	case store.ProviderKimiCoding:
		// Moonshot Kimi Coding requires a fixed User-Agent on every request.
		base := apiBase
		if base == "" {
			base = store.KimiCodingDefaultAPIBase
		}
		prov := providers.NewOpenAIProvider(p.Name, p.APIKey, base, store.KimiCodingDefaultModel)
		prov.WithProviderType(p.ProviderType)
		prov.WithExtraHeaders(map[string]string{
			"User-Agent": store.KimiCodingRequiredUserAgent,
		})
		h.providerReg.RegisterForTenant(p.TenantID, prov)
	case store.ProviderAIMLAPI:
		prov := providers.NewAIMLAPIProvider(p.Name, p.APIKey, apiBase)
		prov.WithProviderType(p.ProviderType)
		h.providerReg.RegisterForTenant(p.TenantID, prov)
	case store.ProviderOllamaCloud:
		base := apiBase
		if base == "" {
			base = "https://ollama.com"
		}
		numCtx := h.resolveOllamaNumCtx(p, base, p.APIKey)
		prov := providers.NewOllamaProvider(p.Name, base, "llama3.3", numCtx, nil).
			WithThinkingEnabled(store.ParseThinkingEnabled(p.Settings))
		h.providerReg.RegisterForTenant(p.TenantID, prov)
	default:
		base, model := openAIProviderDefaults(p.ProviderType, apiBase)
		prov := providers.NewOpenAIProvider(p.Name, p.APIKey, base, model)
		prov.WithThinkingEnabled(store.ParseThinkingEnabled(p.Settings))
		h.providerReg.RegisterForTenant(p.TenantID, prov)
	}
	return providerRuntimeRegistered
}

// resolveOllamaNumCtx returns the num_ctx to pass to NewOllamaProvider. Priority:
//  1. User-configured num_ctx from provider settings JSONB.
//  2. Value queried from Ollama /api/show.
//  3. OllamaDefaultNumCtx (131072) — never nil, so Ollama always uses a large context window.
func (h *ProvidersHandler) resolveOllamaNumCtx(p *store.LLMProviderData, apiBase, apiKey string) *int {
	slog.Debug("ollama.startup: resolveOllamaNumCtx called", "provider", p.Name, "api_base", apiBase)
	if s := store.ParseOllamaSettings(p.Settings); s != nil {
		slog.Info("ollama.startup: using num_ctx from provider settings", "provider", p.Name, "num_ctx", *s.NumCtx)
		return s.NumCtx
	}
	slog.Debug("ollama.startup: no settings num_ctx, querying /api/show", "provider", p.Name)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	numCtx := providers.FetchOllamaModelContext(ctx, apiBase, "llama3.3", apiKey)
	slog.Info("ollama.startup: applying num_ctx from /api/show (or default fallback)", "provider", p.Name, "num_ctx", numCtx)
	return &numCtx
}

func openAIProviderDefaults(providerType, apiBase string) (string, string) {
	switch providerType {
	case store.ProviderMiniMax:
		if apiBase == "" {
			apiBase = store.MiniMaxDefaultAPIBase
		}
		return apiBase, store.MiniMaxDefaultModel
	case store.ProviderAtlasCloud:
		if apiBase == "" {
			apiBase = store.AtlasCloudDefaultAPIBase
		}
		return apiBase, store.AtlasCloudDefaultModel
	case store.ProviderRequesty:
		if apiBase == "" {
			apiBase = store.RequestyDefaultAPIBase
		}
		return apiBase, store.RequestyDefaultModel
	default:
		return apiBase, ""
	}
}

// normalizeOllamaAPIBase normalizes the api_base stored for Ollama providers.
// The /v1 suffix is added at write time for backward compatibility; NewOllamaProvider
// strips it automatically before constructing the native Ollama client URL.
func normalizeOllamaAPIBase(p *store.LLMProviderData) {
	if p.ProviderType != store.ProviderOllama && p.ProviderType != store.ProviderOllamaCloud {
		return
	}
	if p.APIBase == "" {
		return
	}
	p.APIBase = strings.TrimRight(p.APIBase, "/")
	if !strings.HasSuffix(p.APIBase, "/v1") {
		p.APIBase += "/v1"
	}
}

// localURLProviderTypes are provider types that legitimately run on localhost.
// They are restricted to an explicit localhost allowlist
// rather than skipping SSRF validation entirely.
// ACP is intentionally excluded: its api_base carries an executable command/path,
// not a URL (see issue #1481).
var localURLProviderTypes = map[string]bool{
	store.ProviderOllama: true,
}

// allowedLocalHosts are the only hosts permitted for local provider types.
// Explicit allowlist (not blocklist) to prevent new internal addresses from
// slipping through (e.g. 169.254.169.254 via ollama base URL).
var allowedLocalHosts = []stri�^���춻�q�^t�ѡ�����Ʌѽȵ�������ɕ�����Ʉ�����́��ɵ��ѕ�(�����ȁ�������ɽ٥��ȁ����̀��������������٥��=
1]}=115}11=]}!=MQL(����������͕��Ʌѕ�����ѹ���̽%A̰���������ȸ���̸�ı��������������Q��͔��ɔ(������������ѽ���������ݕ�1����!���̰�����ݥ���18����ѕ��=������͕�ٕ�̸(���م�Յѕ��������Ё����Ё�����ͼ�ѕ��́�����ٕ�ɥ���ѡ��مɥ���������ɔ�ѡ�Ё������̸)مȁ���������ݕ�!��������幌�=���Y��Ք��չ����mu��ɥ����(%Ʌ܀��̹�ѕ�ؠ�=
1]}=115}11=]}!=MQL��(%���Ʌ܀�􀈈��($%ɕ��ɸ����(%�(%����̀����ɥ��̹M���СɅܰ�����(%����̀�􁵅���mu��ɥ���������������̤�(%��ȁ|������Ʌ��������́�($%�����ɥ��̹Qɥ�M�������($%������􀈈��($$%����̀�����������̰���($%�(%�(%ɕ��ɸ������)��((���م����ѕAɽ٥���UI0�ɕ����́�ɽ٥��ȁ��͔�UI1́����ѥ���Ѽ���ѕɹ����ɥمє����ݽɭ̸(�������͔�������Ѡ��ɕٕ��́MMI�ݡ����ɽ٥���́�ɔ���ѕȁ�͕����ȁA$�����̸(��(���1�����(����ĸ�����UI0��H�����ݕ����ɽ٥��ȁ��䁹�Ё����������ѽ����͔��(����ȸ�
��Ց��
1$��H����}��͔��́����ᕍ�х������Ѡ������������Ё��UI0�(����̸�M����������������������́���䤃�H�����ɍ�����ȁUI0���͕������̰�����Ց���(�������������UI0�����̸�	����́����輼��������輼������輼���ь�(����и�1�����UI0�����̀���������������H����Ё���Ё����������ݕ�1����!���̰��ȁ��(�������ѡ�����Ʌѽȵ�������ɕ��=
1]}=115}11=]}!=MQL����Ѐ���������(�����������ݱ��Ё�ɕٕ��́ɕ������������и�����Ё�ȁ��ѕɹ���͕�٥��́٥��ѡ�(�����������������������̰�ݡ�����ѥ�������ݥ���18����ѕ��=������͕�ٕ�̤�(����Ը�I���є�����̃�H����=
1]}11=]}AI%YQ}AI=Y%I}UI1L��͕́а�����܁��������(�������=ѡ��ݥ͔�ɕͽ�ٔ�9L����ѹ����ɕ���Ё���9d�ɕͽ�ٕ��%@�ͅѥ͙���(�������͕��ɥ��%�	����������ٕ�́�����������������������ɥمє���ձѥ���а(�������չ�����������P�����Ց������������������ѡ�Ё��ɱ��ȁ�����ɽ����������́���͕���(��(���9L�ɕͽ��ѥ�������ѕ��ԁ���͕́ѡ������������ͱ����������х���ȵ�������������(���ݡ�ɔ������ѹ�������͕́����ѕɅ����ɥ�����������Ё��Ёɕͽ�ٕ́Ѽ����ɥمє�%@�)�չ��م����ѕAɽ٥���UI0�Ʌ�UI0���ɥ�����ɽ٥���Q������ɥ������ɽȁ�(%���Ʌ�UI0��􀈈��($%ɕ��ɸ����(%�(%����ɽ٥���Q�������ѽɔ�Aɽ٥���
��Ց�
1$��($%ɕ��ɸ�م����ѕ
��Ց�
1%ᕍ�х���A�Ѡ�Ʌ�UI0�(%�(%����ɽ٥���Q�������ѽɔ�Aɽ٥���
@��($%ɕ��ɸ�م����ѕ
Aᕍ�х���A�Ѡ�Ʌ�UI0�(%�(%԰���Ȁ���ɰ�A��͔�Ʌ�UI0�(%�����Ȁ�􁹥���($%ɕ��ɸ���й�ɽɘ����م����UI0耕܈����Ȥ(%�($���M�������������́չ�����ѥ�������ȁUI0���͕���ɽ٥��ȁ����̰�����Ց����������UI0�����̸(%�ݥэ��ԹM�������(%��͔��������������̈�(%����ձ��($%ɕ��ɸ���й�ɽɘ���ɽ٥��ȁUI0����Ё�͔�������ȁ����͍́��������Ѐ�Ĉ��ԹM������(%�((%���Ѐ��Թ!��ѹ�����(($���1������ɽ٥��ȁ�����聽��䁅���܁����������Ё��������Ё����ݱ��и($���Q��́�ɕٕ��́�ͥ���ѡ��������������͍������э��Ѽ�ɕ������ѕɹ���͕�٥���($����ȁ���Ր���х��ф���������̸(%��������UI1Aɽ٥���Q����m�ɽ٥���Q���t��($%��ȁ|������Ʌ��������ݕ�1����!���́�($$%�����ɥ��̹�Յ��������а�����($$$%ɕ��ɸ����($$%�($%�($%��ȁ|������Ʌ�������������ݕ�!���������($$%�����ɥ��̹�Յ��������а�����($$$%ɕ��ɸ����($$%�($%�($%ͱ���]�ɸ��͕��ɥ���ɽ٥���}�ɰ������}����}�������������Ј�����а���ɽ٥���}��������ɽ٥���Q����($%ɕ��ɸ���й�ɽɘ���ɽ٥��ȁ������ā���䁅����́��������ЁUI1̀���������а���ܸ����İ���İ����й�����ȹ��ѕɹ�����ȁ=
1]}=115}11=]}!=MQL�����Ё���Ѐ�Ĉ���ɽ٥���Q��������Ф(%�(($���=��Ʌѽȁ��е���Ѽ�����܁�ɥمє����ݽɬ��ɽ٥��ȁUI1̀������18����ѕ���114��($���M���������������ٔ��ѥ���������́�ٕ��ݥѠ�ѡ�́��є������(%��������AɥمѕAɽ٥���UI1������($%ͱ���]�ɸ��͕��ɥ���ɽ٥���}�ɰ��ɥمѕ}����ݕ��������Ј�����а���ɽ٥���}��������ɽ٥���Q����($%ɕ��ɸ����(%�(($���
�������ѕɅ��%@�����Ѐ��ٽ��́չ�����ͅ��9L���������(%�������􁹕йA��͕%@����Ф쁥���􁹥���($%���͕��ɥ��%�	������������($$%ͱ���]�ɸ��͕��ɥ���ɽ٥���}�ɰ���������������Ј�����а���ɽ٥���}��������ɽ٥���Q����($$%ɕ��ɸ���й�ɽɘ���ɽ٥��ȁUI0������Ё����ЁѼ��̈�����Ф($%�($%ɕ��ɸ����(%�(($���	��������ѕɹ�������������ՙ��������ɔ�9L����������Ё��ȁݕ������ݸ����ѕɹ̤�(%�����ɥ��̹!��Mՙ��ࡡ��а�����ѕɹ����������ɥ��̹!��Mՙ��ࡡ��а������������($%ͱ���]�ɸ��͕��ɥ���ɽ٥���}�ɰ���������������Ј�����а���ɽ٥���}��������ɽ٥���Q����($%ɕ��ɸ���й�ɽɘ���ɽ٥��ȁUI0������Ё����ЁѼ���ѕɹ������ѹ���耕̈�����Ф(%�(($���I�ͽ�ٔ�9L������������ٕ��ɕ��ɹ������ɕ�̸($���Aɕٕ��́�����́٥��ݥ����ɐ�͕�٥��̀����������ͱ��������ȁ��х���ȵ����ɽ����($���������́ѡ�Ё����Ѽ��ɥمє�%À�9L�ɕ���������Ё�������ѥ����(%����̰���Ȁ�􁑹�I�ͽ�ٕ������Ф(%�����Ȁ�􁹥���($%ͱ���]�ɸ��͕��ɥ���ɽ٥���}�ɰ����}ɕͽ�ٕ}�������������Ј�����а���ɽ٥���}��������ɽ٥���Q��������ɽȈ����Ȥ($%ɕ��ɸ���й�ɽɘ���ɽ٥��ȁUI0����ѹ�����ā��ձ����Ё���ɕͽ�ٕ�耕܈�����а���Ȥ(%�(%��ȁ|�����Ȁ��Ʌ��������́�($%����􁹕йA��͕%@����Ȥ($%�������􁹥���($$%���ѥ�Ք($%�($%���͕��ɥ��%�	������������($$%ͱ���]�ɸ��͕��ɥ���ɽ٥���}�ɰ��������}ɕͽ�ٕ��������Ј�����а��ɕͽ�ٕ�}��������M�ɥ��������ɽ٥���}��������ɽ٥���Q����($$%ɕ��ɸ���й�ɽɘ���ɽ٥��ȁUI0��āɕͽ�ٕ́Ѽ��ɥمє�ɕ͕�ٕ�����ɕ�̀�̈�����а����($%�(%�(%ɕ��ɸ����)�()�չ��م����ѕ
��Ց�
1%ᕍ�х���A�Ѡ���Ѡ���ɥ������ɽȁ�(%�����ɥ��̹
��х��̡��Ѡ���q�������($%ɕ��ɸ���й�ɽɘ��
��Ց��
1$��ᕍ�х������Ѡ������Ё���х���9U0���є��(%�(%���|����Ȁ���ɰ�A��͕I��Օ��UI$���Ѡ�쁕�Ȁ�􁹥�������ɥ��̹
��х��̡��Ѡ���輼����($%ɕ��ɸ���й�ɽɘ��
��Ց��
1$����}��͔����Ё�������ᕍ�х������Ѡ��Ȁ�İ���ЁUI0��Ĉ������Ց������Ѡ�(%�(%�����Ѡ��􀉍��Ց������������Ѡ�%��̡��Ѡ���($%ɕ��ɸ����(%�(%ɕ��ɸ���й�ɽɘ��
��Ց��
1$����}��͔����Ё����ā�ȁ�����ͽ��є��ᕍ�х������Ѡ����Ѐ�Ĉ������Ց������Ѡ�)�((���م����ѕ
Aᕍ�х���A�Ѡ�م����ѕ́ѡ�Ё���}��͔���ȁ
@��ɽ٥���́���ɥ�́�(������������ȁ��ͽ��є��ᕍ�х������Ѡ����Ё��UI0��Q��́���ɽ�́ѡ���չѥ��(���ɕ����Ʌѥ������������������ѕ݅�}�ɽ٥���̹���)�չ��م����ѕ
Aᕍ�х���A�Ѡ���Ѡ���ɥ������ɽȁ�(%�����ɥ��̹
��х��̡��Ѡ���q�������($%ɕ��ɸ���й�ɽɘ��
@���������Ѡ������Ё���х���9U0���є��(%�(%���|����Ȁ���ɰ�A��͕I��Օ��UI$���Ѡ�쁕�Ȁ�􁹥�������ɥ��̹
��х��̡��Ѡ���輼����($%ɕ��ɸ���й�ɽɘ��
@����}��͔����Ё�������ᕍ�х������Ѡ��ȁ�����������ЁUI0��Ĉ����Ѡ�(%�($���-������ɥ��ݥѠ�ɕ���ѕ�
Aɽ�聉ե�е���������������́�ȁ��ͽ��є���ѡ̸(%�����Ѡ��􀉍��Ց��������Ѡ��􀉍�����������Ѡ��􀉝����������������Ѡ�%��̡��Ѡ���($%ɕ��ɸ����(%�(%ɕ��ɸ���й�ɽɘ��
@����}��͔����Ё����İ��İ��İ��ȁ�����ͽ��є��ᕍ�х������Ѡ����Ѐ�Ĉ������Ց�������������������������Ѡ�)�((�������Aɽ٥��ȁ
IU����()�չ������Aɽ٥����!�����Ȥ�������1���Aɽ٥���̡܁�����I�����͕]ɥѕȰ�Ȁ������I��Օ�Ф��(%�ɽ٥���̰���Ȁ�􁠹�ѽɔ�1���Aɽ٥���̡ȹ
��ѕ�Р��(%�����Ȁ�􁹥���($%ͱ����ɽȠ��ɽ٥���̹���Ј�����ɽȈ����Ȥ($%�����������Ʌ��1������Ȥ($%�ɥѕ)M=8�ܰ������Mх���%�ѕɹ��M��ٕ��ɽȰ����m��ɥ��u��ɥ��쉕�ɽȈ聤�ḹP�����������ḹ5͝�����Q�1��а���ɽ٥���̈���($%ɕ��ɸ(%�((%��ȁ����Ʌ�����ɽ٥���́�($%��ͭA%-�䠙�ɽ٥����m�t�(%�((%�Չ���Aɽ٥���̀�􁵅���mu�ѽɔ�115Aɽ٥����ф����������ɽ٥���̤�(%��ȁ����Ʌ�����ɽ٥���́�($%�Չ���Aɽ٥���̀��������Չ���Aɽ٥���̰������������Aɽ٥�����I�����͔���ɽ٥����m�t��(%�(%�ɥѕ)M=8�ܰ������Mх���=,�����m��ɥ��u�����ɽ٥���̈��Չ���Aɽ٥������)�()�չ������Aɽ٥����!�����Ȥ�������
ɕ�ѕAɽ٥��ȡ܁�����I�����͕]ɥѕȰ�Ȁ������I��Օ�Ф��(%�����������Ʌ��1������Ȥ(%مȁ���ѽɔ�115Aɽ٥����ф(%�����Ȁ��ͽ��9�������ȡ�����5��	�ѕ�I����ȡܰ�ȹ	��䰀�����������������쁕�Ȁ�􁹥���($%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聤�ḹP�����������ḹ5͝%�م���)M=8���($%ɕ��ɸ(%�((%�����9�����􀈈��($%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聤�ḹP�����������ḹ5͝I��եɕ������������($%ɕ��ɸ(%�(%������Y����M�՜���9������($%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聤�ḹP�����������ḹ5͝%�م���M�՜�����������($%ɕ��ɸ(%�(%��Aɽ٥���Q������ѽɔ�9�ɵ����Aɽ٥���Q������Aɽ٥���Q����(%�����ѽɔ�Y����Aɽ٥���Q����m��Aɽ٥���Q���t��($%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聤�ḹP�����������ḹ5͝%�م���I��Օ�а��չ������ѕ���ɽ٥���}��������($%ɕ��ɸ(%�(($���=��䁽���
��Ց��
1$��ɽ٥��ȁ�́����ݕ����ȁ���х�����ā���������ā��Ѡ�͕�ͥ����($���5�ѕ��͕ɥ����́�������ɕ�є�Ѽ��ɕٕ�ЁQ=
Q=T�Ʌ���(%�����Aɽ٥���Q�������ѽɔ�Aɽ٥���
��Ց�
1$��($%�����5Թ1�����($%����ȁ�����5ԹU�������(($%���ѥ����|��􁠹�ѽɔ�1���Aɽ٥���̡ȹ
��ѕ�Р��($%��ȁ|�������Ʌ�������ѥ����($$%������Aɽ٥���Q�������ѽɔ�Aɽ٥���
��Ց�
1$��($$$%�ɥѕ)M=8�ܰ������Mх���
������а����m��ɥ��u��ɥ���($$$$$���ɽȈ聤�ḹP�����������ḹ5͝�ɕ������̰��
��Ց��
1$��ɽ٥��Ȉ������䁽����́����ݕ����ȁ���х������($$$%��($$$%ɕ��ɸ($$%�($%�(%�((%�����Ȁ��م����ѕ
���AQ=�ѡAɽ٥���
������є�ȹ
��ѕ�Р������ѽɔ���ե��9�������쁕�Ȁ�􁹥���($%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聕�ȹ�ɽȠ���($%ɕ��ɸ(%�(%�����Ȁ��م����ѕAɽ٥�����������M��ѥ��̠���쁕�Ȁ�􁹥���($%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聤�ḹP�����������ḹ5͝%�م���I��Օ�а���ȹ�ɽȠ����($%ɕ��ɸ(%�((%�����Ȁ��م����ѕAɽ٥���UI0���A%	�͔����Aɽ٥���Q����쁕�Ȁ�􁹥���($%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聕�ȹ�ɽȠ���($%ɕ��ɸ(%�(($���9�ɵ���锁=��������͔�UI0�Ѽ�����Ց����āͼ������������ѡ�($�������а����������ѥ��������������ٕɥ�䤁�͔�ѡ��ͅ���م�Ք��ɽ���(%��ɵ����=�����A%	�͔����((%�����Ȁ�􁠹�ѽɔ�
ɕ�ѕAɽ٥��ȡȹ
��ѕ�Р������쁕�Ȁ�􁹥���($%ͱ����ɽȠ��ɽ٥���̹�ɕ�є������ɽȈ����Ȥ($%�ɥѕ)M=8�ܰ������Mх���%�ѕɹ��M��ٕ��ɽȰ����m��ɥ��u��ɥ��쉕�ɽȈ聕�ȹ�ɽȠ���($%ɕ��ɸ(%�(($���I����ѕȁ���������ͼ�ٕɥ�位��Ёݽɬ�ݥѡ��Ёɕ�х��(%��ɕ���ѕ�%�5����䠙��(%������Aɽ٥���
����%�م����є�ȹ
��ѕ�Р�����Q�����%����9����((%����Ց�С���͝	�̰�Ȱ���ɽ٥��ȹ�ɕ�ѕ������ɽ٥��Ȉ����%�M�ɥ�����(%��ͭA%-�䠙��(%�Չ���Aɽ٥��Ȁ�􁍅���������Aɽ٥�����I�����͔����(%�ɥѕ)M=8�ܰ������Mх���
ɕ�ѕ����Չ���Aɽ٥��Ȥ)�()�չ������Aɽ٥����!�����Ȥ���������Aɽ٥��ȡ܁�����I�����͕]ɥѕȰ�Ȁ������I��Օ�Ф��(%�����������Ʌ��1������Ȥ(%������Ȁ���ե��A��͔�ȹA�ѡY��Ք�������(%�����Ȁ�􁹥���($%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聤�ḹP�����������ḹ5͝%�م���%����ɽ٥��Ȉ���($%ɕ��ɸ(%�((%�����Ȁ�􁠹�ѽɔ���Aɽ٥��ȡȹ
��ѕ�Р������(%�����Ȁ�􁹥���($%�ɥѕ)M=8�ܰ������Mх���9���չ������m��ɥ��u��ɥ��쉕�ɽȈ聤�ḹP�����������ḹ5͝9���չ�����ɽ٥��Ȉ�����M�ɥ�������($%ɕ��ɸ(%�((%��ͭA%-����(%�Չ���Aɽ٥��Ȁ�􁍅���������Aɽ٥�����I�����͔���(%�ɥѕ)M=8�ܰ������Mх���=,���Չ���Aɽ٥��Ȥ)�()�չ������Aɽ٥����!�����Ȥ�������I��������Aɽ٥��ȡ܁�����I�����͕]ɥѕȰ�Ȁ������I��Օ�Ф��(%�����������Ʌ��1������Ȥ(%������Ȁ���ե��A��͔�ȹA�ѡY��Ք�������(%�����Ȁ�􁹥���($%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聤�ḹP�����������ḹ5͝%�م���%����ɽ٥��Ȉ���($%ɕ��ɸ(%�((%مȁɕā���ՍЁ�($%Y�ɥ�䁉������ͽ��ٕɥ�䉀(%�(%���ȹ	��䀄􁹥�����ȹ	��䀄�����9�	��䀘��ȹ
��ѕ��1���Ѡ������($%�����Ȁ��ͽ��9�������ȡ�����5��	�ѕ�I����ȡܰ�ȹ	��䰀���������������ɕĤ쁕�Ȁ�􁹥���($$%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聤�ḹP�����������ḹ5͝%�م���)M=8���($$%ɕ��ɸ($%�($%���ɕĹY�ɥ���($$%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聤�ḹP�����������ḹ5͝%�م���I��Օ�а��ٕɥ�䁥́��Ё������ѕ����ɕ�������쁍�����ɽ٥��ȁٕɥ��͕��Ʌѕ�䈥��($$%ɕ��ɸ($%�(%�((%�����Ȁ�􁠹�ѽɔ���Aɽ٥��ȡȹ
��ѕ�Р������(%�����Ȁ�􁹥���($%�ɥѕ)M=8�ܰ������Mх���9���չ������m��ɥ��u��ɥ��쉕�ɽȈ聤�ḹP�����������ḹ5͝9���չ�����ɽ٥��Ȉ�����M�ɥ�������($%ɕ��ɸ(%�((%������ɽ٥���I����􁹥���($%���ɽ٥���I���U�ɕ���ѕ���Q����С��Q�����%����9����(%�((%�х��̀�􀉑�ͅ�����(%ɕ������U���ѕ���􁙅�͔(%�������������($%�ݥэ����ɕ���ѕ�%�5���������($%��͔��ɽ٥���Iչѥ��I����ѕɕ��($$%�х��̀�ɕ������ѕ��($$%ɕ������U���ѕ�����Ք($%����ձ��($$%�х��̀􀉹��}ɕ���ѕɕ��($%�(%�(%�����%�م����ѕ���􁠹����Aɽ٥���
����%�م����є�ȹ
��ѕ�Р�����Q�����%����9����(%����Ց�С���͝	�̰�Ȱ���ɽ٥��ȹɕ������ѕ������ɽ٥��Ȉ����%�M�ɥ�����((%��ͭA%-����(%�Չ���Aɽ٥��Ȁ�􁍅���������Aɽ٥�����I�����͔���(%�ɥѕ)M=8�ܰ������Mх���=,�����m��ɥ��u����($$��х��̈耀�����������х��̰($$��ɽ٥��Ȉ耀���������Չ���Aɽ٥��Ȱ($$�ɕ������}����ѕ��老ɕ������U���ѕ��($$������}��م����ѕ��聍����%�م����ѕ��(%��)�()�չ������Aɽ٥����!�����Ȥ�������U���ѕAɽ٥��ȡ܁�����I�����͕]ɥѕȰ�Ȁ������I��Օ�Ф��(%�����������Ʌ��1������Ȥ(%������Ȁ���ե��A��͔�ȹA�ѡY��Ք�������(%�����Ȁ�􁹥���($%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聤�ḹP�����������ḹ5͝%�م���%����ɽ٥��Ȉ���($%ɕ��ɸ(%�((%مȁ����ѕ́���m��ɥ��u���(%�����Ȁ��ͽ��9�������ȡ�����5��	�ѕ�I����ȡܰ�ȹ	��䰀�������������������ѕ̤쁕�Ȁ�􁹥���($%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聤�ḹP�����������ḹ5͝%�م���)M=8���($%ɕ��ɸ(%�(($���Y�����є�������������������ѕ�(%������������������ѕ�l������t쁽���($%���̰�|��􁹅������ɥ���쀅��Y����M�՜�̤��($$%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聤�ḹP�����������ḹ5͝%�م���M�՜�����������($$%ɕ��ɸ($%�(%�(($���M�ɥ����ͭ���A$���䃊P�����Ё�ٕ��ɥє�ɕ���م�Ք�ݥѠ������(%������-�䰁���������ѕ�l����}���t쁽���($%���̰�|�����-�両��ɥ����̀�􀈨�������̀�􀈈��($$%����є�����ѕ̰�����}��䈤($%�(%�(($������ݱ���聽�����ɵ�Ё���ݸ��ɽ٥��ȁ���յ�̸(%����ѕ̀􁙥�ѕ����ݕ�-��̡����ѕ̰��ɽ٥������ݕ�����̤((%���ɕ��Aɽ٥��Ȱ���Ȁ�􁠹�ѽɔ���Aɽ٥��ȡȹ
��ѕ�Р������(%�����Ȁ�􁹥���($%�ɥѕ)M=8�ܰ������Mх���9���չ������m��ɥ��u��ɥ��쉕�ɽȈ聤�ḹP�����������ḹ5͝9���չ�����ɽ٥��Ȉ�����M�ɥ�������($%ɕ��ɸ(%�((%�������є��􀩍��ɕ��Aɽ٥���(%������������������ѕ�l������t����ɥ���쁽�����������􀈈��($%�������є�9����􁹅��(%�(%������-�䰁���������ѕ�l����}���t����ɥ���쁽���($%�������є�A%-�����-��(%�(%������	�͔�����������ѕ�l����}��͔�t����ɥ���쁽���($%�������є�A%	�͔����	�͔(%�(%���������������������ѕ�l���������t�������쁽���($%�������є��������􁕹�����(%�(%����������9��������������ѕ�l��������}�����t����ɥ���쁽���($%�������є�������9����􁑥�����9���(%�(%����а����������ѕ�l��ɽ٥���}�����t����ɥ���쁽������Ѐ�􀈈��($%�Ѐ��ѽɔ�9�ɵ����Aɽ٥���Q�����Ф($%����ѕ�l��ɽ٥���}�����t����($%�������є�Aɽ٥���Q�������(%�(%���͕�ѥ��̰����������ѕ�l�͕�ѥ��̉t쁽���($%Ʌ�M��ѥ��̰���Ȁ�􁵅�͡��)M=9I�ܡ͕�ѥ��̤($%�����Ȁ�􁹥���($$%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聤�ḹP�����������ḹ5͝%�م���)M=8���($$%ɕ��ɸ($%�($%�������є�M��ѥ��̀�Ʌ�M��ѥ���($%����ѕ�l�͕�ѥ��̉t��Ʌ�M��ѥ���(%�(($���I��م����є�UI1́������Ёѡ������ͥ��䁹�ܤ��ɽ٥��ȁ�����($���]�����ɽ٥���}�����������̰����ѥ������}��͔����Ё��ͼ����́م����ѥ��($�����ȁѡ����܁������P��ɕٕ��́MMI�٥��
C�I����
@�������ݥэ��(%����
��������􁍅�����є�Aɽ٥���Q��������ɕ��Aɽ٥��ȹAɽ٥���Q���((%������	�͔�����������ѕ�l����}��͔�t쁽���($%���̰�|�����	�͔����ɥ����̀�􀈈��($$%�����Ȁ��م����ѕAɽ٥���UI0�̰��������є�Aɽ٥���Q����쁕�Ȁ�􁹥���($$$%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聕�ȹ�ɽȠ���($$$%ɕ��ɸ($$%�($%�(%􁕱͔��������
�����������������є�A%	�͔��􀈈��($%�����Ȁ��م����ѕAɽ٥���UI0��������є�A%	�͔���������є�Aɽ٥���Q����쁕�Ȁ�􁹥���($$%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聕�ȹ�ɽȠ���($$%ɕ��ɸ($%�(%�(%�����͕UI0�����������ѕ�l���͕}�ɰ�t쁽���($%���̰�|��􁉅͕UI0����ɥ����̀�􀈈��($$%�����Ȁ��م����ѕAɽ٥���UI0�̰��������є�Aɽ٥���Q����쁕�Ȁ�􁹥���($$$%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聕�ȹ�ɽȠ���($$$%ɕ��ɸ($$%�($%�(%�(($���9�ɵ���锁=��������͔�UI0�Ѽ�����Ց����āͼ������������ѡ́�͔�ѡ��ͅ���م�Ք�(%��ɵ����=�����A%	�͔���������є�(%����������є�A%	�͔�����ɕ��Aɽ٥��ȹA%	�͔��($%����ѕ�l����}��͔�t�􁍅�����є�A%	�͔(%�((%�����Ȁ��م����ѕ
���AQ=�ѡAɽ٥���
������є�ȹ
��ѕ�Р������ѽɔ��������������є�쁕�Ȁ�􁹥���($%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聕�ȹ�ɽȠ���($%ɕ��ɸ(%�(%�����Ȁ��م����ѕAɽ٥�����������M��ѥ��̠��������є�쁕�Ȁ�􁹥���($%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聤�ḹP�����������ḹ5͝%�م���I��Օ�а���ȹ�ɽȠ����($%ɕ��ɸ(%�(($���QɅ����������������ɔ�����є���ȁɕ�����䁍������(%مȁ���9������ɥ��(%������ɽ٥���I����􁹥���($%���9�������ɕ��Aɽ٥��ȹ9���(%�((%�����Ȁ�􁠹�ѽɔ�U���ѕAɽ٥��ȡȹ
��ѕ�Р�����������ѕ̤쁕�Ȁ�􁹥���($%ͱ����ɽȠ��ɽ٥���̹����є������ɽȈ����Ȥ($%�ɥѕ)M=8�ܰ������Mх���%�ѕɹ��M��ٕ��ɽȰ����m��ɥ��u��ɥ��쉕�ɽȈ聕�ȹ�ɽȠ���($%ɕ��ɸ(%�(($���M幌����������ɕ������ݥѠ�����ѕ���ɽ٥���(%������ɽ٥���I����􁹥���($%�������ѕ�����Ȁ�􁠹�ѽɔ���Aɽ٥��ȡȹ
��ѕ�Р������쁕�Ȁ�􁹥���($$$���U�ɕ���ѕȁ������������ɕ������Ѽ��ɕٕ�Ё����Ё���ɥ��($$%������9�����􀈈�������9����������ѕ��9�����($$$%���ɽ٥���I���U�ɕ���ѕ���Q����С����ѕ��Q�����%�����9����($$%�($$%��������ѕ����������($$$%���ɽ٥���I���U�ɕ���ѕ���Q����С����ѕ��Q�����%������ѕ��9����($$%􁕱͔��($$$%��ɕ���ѕ�%�5���������ѕ��($$%�($%�(%�(($���9�ѥ���Չ͍ɥ���̀������
@�ɔ�ɕ����Ʌѥ��������Ёѡ��������(%�������ѕ�����Ȁ�􁠹�ѽɔ���Aɽ٥��ȡȹ
��ѕ�Р������쁕�Ȁ�􁹥���($%������Aɽ٥���
����%�م����є�ȹ
��ѕ�Р�������ѕ��Q�����%������ѕ��9����($%������9�����􀈈�������9����������ѕ��9�����($$%������Aɽ٥���
����%�م����є�ȹ
��ѕ�Р�������ѕ��Q�����%�����9����($%�(%�((%����Ց�С���͝	�̰�Ȱ���ɽ٥��ȹ����ѕ������ɽ٥��Ȉ�����M�ɥ�����(%�ɥѕ)M=8�ܰ������Mх���=,�����m��ɥ��u��ɥ����х��̈耉����ѕ����)�()�չ������Aɽ٥����!�����Ȥ����������ѕAɽ٥��ȡ܁�����I�����͕]ɥѕȰ�Ȁ������I��Օ�Ф��(%�����������Ʌ��1������Ȥ(%������Ȁ���ե��A��͔�ȹA�ѡY��Ք�������(%�����Ȁ�􁹥���($%�ɥѕ)M=8�ܰ������Mх���	��I��Օ�а����m��ɥ��u��ɥ��쉕�ɽȈ聤�ḹP�����������ḹ5͝%�م���%����ɽ٥��Ȉ���($%ɕ��ɸ(%�(($���I�����ɽ٥��ȁ����ɔ�����ѥ���ͼ�ݔ�����չɕ���ѕȁ��(%مȁ�ɽ٥���9������ɥ��(%مȁ�ɽ٥���Q�����%��ե��UU%(%��������Ȁ�􁠹�ѽɔ���Aɽ٥��ȡȹ
��ѕ�Р������쁕�Ȁ�􁹥���($%�ɽ٥���9�������9���($%�ɽ٥���Q�����%����Q�����%(%�((%�����Ȁ�􁠹�ѽɔ����ѕAɽ٥��ȡȹ
��ѕ�Р������쁕�Ȁ�􁹥���($%ͱ����ɽȠ��ɽ٥���̹����є������ɽȈ����Ȥ($%�ɥѕ)M=8�ܰ������Mх���%�ѕɹ��M��ٕ��ɽȰ����m��ɥ��u��ɥ��쉕�ɽȈ聕�ȹ�ɽȠ���($%ɕ��ɸ(%�((%������ɽ٥���I����􁹥������ɽ٥���9�����􀈈��($%���ɽ٥���I���U�ɕ���ѕ���Q����С�ɽ٥���Q�����%���ɽ٥���9����(%�(%����ɽ٥���9�����􀈈��($%������Aɽ٥���
����%�م����є�ȹ
��ѕ�Р����ɽ٥���Q�����%���ɽ٥���9����(%�((%����Ց�С���͝	�̰�Ȱ���ɽ٥��ȹ����ѕ������ɽ٥��Ȉ�����M�ɥ�����(%�ɥѕ)M=8�ܰ������Mх���=,�����m��ɥ��u��ɥ����х��̈耉����ѕ����)�(
