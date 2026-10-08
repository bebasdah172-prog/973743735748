use axum::{
    Router,
    body::{Body, to_bytes},
    extract::{ConnectInfo, Path, Request},
    http::{Method, StatusCode},
    response::{IntoResponse, Json, Response},
    routing::{any, get},
};
use futures_util::StreamExt;
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode};
use rusqlite::{Connection, ErrorCode, OpenFlags, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    env,
    fs::{self, OpenOptions},
    io::Write,
    net::SocketAddr,
    path::PathBuf,
    sync::{Mutex as StdMutex, OnceLock},
    time::{Duration, Instant},
};
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};

mod auth;
mod oidc;

const VERSION_CACHE_TTL: Duration = Duration::from_secs(300);
const VERSION_FAILURE_BACKOFF: Duration = Duration::from_secs(30);
const NPM_VERSION_URL: &str = "https://registry.npmjs.org/vansrouter/latest";
const MODEL_CATALOG_JSON: &str = include_str!("../models.json");
const PRICING_CATALOG_JSON: &str = include_str!("../pricing.json");
const MAX_CHAT_BODY_BYTES: usize = 128 * 1024 * 1024;
const MAX_ADMIN_BODY_BYTES: usize = 1024 * 1024;

#[derive(Deserialize)]
struct ModelCatalog {
    providers: Vec<CatalogProvider>,
}

#[derive(Deserialize)]
struct PricingPattern {
    pattern: String,
    pricing: Value,
}

#[derive(Deserialize)]
struct PricingCatalog {
    canonical: HashMap<String, Value>,
    providers: HashMap<String, HashMap<String, Value>>,
    patterns: Vec<PricingPattern>,
}

fn pricing_catalog() -> &'static PricingCatalog {
    static CATALOG: OnceLock<PricingCatalog> = OnceLock::new();
    CATALOG.get_or_init(|| {
        serde_json::from_str(PRICING_CATALOG_JSON)
            .expect("bundled pricing catalog must be valid JSON")
    })
}

fn merge_pricing(base: Option<&Value>, custom: Option<&Value>) -> Option<Value> {
    match (base, custom) {
        (Some(base), Some(custom)) => {
            let mut merged = base.as_object()?.clone();
            if let Some(overrides) = custom.as_object() {
                merged.extend(overrides.clone());
            }
            Some(Value::Object(merged))
        }
        (Some(base), None) => Some(base.clone()),
        (None, Some(custom)) => Some(custom.clone()),
        (None, None) => None,
    }
}

fn model_pricing(
    connection: &Connection,
    provider: &str,
    model: &str,
) -> Result<Option<Value>, String> {
    let catalog = pricing_catalog();
    let base_model = model.rsplit('/').next().unwrap_or(model);
    let user_pricing: Option<String> = connection
        .query_row(
            "SELECT value FROM kv WHERE scope = 'pricing' AND key = ?1",
            params![provider],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| error.to_string())?;
    let user_pricing = user_pricing
        .map(|raw| serde_json::from_str::<Value>(&raw).map_err(|error| error.to_string()))
        .transpose()?;
    let custom = user_pricing.as_ref().and_then(|pricing| pricing.get(model));
    let default = catalog
        .providers
        .get(provider)
        .and_then(|pricing| pricing.get(model))
        .or_else(|| {
            catalog
                .canonical
                .get(base_model)
                .or_else(|| catalog.canonical.get(model))
        })
        .or_else(|| {
            catalog
                .patterns
                .iter()
                .find(|entry| {
                    wildcard_match(&entry.pattern, base_model)
                        || wildcard_match(&entry.pattern, model)
                })
                .map(|entry| &entry.pricing)
        });
    if default.is_some() || custom.is_some() {
        return Ok(merge_pricing(default, custom));
    }
    Ok(None)
}

fn wildcard_match(pattern: &str, value: &str) -> bool {
    let pattern = pattern.to_ascii_lowercase().into_bytes();
    let value = value.to_ascii_lowercase().into_bytes();
    let mut matches = vec![vec![false; value.len() + 1]; pattern.len() + 1];
    matches[0][0] = true;
    for pattern_index in 0..pattern.len() {
        for value_index in 0..=value.len() {
            if !matches[pattern_index][value_index] {
                continue;
            }
            if pattern[pattern_index] == b'*' {
                matches[pattern_index + 1][value_index] = true;
                if value_index < value.len() {
                    matches[pattern_index][value_index + 1] = true;
                }
            } else if value.get(value_index) == Some(&pattern[pattern_index]) {
                matches[pattern_index + 1][value_index + 1] = true;
            }
        }
    }
    matches[pattern.len()][value.len()]
}

fn calculate_chat_cost(usage: &Value, pricing: &Value) -> f64 {
    let token_count = |names: &[&str]| {
        names
            .iter()
            .find_map(|name| {
                name.split('.')
                    .try_fold(usage, |value, part| value.get(part))
                    .and_then(Value::as_f64)
            })
            .unwrap_or_default()
    };
    let rate = |name: &str, fallback: &str| {
        pricing
            .get(name)
            .and_then(Value::as_f64)
            .or_else(|| pricing.get(fallback).and_then(Value::as_f64))
            .unwrap_or_default()
    };
    let input_tokens = token_count(&["prompt_tokens", "input_tokens"]);
    let cached_tokens = token_count(&[
        "cached_tokens",
        "cache_read_input_tokens",
        "prompt_tokens_details.cached_tokens",
    ]);
    let cache_creation_tokens = token_count(&["cache_creation_input_tokens"]);
    let output_tokens = token_count(&["completion_tokens", "output_tokens"]);
    let reasoning_tokens = token_count(&["reasoning_tokens"]);
    let non_cached_input = (input_tokens - cached_tokens - cache_creation_tokens).max(0.0);
    (non_cached_input * rate("input", "input")
        + cached_tokens * rate("cached", "input")
        + cache_creation_tokens * rate("cache_creation", "input")
        + output_tokens * rate("output", "output")
        + reasoning_tokens * rate("reasoning", "output"))
        / 1_000_000.0
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CatalogProvider {
    id: String,
    alias: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    display_name: String,
    no_auth: bool,
    #[serde(default)]
    auth_modes: Vec<String>,
    #[serde(default)]
    service_kinds: Vec<String>,
    models: Vec<CatalogModel>,
}

#[derive(Clone, Deserialize)]
struct CatalogModel {
    id: String,
    capabilities: Value,
}

#[derive(Clone, Debug, Serialize)]
struct ListedModel {
    id: String,
    object: &'static str,
    owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    capabilities: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_length: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u64>,
}

#[derive(Debug, Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ListedModel>,
}

#[derive(Default)]
struct ApiKeyAcl {
    allowed_providers: Option<Vec<String>>,
    allowed_combos: Option<Vec<String>>,
    allowed_kinds: Option<Vec<String>>,
}

struct ApiKeyLimits {
    id: String,
    requests_per_minute: Option<u64>,
    max_input_tokens: Option<u64>,
    max_output_tokens: Option<u64>,
    has_ip_whitelist: bool,
}

#[derive(Clone)]
struct ActiveConnection {
    provider: String,
    output_alias: String,
    enabled_models: Option<Vec<String>>,
}

fn model_catalog() -> &'static ModelCatalog {
    static CATALOG: OnceLock<ModelCatalog> = OnceLock::new();
    CATALOG.get_or_init(|| {
        serde_json::from_str(MODEL_CATALOG_JSON)
            .expect("bundled provider model catalog must be valid JSON")
    })
}

fn database_path() -> Result<PathBuf, String> {
    Ok(data_dir()?.join("db").join("data.sqlite"))
}

fn data_dir() -> Result<PathBuf, String> {
    let home = env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .ok_or_else(|| "HOME is not set".to_owned())?;
    let default_dir = || {
        #[cfg(windows)]
        {
            let app_data = env::var_os("APPDATA")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(&home).join("AppData").join("Roaming"));
            app_data.join("9router")
        }
        #[cfg(not(windows))]
        {
            PathBuf::from(&home).join(".9router")
        }
    };
    let configured = env::var("DATA_DIR").ok();
    let production =
        env::var("NODE_ENV").as_deref() == Ok("production") || env::var_os("pm_id").is_some();
    let allow_temp = env::var("DATA_DIR_ALLOW_TEMP").as_deref() == Ok("1");
    Ok(resolve_data_dir(
        configured.as_deref(),
        default_dir(),
        production,
        allow_temp,
        cfg!(windows),
        cfg!(target_os = "macos"),
    ))
}

fn resolve_data_dir(
    configured: Option<&str>,
    default_dir: PathBuf,
    production: bool,
    allow_temp: bool,
    windows: bool,
    macos: bool,
) -> PathBuf {
    let Some(configured) = configured else {
        return default_dir;
    };
    let normalized = configured.to_ascii_lowercase().replace('\\', "/");
    let looks_temporary = normalized.contains("smoke")
        || normalized.starts_with("/tmp/9router-data-")
        || (macos && normalized.contains("/var/folders/"));
    if looks_temporary && !allow_temp && production {
        eprintln!(
            "[DATA_DIR] '{}' looks temporary; using persistent default '{}'",
            configured,
            default_dir.display()
        );
        return default_dir;
    }
    if windows && configured.starts_with('/') {
        eprintln!(
            "[DATA_DIR] '{}' is a Unix path on Windows; using persistent default '{}'",
            configured,
            default_dir.display()
        );
        return default_dir;
    }
    PathBuf::from(configured)
}

fn open_database() -> Result<Connection, String> {
    let path = database_path()?;
    if !path.is_file() {
        return Err(format!("database file does not exist: {}", path.display()));
    }
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|error| error.to_string())?;
    connection
        .busy_timeout(Duration::from_secs(5))
        .map_err(|error| error.to_string())?;
    connection
        .pragma_update(None, "query_only", "ON")
        .map_err(|error| error.to_string())?;
    Ok(connection)
}

fn open_database_for_writes() -> Result<Connection, String> {
    let path = database_path()?;
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|error| error.to_string())?;
    connection
        .busy_timeout(Duration::from_secs(5))
        .map_err(|error| error.to_string())?;
    Ok(connection)
}

fn parse_acl(raw: Option<String>) -> Option<Vec<String>> {
    raw.and_then(|value| serde_json::from_str::<Vec<String>>(&value).ok())
}

fn parse_string_list(value: Option<&Value>) -> Option<Vec<String>> {
    value.and_then(Value::as_array).map(|items| {
        items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect()
    })
}

fn extract_api_key(request: &Request) -> Option<String> {
    if let Some(value) = request
        .headers()
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
    {
        return Some(value.to_owned());
    }
    request
        .headers()
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

fn extract_cli_token(request: &Request) -> Option<&str> {
    request
        .headers()
        .get("x-9r-cli-token")
        .and_then(|value| value.to_str().ok())
}

fn is_valid_cli_token(token: &str) -> bool {
    let Ok(directory) = data_dir() else {
        return false;
    };
    let Ok(machine_id) = std::fs::read_to_string(directory.join("machine-id")) else {
        return false;
    };
    let Ok(secret) = std::fs::read_to_string(directory.join("auth").join("cli-secret")) else {
        return false;
    };
    let mut hasher = Sha256::new();
    hasher.update(machine_id.trim());
    hasher.update("9r-cli-auth");
    hasher.update(secret.trim());
    let digest = format!("{:x}", hasher.finalize());
    token == &digest[..16]
}

fn is_loopback_host(host: &str) -> bool {
    let host = host.trim().trim_matches(['[', ']']).to_ascii_lowercase();
    matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1")
}

fn is_local_request(request: &Request, peer: SocketAddr) -> bool {
    let peer_is_loopback = match peer.ip() {
        std::net::IpAddr::V4(address) => address.is_loopback(),
        std::net::IpAddr::V6(address) => {
            address.is_loopback()
                || address
                    .to_ipv4_mapped()
                    .is_some_and(|mapped| mapped.is_loopback())
        }
    };
    let has_proxy_headers = request.headers().contains_key("x-9r-via-proxy")
        || request.headers().contains_key("x-forwarded-for")
        || request.headers().contains_key("x-forwarded-host")
        || request.headers().contains_key("forwarded");
    if !peer_is_loopback || has_proxy_headers {
        return false;
    }
    let Some(origin) = request.headers().get("origin") else {
        return true;
    };
    let Ok(origin) = origin.to_str() else {
        return false;
    };
    let Ok(uri) = origin.parse::<axum::http::Uri>() else {
        return false;
    };
    uri.host().is_some_and(is_loopback_host)
}

fn public_api_access_allowed(
    is_local: bool,
    valid_cli_token: bool,
    valid_api_key: bool,
    require_api_key: bool,
    allow_remote_no_api_key: bool,
) -> bool {
    is_local || valid_cli_token || valid_api_key || (!require_api_key && allow_remote_no_api_key)
}

fn setting_bool(connection: &Connection, name: &str) -> Result<Option<bool>, String> {
    let raw = connection
        .query_row("SELECT data FROM settings WHERE id = 1", [], |row| {
            row.get::<_, String>(0)
        })
        .optional()
        .map_err(|error| error.to_string())?;
    Ok(raw
        .as_deref()
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .and_then(|settings| settings.get(name).and_then(Value::as_bool)))
}

fn api_error(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(
            json!({ "error": { "message": message, "type": match status {
            StatusCode::UNAUTHORIZED => "authentication_error",
            StatusCode::FORBIDDEN => "permission_error",
            StatusCode::BAD_REQUEST | StatusCode::PAYLOAD_TOO_LARGE => "invalid_request_error",
            StatusCode::NOT_FOUND => "not_found_error",
            StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
            _ => "server_error",
        } } }),
        ),
    )
        .into_response()
}

fn admin_error(status: StatusCode, message: &str) -> Response {
    (status, Json(json!({ "error": message }))).into_response()
}

fn is_sqlite_constraint(error: &rusqlite::Error) -> bool {
    matches!(
        error,
        rusqlite::Error::SqliteFailure(code, _) if code.code == ErrorCode::ConstraintViolation
    )
}

fn dashboard_token_valid(token: &str, secret: &str) -> bool {
    let mut validation = Validation::new(Algorithm::HS256);
    validation.leeway = 0;
    validation.validate_nbf = true;
    validation.required_spec_claims.insert("exp".to_owned());
    decode::<Value>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &validation,
    )
    .is_ok()
}

fn dashboard_session_token(request: &Request) -> Option<&str> {
    request
        .headers()
        .get("cookie")?
        .to_str()
        .ok()?
        .split(';')
        .filter_map(|part| part.trim().split_once('='))
        .find_map(|(name, value)| (name == "auth_token").then_some(value))
}

fn require_dashboard_auth(request: &Request) -> Result<(), (StatusCode, &'static str)> {
    if extract_cli_token(request).is_some_and(is_valid_cli_token) {
        return Ok(());
    }
    let secret = auth::dashboard_secret().map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Dashboard JWT_SECRET is not configured on the Rust backend",
        )
    })?;
    if dashboard_session_token(request).is_some_and(|token| dashboard_token_valid(token, &secret)) {
        return Ok(());
    }
    Err((
        StatusCode::UNAUTHORIZED,
        "Dashboard authentication required",
    ))
}

async fn parse_admin_json(body: Body) -> Result<Value, (StatusCode, &'static str)> {
    let bytes = to_bytes(body, MAX_ADMIN_BODY_BYTES)
        .await
        .map_err(|_| (StatusCode::PAYLOAD_TOO_LARGE, "Request body too large"))?;
    serde_json::from_slice(&bytes).map_err(|_| (StatusCode::BAD_REQUEST, "Invalid JSON body"))
}

fn load_key_acl(connection: &Connection, api_key: &str) -> Result<Option<ApiKeyAcl>, String> {
    let row = connection
        .query_row(
            "SELECT isActive, allowedProviders, allowedCombos, allowedKinds FROM apiKeys WHERE key = ?1",
            params![api_key],
            |row| {
                Ok((
                    row.get::<_, Option<i64>>(0)?.unwrap_or_default() != 0,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, Option<String>>(3)?,
                ))
            },
        )
        .optional()
        .map_err(|error| error.to_string())?;
    let Some((active, providers, combos, kinds)) = row else {
        return Ok(None);
    };
    if !active {
        return Ok(None);
    }
    Ok(Some(ApiKeyAcl {
        allowed_providers: parse_acl(providers),
        allowed_combos: parse_acl(combos),
        allowed_kinds: parse_acl(kinds),
    }))
}

fn load_api_key_limits(
    connection: &Connection,
    api_key: &str,
) -> Result<Option<ApiKeyLimits>, String> {
    connection
        .query_row(
            "SELECT id, requestsPerMinute, maxInputTokens, maxOutputTokens, ipWhitelist, isActive FROM apiKeys WHERE key = ?1",
            params![api_key],
            |row| {
                let active = row.get::<_, Option<i64>>(5)?.unwrap_or_default() != 0;
                let whitelist = row.get::<_, Option<String>>(4)?;
                Ok((
                    active,
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<i64>>(1)?,
                    row.get::<_, Option<i64>>(2)?,
                    row.get::<_, Option<i64>>(3)?,
                    whitelist,
                ))
            },
        )
        .optional()
        .map_err(|error| error.to_string())
        .map(|row| {
            row.and_then(|(active, id, requests, input, output, whitelist)| {
                active.then(|| ApiKeyLimits {
                    id,
                    requests_per_minute: requests.and_then(|value| u64::try_from(value).ok()),
                    max_input_tokens: input.and_then(|value| u64::try_from(value).ok()),
                    max_output_tokens: output.and_then(|value| u64::try_from(value).ok()),
                    has_ip_whitelist: whitelist
                        .as_deref()
                        .and_then(|raw| serde_json::from_str::<Vec<String>>(raw).ok())
                        .is_some(),
                })
            })
        })
}

fn consume_api_key_request(
    connection: &Connection,
    api_key_id: &str,
    limit: u64,
    now_ms: u64,
) -> Result<(bool, u64), String> {
    let window_start = now_ms / 60_000 * 60_000;
    let limit = i64::try_from(limit).map_err(|error| error.to_string())?;
    let window_start_sql = i64::try_from(window_start).map_err(|error| error.to_string())?;
    let result = connection
        .execute(
            "INSERT INTO apiKeyRateLimits(apiKeyId, windowStart, requestCount) VALUES(?1, ?2, 1)
             ON CONFLICT(apiKeyId) DO UPDATE SET
               requestCount = CASE
                 WHEN apiKeyRateLimits.windowStart = excluded.windowStart THEN apiKeyRateLimits.requestCount + 1
                 ELSE 1
               END,
               windowStart = excluded.windowStart
             WHERE apiKeyRateLimits.windowStart != excluded.windowStart
               OR apiKeyRateLimits.requestCount < ?3",
            params![api_key_id, window_start_sql, limit],
        )
        .map_err(|error| error.to_string())?;
    if result > 0 {
        Ok((true, 0))
    } else {
        Ok((
            false,
            ((window_start + 60_000 - now_ms).div_ceil(1000)).max(1),
        ))
    }
}

fn read_active_connections(connection: &Connection) -> Result<Vec<ActiveConnection>, String> {
    let mut statement = connection
        .prepare("SELECT provider, data FROM providerConnections WHERE isActive = 1")
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| {
            let provider: String = row.get(0)?;
            let data: String = row.get(1)?;
            Ok((provider, data))
        })
        .map_err(|error| error.to_string())?;
    let mut result = Vec::new();
    let mut seen = HashSet::new();
    for row in rows {
        let (provider, raw_data) = row.map_err(|error| error.to_string())?;
        if !seen.insert(provider.clone()) {
            continue;
        }
        let data: Value = serde_json::from_str(&raw_data).unwrap_or_else(|_| json!({}));
        let provider_data = data.get("providerSpecificData").and_then(Value::as_object);
        let provider_catalog = model_catalog()
            .providers
            .iter()
            .find(|entry| entry.id == provider || entry.alias == provider);
        let static_alias = provider_catalog
            .map(|entry| entry.alias.as_str())
            .unwrap_or(provider.as_str());
        let output_alias = provider_data
            .and_then(|data| data.get("prefix"))
            .and_then(Value::as_str)
            .filter(|prefix| !prefix.is_empty())
            .unwrap_or(static_alias)
            .to_owned();
        let enabled_models = provider_data
            .and_then(|data| data.get("enabledModels"))
            .and_then(|value| parse_string_list(Some(value)))
            .filter(|models| !models.is_empty());
        result.push(ActiveConnection {
            provider,
            output_alias,
            enabled_models,
        });
    }
    Ok(result)
}

fn read_kv_scope(connection: &Connection, scope: &str) -> Result<HashMap<String, Value>, String> {
    let mut statement = connection
        .prepare("SELECT key, value FROM kv WHERE scope = ?1")
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map(params![scope], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|error| error.to_string())?;
    let mut values = HashMap::new();
    for row in rows {
        let (key, raw_value) = row.map_err(|error| error.to_string())?;
        let value = serde_json::from_str(&raw_value).unwrap_or(Value::Null);
        values.insert(key, value);
    }
    Ok(values)
}

fn provider_allowed(
    connection: &Connection,
    acl: &ApiKeyAcl,
    provider: &str,
    provider_id: &str,
) -> Result<bool, String> {
    let Some(allowed) = &acl.allowed_providers else {
        return Ok(true);
    };
    if allowed
        .iter()
        .any(|entry| entry == provider || entry == provider_id)
    {
        return Ok(true);
    }
    if let Some(catalog_provider) = model_catalog()
        .providers
        .iter()
        .find(|entry| entry.id == provider_id)
        && allowed.iter().any(|entry| entry == &catalog_provider.alias)
    {
        return Ok(true);
    }
    let node_data = connection
        .query_row(
            "SELECT data FROM providerNodes WHERE id = ?1",
            params![provider_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|error| error.to_string())?;
    let prefix = node_data
        .as_deref()
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .and_then(|data| {
            data.get("prefix")
                .and_then(Value::as_str)
                .map(str::to_owned)
        });
    Ok(prefix.is_some_and(|prefix| allowed.iter().any(|entry| entry == &prefix)))
}

fn kind_allowed(acl: &ApiKeyAcl, kind: &str) -> bool {
    acl.allowed_kinds
        .as_ref()
        .is_none_or(|allowed| allowed.iter().any(|entry| entry == kind))
}

fn combo_allowed(acl: &ApiKeyAcl, name: &str) -> bool {
    acl.allowed_combos
        .as_ref()
        .is_none_or(|allowed| allowed.iter().any(|entry| entry == name))
}

fn model_entry(id: String, owner: String, capabilities: Option<Value>) -> ListedModel {
    let context_length = capabilities
        .as_ref()
        .and_then(|caps| caps.get("contextWindow"))
        .and_then(Value::as_u64);
    let max_completion_tokens = capabilities
        .as_ref()
        .and_then(|caps| caps.get("maxOutput"))
        .and_then(Value::as_u64);
    ListedModel {
        id,
        object: "model",
        owned_by: owner,
        capabilities,
        context_length,
        max_completion_tokens,
    }
}

fn normalize_model_id(id: &str, aliases: &[&str]) -> Option<String> {
    let mut normalized = id.trim();
    for alias in aliases {
        if let Some(suffix) = normalized.strip_prefix(&format!("{alias}/")) {
            normalized = suffix;
            break;
        }
    }
    (!normalized.is_empty()).then(|| normalized.to_owned())
}

fn infer_llm_kind(model_id: &str) -> bool {
    let id = model_id.to_ascii_lowercase();
    ![
        "embed",
        "tts",
        "speech",
        "audio",
        "voice",
        "image",
        "imagen",
        "dall-e",
        "dalle",
        "flux",
        "sdxl",
        "sd-",
        "stable-diffusion",
        "video",
    ]
    .iter()
    .any(|marker| id.contains(marker))
}

fn read_v1_models(
    api_key: Option<String>,
    cli_token: Option<String>,
    is_local: bool,
) -> Result<ModelList, (StatusCode, String)> {
    let connection = open_database().map_err(|error| {
        eprintln!("[v1/models] database unavailable: {error}");
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Backend database unavailable".to_owned(),
        )
    })?;
    read_v1_models_from_connection(&connection, api_key, cli_token, is_local)
}

fn read_v1_models_from_connection(
    connection: &Connection,
    api_key: Option<String>,
    cli_token: Option<String>,
    is_local: bool,
) -> Result<ModelList, (StatusCode, String)> {
    let settings_raw = connection
        .query_row("SELECT data FROM settings WHERE id = 1", [], |row| {
            row.get::<_, String>(0)
        })
        .optional()
        .map_err(|error| {
            eprintln!("[v1/models] settings query failed: {error}");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "Backend database unavailable".to_owned(),
            )
        })?;
    let require_api_key = settings_raw
        .as_deref()
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .and_then(|settings| settings.get("requireApiKey").and_then(Value::as_bool))
        .unwrap_or_else(|| env::var("REQUIRE_API_KEY").as_deref() == Ok("true"));
    let allow_remote_no_api_key = settings_raw
        .as_deref()
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .and_then(|settings| settings.get("allowRemoteNoApiKey").and_then(Value::as_bool))
        .unwrap_or(false);
    let cli_token_valid = cli_token.as_deref().is_some_and(is_valid_cli_token);
    let key_acl = if let Some(api_key) = api_key.as_deref() {
        load_key_acl(connection, api_key).map_err(|error| {
            eprintln!("[v1/models] API key lookup failed: {error}");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "Backend database unavailable".to_owned(),
            )
        })?
    } else {
        None
    };
    if !public_api_access_allowed(
        is_local,
        cli_token_valid,
        key_acl.is_some(),
        require_api_key,
        allow_remote_no_api_key,
    ) {
        return Err((
            StatusCode::UNAUTHORIZED,
            "API key required for remote API access".to_owned(),
        ));
    }

    let acl = if require_api_key {
        if api_key.is_none() {
            return Err((StatusCode::UNAUTHORIZED, "Missing API key".to_owned()));
        }
        Some(key_acl.ok_or_else(|| (StatusCode::UNAUTHORIZED, "Invalid API key".to_owned()))?)
    } else {
        None
    };
    let acl = acl.unwrap_or_default();

    let connections = read_active_connections(connection).map_err(|error| {
        eprintln!("[v1/models] provider connection query failed: {error}");
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Backend database unavailable".to_owned(),
        )
    })?;
    let custom_models = read_kv_scope(connection, "customModels").map_err(|error| {
        eprintln!("[v1/models] custom model query failed: {error}");
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Backend database unavailable".to_owned(),
        )
    })?;
    let aliases = read_kv_scope(connection, "modelAliases").map_err(|error| {
        eprintln!("[v1/models] model alias query failed: {error}");
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Backend database unavailable".to_owned(),
        )
    })?;
    let disabled = read_kv_scope(connection, "disabledModels").map_err(|error| {
        eprintln!("[v1/models] disabled model query failed: {error}");
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Backend database unavailable".to_owned(),
        )
    })?;

    let mut entries = Vec::new();
    let mut seen = HashSet::new();
    let mut add_entry = |entry: ListedModel| {
        if seen.insert(entry.id.clone()) {
            entries.push(entry);
        }
    };

    {
        let mut statement = connection
            .prepare("SELECT name, kind FROM combos ORDER BY createdAt ASC")
            .map_err(|error| {
                eprintln!("[v1/models] combo query failed: {error}");
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Backend database unavailable".to_owned(),
                )
            })?;
        let rows = statement
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
            })
            .map_err(|error| {
                eprintln!("[v1/models] combo query failed: {error}");
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Backend database unavailable".to_owned(),
                )
            })?;
        for row in rows {
            let (name, kind) = row.map_err(|error| {
                eprintln!("[v1/models] combo row invalid: {error}");
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Backend database unavailable".to_owned(),
                )
            })?;
            if kind.as_deref().unwrap_or("llm") != "llm"
                || !kind_allowed(&acl, "llm")
                || !combo_allowed(&acl, &name)
            {
                continue;
            }
            add_entry(model_entry(
                format!("combo/{name}"),
                "combo".to_owned(),
                None,
            ));
        }
    }

    let mut active_by_id = HashMap::new();
    for active in &connections {
        active_by_id.insert(active.provider.as_str(), active);
    }
    for provider in &model_catalog().providers {
        let active = active_by_id
            .get(provider.id.as_str())
            .copied()
            .or_else(|| active_by_id.get(provider.alias.as_str()).copied());
        if active.is_none() && !provider.no_auth {
            continue;
        }
        let output_alias = active
            .map(|connection| connection.output_alias.as_str())
            .unwrap_or(provider.alias.as_str());
        if !kind_allowed(&acl, "llm")
            || !provider_allowed(connection, &acl, output_alias, &provider.id).map_err(|error| {
                eprintln!("[v1/models] provider ACL lookup failed: {error}");
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Backend database unavailable".to_owned(),
                )
            })?
        {
            continue;
        }
        let mut ids: Vec<String> = active
            .and_then(|connection| connection.enabled_models.clone())
            .unwrap_or_else(|| {
                provider
                    .models
                    .iter()
                    .map(|model| model.id.clone())
                    .collect()
            });
        let mut custom_llm_ids = HashSet::new();
        let mut model_aliases = vec![output_alias, provider.alias.as_str(), provider.id.as_str()];
        model_aliases.dedup();

        for value in aliases.values() {
            let Some(target) = value.as_str() else {
                continue;
            };
            if model_aliases
                .iter()
                .any(|alias| target.starts_with(&format!("{alias}/")))
                && let Some(model_id) = normalize_model_id(target, &model_aliases)
            {
                ids.push(model_id);
            }
        }
        for value in custom_models.values() {
            let Some(custom) = value.as_object() else {
                continue;
            };
            let custom_kind = custom.get("type").and_then(Value::as_str).unwrap_or("llm");
            if custom_kind != "llm" && custom_kind != "imageToText" {
                continue;
            }
            if !custom
                .get("providerAlias")
                .and_then(Value::as_str)
                .is_some_and(|alias| model_aliases.contains(&alias))
            {
                continue;
            }
            if let Some(id) = custom.get("id").and_then(Value::as_str) {
                ids.push(id.to_owned());
                custom_llm_ids.insert(id.to_owned());
            }
        }
        ids.sort();
        ids.dedup();

        let disabled_for_alias = disabled.get(output_alias).and_then(Value::as_array);
        let disabled_for_static_alias = disabled.get(&provider.alias).and_then(Value::as_array);
        for model_id in ids {
            let Some(model_id) = normalize_model_id(&model_id, &model_aliases) else {
                continue;
            };
            let is_static_llm = provider.models.iter().any(|model| model.id == model_id);
            if !is_static_llm && !custom_llm_ids.contains(&model_id) && !infer_llm_kind(&model_id) {
                continue;
            }
            if disabled_for_alias
                .is_some_and(|items| items.iter().any(|item| item.as_str() == Some(&model_id)))
                || disabled_for_static_alias
                    .is_some_and(|items| items.iter().any(|item| item.as_str() == Some(&model_id)))
            {
                continue;
            }
            let capabilities = if provider.no_auth {
                None
            } else {
                provider
                    .models
                    .iter()
                    .find(|model| model.id == model_id)
                    .map(|model| model.capabilities.clone())
            };
            add_entry(model_entry(
                format!("{output_alias}/{model_id}"),
                output_alias.to_owned(),
                capabilities,
            ));
        }
    }
    for active in &connections {
        let has_static_catalog = model_catalog()
            .providers
            .iter()
            .any(|provider| provider.id == active.provider || provider.alias == active.provider);
        if has_static_catalog || !kind_allowed(&acl, "llm") {
            continue;
        }
        if !provider_allowed(connection, &acl, &active.output_alias, &active.provider).map_err(
            |error| {
                eprintln!("[v1/models] provider ACL lookup failed: {error}");
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Backend database unavailable".to_owned(),
                )
            },
        )? {
            continue;
        }
        let mut prefix_aliases = vec![active.output_alias.as_str(), active.provider.as_str()];
        prefix_aliases.dedup();
        let mut ids = active.enabled_models.clone().unwrap_or_default();
        let mut custom_llm_ids = HashSet::new();
        for value in aliases.values() {
            let Some(target) = value.as_str() else {
                continue;
            };
            if prefix_aliases
                .iter()
                .any(|alias| target.starts_with(&format!("{alias}/")))
                && let Some(model_id) = normalize_model_id(target, &prefix_aliases)
            {
                ids.push(model_id);
            }
        }
        for value in custom_models.values() {
            let Some(custom) = value.as_object() else {
                continue;
            };
            let custom_kind = custom.get("type").and_then(Value::as_str).unwrap_or("llm");
            if custom_kind != "llm" && custom_kind != "imageToText"
                || !custom
                    .get("providerAlias")
                    .and_then(Value::as_str)
                    .is_some_and(|alias| prefix_aliases.contains(&alias))
            {
                continue;
            }
            if let Some(id) = custom.get("id").and_then(Value::as_str) {
                ids.push(id.to_owned());
                custom_llm_ids.insert(id.to_owned());
            }
        }
        ids.sort();
        ids.dedup();
        let disabled_models = disabled.get(&active.output_alias).and_then(Value::as_array);
        for id in ids {
            let Some(id) = normalize_model_id(&id, &prefix_aliases) else {
                continue;
            };
            if !custom_llm_ids.contains(&id) && !infer_llm_kind(&id) {
                continue;
            }
            if disabled_models
                .is_some_and(|items| items.iter().any(|item| item.as_str() == Some(&id)))
            {
                continue;
            }
            add_entry(model_entry(
                format!("{}/{id}", active.output_alias),
                active.output_alias.clone(),
                None,
            ));
        }
    }

    Ok(ModelList {
        object: "list",
        data: entries,
    })
}

async fn v1_models(ConnectInfo(peer): ConnectInfo<SocketAddr>, request: Request) -> Response {
    let api_key = extract_api_key(&request);
    let cli_token = extract_cli_token(&request).map(str::to_owned);
    let is_local = is_local_request(&request, peer);
    match tokio::task::spawn_blocking(move || read_v1_models(api_key, cli_token, is_local)).await {
        Ok(Ok(models)) => {
            Json(json!({ "object": models.object, "data": models.data })).into_response()
        }
        Ok(Err((status, message))) => api_error(status, &message),
        Err(error) => {
            eprintln!("[v1/models] database task failed: {error}");
            api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Backend database unavailable",
            )
        }
    }
}

fn open_database_for_admin() -> Result<Connection, String> {
    let path = database_path()?;
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|error| error.to_string())?;
    connection
        .busy_timeout(Duration::from_secs(5))
        .map_err(|error| error.to_string())?;
    Ok(connection)
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}

fn hmac_sha256_hex(secret: &[u8], message: &[u8]) -> String {
    let mut key = [0_u8; 64];
    if secret.len() > key.len() {
        key[..32].copy_from_slice(&Sha256::digest(secret));
    } else {
        key[..secret.len()].copy_from_slice(secret);
    }
    let mut inner_pad = [0x36_u8; 64];
    let mut outer_pad = [0x5c_u8; 64];
    for (index, byte) in key.iter().enumerate() {
        inner_pad[index] ^= byte;
        outer_pad[index] ^= byte;
    }
    let mut inner = Sha256::new();
    inner.update(inner_pad);
    inner.update(message);
    let inner_digest = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(outer_pad);
    outer.update(inner_digest);
    hex_encode(&outer.finalize())
}

fn read_or_create_api_key_secret() -> Result<Vec<u8>, String> {
    if let Ok(secret) = env::var("API_KEY_SECRET")
        && !secret.is_empty()
    {
        return Ok(secret.into_bytes());
    }
    let path = data_dir()?.join("auth").join("api-key-secret");
    match fs::read_to_string(&path) {
        Ok(secret) if !secret.trim().is_empty() => {
            return Ok(secret.trim().as_bytes().to_vec());
        }
        Ok(_) => return Err("API key secret file is empty".to_owned()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.to_string()),
    }

    let mut secret = [0_u8; 32];
    getrandom::getrandom(&mut secret).map_err(|error| error.to_string())?;
    let parent = path
        .parent()
        .ok_or_else(|| "API key secret path has no parent directory".to_owned())?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    match options.open(&path) {
        Ok(mut file) => {
            let encoded = hex_encode(&secret);
            file.write_all(encoded.as_bytes())
                .map_err(|error| error.to_string())?;
            Ok(encoded.into_bytes())
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let value = fs::read_to_string(path).map_err(|error| error.to_string())?;
            if value.trim().is_empty() {
                return Err("API key secret file is empty".to_owned());
            }
            Ok(value.trim().as_bytes().to_vec())
        }
        Err(error) => Err(error.to_string()),
    }
}

fn consistent_machine_id() -> Result<String, String> {
    let raw = fs::read_to_string(data_dir()?.join("machine-id")).map_err(|_| {
        "Machine identity is not initialized on the shared data directory".to_owned()
    })?;
    let raw = raw.trim();
    if raw.is_empty() {
        return Err("Machine identity is empty".to_owned());
    }
    let salt = env::var("MACHINE_ID_SALT")
        .ok()
        .filter(|salt| !salt.is_empty())
        .unwrap_or_else(|| "endpoint-proxy-salt".to_owned());
    let digest = Sha256::digest(format!("{raw}{salt}").as_bytes());
    Ok(hex_encode(&digest)[..16].to_owned())
}

fn generate_api_key(machine_id: &str, secret: &[u8]) -> Result<String, String> {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let mut key_id = String::with_capacity(6);
    while key_id.len() < 6 {
        let mut random = [0_u8; 16];
        getrandom::getrandom(&mut random).map_err(|error| error.to_string())?;
        for byte in random {
            if byte < 252 {
                key_id.push(ALPHABET[(byte % 36) as usize] as char);
                if key_id.len() == 6 {
                    break;
                }
            }
        }
    }
    let crc = hmac_sha256_hex(secret, format!("{machine_id}{key_id}").as_bytes());
    Ok(format!("sk-{machine_id}-{key_id}-{}", &crc[..8]))
}

fn api_key_limit(value: Option<&Value>, field: &str) -> Result<Option<i64>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() || value.as_str().is_some_and(|value| value.is_empty()) {
        return Ok(None);
    }
    let parsed = if let Some(text) = value.as_str() {
        text.parse::<f64>().ok()
    } else {
        value.as_f64()
    };
    let Some(parsed) = parsed.filter(|number| {
        number.is_finite()
            && number.fract() == 0.0
            && (1.0..=9_007_199_254_740_991.0).contains(number)
    }) else {
        return Err(format!(
            "{field} must be a positive whole number or left blank"
        ));
    };
    Ok(Some(parsed as i64))
}

fn normalize_ip_address(value: &str) -> Option<String> {
    let value = value.trim();
    if value.to_ascii_lowercase().starts_with("::ffff:") {
        return value[7..]
            .parse::<std::net::Ipv4Addr>()
            .ok()
            .map(|address| address.to_string());
    }
    value
        .parse::<std::net::IpAddr>()
        .ok()
        .map(|address| address.to_string().to_ascii_lowercase())
}

fn normalize_ip_whitelist(value: Option<&Value>) -> Result<Option<Vec<String>>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() || value.as_str().is_some_and(|value| value.is_empty()) {
        return Ok(None);
    }
    let candidates: Vec<&str> = if let Some(values) = value.as_array() {
        values.iter().filter_map(Value::as_str).collect()
    } else if let Some(value) = value.as_str() {
        value.split([',', '\n']).collect()
    } else {
        return Err("ipWhitelist must be a list of IP addresses".to_owned());
    };
    let mut addresses = Vec::new();
    for candidate in candidates {
        let candidate = candidate.trim();
        if candidate.is_empty() {
            continue;
        }
        let Some(address) = normalize_ip_address(candidate) else {
            return Err(format!("Invalid IP address: {candidate}"));
        };
        if !addresses.contains(&address) {
            addresses.push(address);
        }
    }
    Ok((!addresses.is_empty()).then_some(addresses))
}

fn permission_list_value(value: Option<&Value>) -> Value {
    value
        .filter(|value| value.is_array())
        .cloned()
        .unwrap_or(Value::Null)
}

fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

fn api_key_value(connection: &Connection, id: &str) -> Result<Option<Value>, String> {
    let row = connection
        .query_row(
            "SELECT id, key, name, machineId, isActive, createdAt,
                    allowedProviders, allowedCombos, allowedKinds,
                    maxInputTokens, maxOutputTokens, requestsPerMinute, ipWhitelist
             FROM apiKeys WHERE id = ?1",
            params![id],
            |row| {
                let parse_list = |index| -> rusqlite::Result<Value> {
                    let raw: Option<String> = row.get(index)?;
                    Ok(raw
                        .as_deref()
                        .and_then(|value| serde_json::from_str(value).ok())
                        .unwrap_or(Value::Null))
                };
                Ok(json!({
                    "id": row.get::<_, String>(0)?,
                    "key": row.get::<_, String>(1)?,
                    "name": row.get::<_, Option<String>>(2)?,
                    "machineId": row.get::<_, Option<String>>(3)?,
                    "isActive": row.get::<_, Option<i64>>(4)?.unwrap_or_default() == 1,
                    "createdAt": row.get::<_, String>(5)?,
                    "allowedProviders": parse_list(6)?,
                    "allowedCombos": parse_list(7)?,
                    "allowedKinds": parse_list(8)?,
                    "maxInputTokens": row.get::<_, Option<i64>>(9)?,
                    "maxOutputTokens": row.get::<_, Option<i64>>(10)?,
                    "requestsPerMinute": row.get::<_, Option<i64>>(11)?,
                    "ipWhitelist": parse_list(12)?,
                }))
            },
        )
        .optional()
        .map_err(|error| error.to_string())?;
    Ok(row)
}

fn api_keys_list_response(connection: &Connection) -> Result<Value, String> {
    let mut statement = connection
        .prepare("SELECT id FROM apiKeys ORDER BY createdAt ASC")
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|error| error.to_string())?;
    let mut keys = Vec::new();
    for row in rows {
        if let Some(key) = api_key_value(connection, &row.map_err(|error| error.to_string())?)? {
            keys.push(key);
        }
    }
    Ok(json!({ "keys": keys }))
}

async fn api_keys_collection(request: Request) -> Response {
    if let Err((status, message)) = require_dashboard_auth(&request) {
        return admin_error(status, message);
    }
    let (parts, body) = request.into_parts();
    match parts.method {
        Method::GET => match open_database_for_admin()
            .and_then(|connection| api_keys_list_response(&connection))
        {
            Ok(value) => Json(value).into_response(),
            Err(error) => {
                eprintln!("[api/keys] list failed: {error}");
                admin_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Backend database unavailable",
                )
            }
        },
        Method::POST => {
            let payload = match parse_admin_json(body).await {
                Ok(Value::Object(payload)) => Value::Object(payload),
                Ok(_) => {
                    return admin_error(
                        StatusCode::BAD_REQUEST,
                        "Request body must be a JSON object",
                    );
                }
                Err((status, message)) => return admin_error(status, message),
            };
            let name = payload
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if name.is_empty() {
                return admin_error(StatusCode::BAD_REQUEST, "Name is required");
            }
            let input_limit = match api_key_limit(payload.get("maxInputTokens"), "maxInputTokens") {
                Ok(value) => value,
                Err(error) => return admin_error(StatusCode::BAD_REQUEST, &error),
            };
            let output_limit =
                match api_key_limit(payload.get("maxOutputTokens"), "maxOutputTokens") {
                    Ok(value) => value,
                    Err(error) => return admin_error(StatusCode::BAD_REQUEST, &error),
                };
            let requests_per_minute =
                match api_key_limit(payload.get("requestsPerMinute"), "requestsPerMinute") {
                    Ok(value) => value,
                    Err(error) => return admin_error(StatusCode::BAD_REQUEST, &error),
                };
            let ip_whitelist = match normalize_ip_whitelist(payload.get("ipWhitelist")) {
                Ok(value) => value,
                Err(error) => return admin_error(StatusCode::BAD_REQUEST, &error),
            };
            let machine_id = match consistent_machine_id() {
                Ok(machine_id) => machine_id,
                Err(error) => {
                    eprintln!("[api/keys] machine identity unavailable: {error}");
                    return admin_error(StatusCode::SERVICE_UNAVAILABLE, &error);
                }
            };
            let secret = match read_or_create_api_key_secret() {
                Ok(secret) => secret,
                Err(error) => {
                    eprintln!("[api/keys] key secret unavailable: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "API key secret unavailable",
                    );
                }
            };
            let key = match generate_api_key(&machine_id, &secret) {
                Ok(key) => key,
                Err(error) => {
                    eprintln!("[api/keys] key generation failed: {error}");
                    return admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "API key could not be generated",
                    );
                }
            };
            let mut id_bytes = [0_u8; 16];
            if let Err(error) = getrandom::getrandom(&mut id_bytes) {
                eprintln!("[api/keys] id generation failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "API key could not be generated",
                );
            }
            let id = hex_encode(&id_bytes);
            let created_at = utc_timestamp();
            let ip_whitelist_json = ip_whitelist.as_ref().map(|value| json!(value).to_string());
            let connection = match open_database_for_admin() {
                Ok(connection) => connection,
                Err(error) => {
                    eprintln!("[api/keys] database unavailable: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            let inserted = connection.execute(
                "INSERT INTO apiKeys
                 (id, key, name, machineId, isActive, createdAt, allowedProviders,
                  allowedCombos, allowedKinds, maxInputTokens, maxOutputTokens,
                  requestsPerMinute, ipWhitelist)
                 VALUES (?1, ?2, ?3, ?4, 1, ?5, NULL, NULL, NULL, ?6, ?7, ?8, ?9)",
                params![
                    id,
                    key,
                    name,
                    machine_id,
                    created_at,
                    input_limit,
                    output_limit,
                    requests_per_minute,
                    ip_whitelist_json
                ],
            );
            if let Err(error) = inserted {
                eprintln!("[api/keys] create failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "API key could not be created",
                );
            }
            (
                StatusCode::CREATED,
                Json(json!({
                    "key": key,
                    "name": name,
                    "id": id,
                    "machineId": machine_id,
                    "maxInputTokens": input_limit,
                    "maxOutputTokens": output_limit,
                    "requestsPerMinute": requests_per_minute,
                    "ipWhitelist": ip_whitelist,
                })),
            )
                .into_response()
        }
        _ => admin_error(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed"),
    }
}

async fn api_key_resource(request: Request, id: String) -> Response {
    if let Err((status, message)) = require_dashboard_auth(&request) {
        return admin_error(status, message);
    }
    let (parts, body) = request.into_parts();
    let mut connection = match open_database_for_admin() {
        Ok(connection) => connection,
        Err(error) => {
            eprintln!("[api/keys/{id}] database unavailable: {error}");
            return admin_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Backend database unavailable",
            );
        }
    };
    match parts.method {
        Method::GET => match api_key_value(&connection, &id) {
            Ok(Some(key)) => Json(json!({ "key": key })).into_response(),
            Ok(None) => admin_error(StatusCode::NOT_FOUND, "Key not found"),
            Err(error) => {
                eprintln!("[api/keys/{id}] read failed: {error}");
                admin_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Backend database unavailable",
                )
            }
        },
        Method::PUT => {
            let payload = match parse_admin_json(body).await {
                Ok(Value::Object(payload)) => Value::Object(payload),
                Ok(_) => {
                    return admin_error(
                        StatusCode::BAD_REQUEST,
                        "Request body must be a JSON object",
                    );
                }
                Err((status, message)) => return admin_error(status, message),
            };
            let existing = match api_key_value(&connection, &id) {
                Ok(Some(key)) => key,
                Ok(None) => return admin_error(StatusCode::NOT_FOUND, "Key not found"),
                Err(error) => {
                    eprintln!("[api/keys/{id}] read failed: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            let name = payload
                .get("name")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map_or_else(
                    || existing["name"].clone(),
                    |name| Value::String(name.to_owned()),
                );
            let is_active = match payload.get("isActive") {
                Some(value) => json_truthy(value),
                None => existing["isActive"].as_bool().unwrap_or(false),
            };
            let permission = |field: &str| {
                if payload.get(field).is_some() {
                    permission_list_value(payload.get(field))
                } else {
                    existing[field].clone()
                }
            };
            let allowed_providers = permission("allowedProviders");
            let allowed_combos = permission("allowedCombos");
            let allowed_kinds = permission("allowedKinds");
            let input_limit = if payload.get("maxInputTokens").is_some() {
                match api_key_limit(payload.get("maxInputTokens"), "maxInputTokens") {
                    Ok(value) => value,
                    Err(error) => return admin_error(StatusCode::BAD_REQUEST, &error),
                }
            } else {
                existing["maxInputTokens"].as_i64()
            };
            let output_limit = if payload.get("maxOutputTokens").is_some() {
                match api_key_limit(payload.get("maxOutputTokens"), "maxOutputTokens") {
                    Ok(value) => value,
                    Err(error) => return admin_error(StatusCode::BAD_REQUEST, &error),
                }
            } else {
                existing["maxOutputTokens"].as_i64()
            };
            let requests_per_minute = if payload.get("requestsPerMinute").is_some() {
                match api_key_limit(payload.get("requestsPerMinute"), "requestsPerMinute") {
                    Ok(value) => value,
                    Err(error) => return admin_error(StatusCode::BAD_REQUEST, &error),
                }
            } else {
                existing["requestsPerMinute"].as_i64()
            };
            let ip_whitelist = if payload.get("ipWhitelist").is_some() {
                match normalize_ip_whitelist(payload.get("ipWhitelist")) {
                    Ok(value) => value,
                    Err(error) => return admin_error(StatusCode::BAD_REQUEST, &error),
                }
            } else {
                existing["ipWhitelist"].as_array().map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
            };
            let serialize_list = |value: &Value| value.as_array().map(|_| value.to_string());
            let ip_whitelist_json = ip_whitelist.as_ref().map(|value| json!(value).to_string());
            if let Err(error) = connection.execute(
                "UPDATE apiKeys SET name = ?1, isActive = ?2, allowedProviders = ?3,
                 allowedCombos = ?4, allowedKinds = ?5, maxInputTokens = ?6,
                 maxOutputTokens = ?7, requestsPerMinute = ?8, ipWhitelist = ?9
                 WHERE id = ?10",
                params![
                    name.as_str().or_else(|| existing["name"].as_str()),
                    is_active,
                    serialize_list(&allowed_providers),
                    serialize_list(&allowed_combos),
                    serialize_list(&allowed_kinds),
                    input_limit,
                    output_limit,
                    requests_per_minute,
                    ip_whitelist_json,
                    id,
                ],
            ) {
                eprintln!("[api/keys/{id}] update failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "API key could not be updated",
                );
            }
            match api_key_value(&connection, &id) {
                Ok(Some(key)) => Json(json!({ "key": key })).into_response(),
                Ok(None) => admin_error(StatusCode::NOT_FOUND, "Key not found"),
                Err(error) => {
                    eprintln!("[api/keys/{id}] updated key read failed: {error}");
                    admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "API key could not be loaded",
                    )
                }
            }
        }
        Method::DELETE => {
            let transaction = match connection.transaction() {
                Ok(transaction) => transaction,
                Err(error) => {
                    eprintln!("[api/keys/{id}] delete transaction failed: {error}");
                    return admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "API key could not be deleted",
                    );
                }
            };
            if let Err(error) = transaction.execute(
                "DELETE FROM apiKeyRateLimits WHERE apiKeyId = ?1",
                params![id],
            ) {
                eprintln!("[api/keys/{id}] rate-limit cleanup failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "API key could not be deleted",
                );
            }
            let deleted =
                match transaction.execute("DELETE FROM apiKeys WHERE id = ?1", params![id]) {
                    Ok(deleted) => deleted,
                    Err(error) => {
                        eprintln!("[api/keys/{id}] delete failed: {error}");
                        return admin_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "API key could not be deleted",
                        );
                    }
                };
            if let Err(error) = transaction.commit() {
                eprintln!("[api/keys/{id}] delete transaction commit failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "API key could not be deleted",
                );
            }
            if deleted == 0 {
                return admin_error(StatusCode::NOT_FOUND, "Key not found");
            }
            Json(json!({ "message": "Key deleted successfully" })).into_response()
        }
        _ => admin_error(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed"),
    }
}

fn kv_get_all(connection: &Connection, scope: &str) -> Result<Value, String> {
    let mut statement = connection
        .prepare("SELECT key, value FROM kv WHERE scope = ?1")
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map(params![scope], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|error| error.to_string())?;
    let mut values = serde_json::Map::new();
    for row in rows {
        let (key, value) = row.map_err(|error| error.to_string())?;
        let value = serde_json::from_str(&value).map_err(|error| error.to_string())?;
        values.insert(key, value);
    }
    Ok(Value::Object(values))
}

fn kv_set(connection: &Connection, scope: &str, key: &str, value: &Value) -> Result<(), String> {
    connection
        .execute(
            "INSERT INTO kv(scope, key, value) VALUES(?1, ?2, ?3)
             ON CONFLICT(scope, key) DO UPDATE SET value = excluded.value",
            params![scope, key, value.to_string()],
        )
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn custom_model_key(provider_alias: &str, id: &str, kind: &str) -> String {
    format!("{provider_alias}|{id}|{kind}")
}

fn sanitize_custom_model(body: &Value, bulk: bool) -> Option<(String, Value)> {
    let provider_alias = body.get("providerAlias")?.as_str()?.to_owned();
    let id = body.get("id")?.as_str()?.to_owned();
    if provider_alias.is_empty() || id.is_empty() {
        return None;
    }
    let kind = body
        .get("type")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or("llm");
    let mut model = serde_json::Map::new();
    model.insert("providerAlias".to_owned(), json!(provider_alias));
    model.insert("id".to_owned(), json!(id));
    model.insert("type".to_owned(), json!(kind));
    if bulk {
        let name = body
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .unwrap_or(&id);
        model.insert("name".to_owned(), json!(name));
    } else if let Some(name) = body
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
    {
        model.insert("name".to_owned(), json!(name));
    }

    if !bulk && let Some(caps) = body.get("caps").and_then(Value::as_object) {
        let mut clean_caps = serde_json::Map::new();
        for key in ["vision", "reasoning"] {
            if let Some(value @ Value::Bool(_)) = caps.get(key) {
                clean_caps.insert(key.to_owned(), value.clone());
            }
        }
        if !clean_caps.is_empty() {
            model.insert("caps".to_owned(), Value::Object(clean_caps));
        }
    }
    if !bulk
        && kind == "stt"
        && body.get("transport").and_then(Value::as_str) == Some("gemini-live")
    {
        model.insert("transport".to_owned(), json!("gemini-live"));
    }
    Some((
        custom_model_key(&provider_alias, &id, kind),
        Value::Object(model),
    ))
}

async fn model_aliases_route(request: Request) -> Response {
    if let Err((status, message)) = require_dashboard_auth(&request) {
        return admin_error(status, message);
    }
    let (parts, body) = request.into_parts();
    match parts.method {
        Method::GET => match open_database_for_admin()
            .and_then(|connection| kv_get_all(&connection, "modelAliases"))
        {
            Ok(aliases) => Json(json!({ "aliases": aliases })).into_response(),
            Err(error) => {
                eprintln!("[api/models/alias] read failed: {error}");
                admin_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Backend database unavailable",
                )
            }
        },
        Method::PUT => {
            let payload = match parse_admin_json(body).await {
                Ok(payload) => payload,
                Err((status, message)) => return admin_error(status, message),
            };
            let model = payload.get("model").and_then(Value::as_str).unwrap_or("");
            let alias = payload.get("alias").and_then(Value::as_str).unwrap_or("");
            if model.is_empty() || alias.is_empty() {
                return admin_error(StatusCode::BAD_REQUEST, "Model and alias required");
            }
            let connection = match open_database_for_admin() {
                Ok(connection) => connection,
                Err(error) => {
                    eprintln!("[api/models/alias] database unavailable: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            if let Err(error) = kv_set(&connection, "modelAliases", alias, &json!(model)) {
                eprintln!("[api/models/alias] write failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Alias could not be saved",
                );
            }
            Json(json!({ "success": true, "model": model, "alias": alias })).into_response()
        }
        Method::DELETE => {
            let query = match parts.uri.query() {
                Some(query) => query,
                None => return admin_error(StatusCode::BAD_REQUEST, "Alias required"),
            };
            let alias = url::form_urlencoded::parse(query.as_bytes())
                .find_map(|(key, value)| (key == "alias").then(|| value.into_owned()));
            let Some(alias) = alias.filter(|alias| !alias.is_empty()) else {
                return admin_error(StatusCode::BAD_REQUEST, "Alias required");
            };
            let connection = match open_database_for_admin() {
                Ok(connection) => connection,
                Err(error) => {
                    eprintln!("[api/models/alias] database unavailable: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            if let Err(error) = connection.execute(
                "DELETE FROM kv WHERE scope = 'modelAliases' AND key = ?1",
                params![alias],
            ) {
                eprintln!("[api/models/alias] delete failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Alias could not be deleted",
                );
            }
            Json(json!({ "success": true })).into_response()
        }
        _ => admin_error(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed"),
    }
}

async fn custom_models_route(request: Request) -> Response {
    if let Err((status, message)) = require_dashboard_auth(&request) {
        return admin_error(status, message);
    }
    let (parts, body) = request.into_parts();
    match parts.method {
        Method::GET => {
            let connection = match open_database_for_admin() {
                Ok(connection) => connection,
                Err(error) => {
                    eprintln!("[api/models/custom] database unavailable: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            match kv_get_all(&connection, "customModels") {
                Ok(Value::Object(models)) => {
                    Json(json!({ "models": models.into_values().collect::<Vec<_>>() }))
                        .into_response()
                }
                Ok(_) => Json(json!({ "models": [] })).into_response(),
                Err(error) => {
                    eprintln!("[api/models/custom] read failed: {error}");
                    admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    )
                }
            }
        }
        Method::POST => {
            let payload = match parse_admin_json(body).await {
                Ok(payload) => payload,
                Err((status, message)) => return admin_error(status, message),
            };
            let connection = match open_database_for_admin() {
                Ok(connection) => connection,
                Err(error) => {
                    eprintln!("[api/models/custom] database unavailable: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            if let Some(models) = payload.get("models").and_then(Value::as_array) {
                let transaction = match connection.unchecked_transaction() {
                    Ok(transaction) => transaction,
                    Err(error) => {
                        eprintln!("[api/models/custom] transaction failed: {error}");
                        return admin_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "Models could not be added",
                        );
                    }
                };
                let mut added = 0;
                for item in models {
                    let Some((key, model)) = sanitize_custom_model(item, true) else {
                        continue;
                    };
                    let exists = transaction
                        .query_row(
                            "SELECT 1 FROM kv WHERE scope = 'customModels' AND key = ?1",
                            params![key],
                            |_| Ok(()),
                        )
                        .optional();
                    match exists {
                        Ok(Some(())) => continue,
                        Ok(None) => {}
                        Err(error) => {
                            eprintln!("[api/models/custom] duplicate lookup failed: {error}");
                            return admin_error(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                "Models could not be added",
                            );
                        }
                    }
                    if let Err(error) = kv_set(&transaction, "customModels", &key, &model) {
                        eprintln!("[api/models/custom] bulk insert failed: {error}");
                        return admin_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "Models could not be added",
                        );
                    }
                    added += 1;
                }
                if let Err(error) = transaction.commit() {
                    eprintln!("[api/models/custom] transaction commit failed: {error}");
                    return admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Models could not be added",
                    );
                }
                return Json(json!({ "success": true, "count": added })).into_response();
            }
            let Some((key, mut model)) = sanitize_custom_model(&payload, false) else {
                return admin_error(StatusCode::BAD_REQUEST, "providerAlias and id required");
            };
            let existing_raw = connection
                .query_row(
                    "SELECT value FROM kv WHERE scope = 'customModels' AND key = ?1",
                    params![key],
                    |row| row.get::<_, String>(0),
                )
                .optional();
            let previous = match existing_raw {
                Ok(Some(raw)) => serde_json::from_str::<Value>(&raw).ok(),
                Ok(None) => None,
                Err(error) => {
                    eprintln!("[api/models/custom] lookup failed: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            let added = previous.is_none();
            if let Some(previous) = previous.and_then(|value| value.as_object().cloned())
                && let Some(next) = model.as_object_mut()
            {
                for (field, value) in previous {
                    next.entry(field).or_insert(value);
                }
            }
            if let Some(next) = model.as_object_mut() {
                let fallback_name = next.get("id").cloned().unwrap_or(Value::Null);
                next.entry("name").or_insert(fallback_name);
            }
            if let Err(error) = kv_set(&connection, "customModels", &key, &model) {
                eprintln!("[api/models/custom] upsert failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Model could not be added",
                );
            }
            Json(json!({ "success": true, "added": added })).into_response()
        }
        Method::DELETE => {
            let query = parts.uri.query().unwrap_or_default();
            let values: HashMap<_, _> = url::form_urlencoded::parse(query.as_bytes())
                .into_owned()
                .collect();
            let provider_alias = values
                .get("providerAlias")
                .map(String::as_str)
                .unwrap_or("");
            let id = values.get("id").map(String::as_str).unwrap_or("");
            let kind = values.get("type").map(String::as_str).unwrap_or("llm");
            if provider_alias.is_empty() || id.is_empty() {
                return admin_error(StatusCode::BAD_REQUEST, "providerAlias and id required");
            }
            let key = custom_model_key(provider_alias, id, kind);
            let connection = match open_database_for_admin() {
                Ok(connection) => connection,
                Err(error) => {
                    eprintln!("[api/models/custom] database unavailable: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            if let Err(error) = connection.execute(
                "DELETE FROM kv WHERE scope = 'customModels' AND key = ?1",
                params![key],
            ) {
                eprintln!("[api/models/custom] delete failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Model could not be deleted",
                );
            }
            Json(json!({ "success": true })).into_response()
        }
        _ => admin_error(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed"),
    }
}

async fn disabled_models_route(request: Request) -> Response {
    if let Err((status, message)) = require_dashboard_auth(&request) {
        return admin_error(status, message);
    }
    let (parts, body) = request.into_parts();
    match parts.method {
        Method::GET => {
            let query = parts.uri.query().unwrap_or_default();
            let provider_alias = url::form_urlencoded::parse(query.as_bytes())
                .find_map(|(key, value)| (key == "providerAlias").then(|| value.into_owned()));
            let connection = match open_database_for_admin() {
                Ok(connection) => connection,
                Err(error) => {
                    eprintln!("[api/models/disabled] database unavailable: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            match kv_get_all(&connection, "disabledModels") {
                Ok(Value::Object(mut disabled)) => {
                    if let Some(provider_alias) = provider_alias {
                        Json(json!({ "ids": disabled.remove(&provider_alias).unwrap_or_else(|| json!([])) })).into_response()
                    } else {
                        Json(json!({ "disabled": disabled })).into_response()
                    }
                }
                Ok(_) => Json(json!({ "disabled": {} })).into_response(),
                Err(error) => {
                    eprintln!("[api/models/disabled] read failed: {error}");
                    admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    )
                }
            }
        }
        Method::POST => {
            let payload = match parse_admin_json(body).await {
                Ok(payload) => payload,
                Err((status, message)) => return admin_error(status, message),
            };
            let provider_alias = payload
                .get("providerAlias")
                .and_then(Value::as_str)
                .unwrap_or("");
            let Some(ids) = payload.get("ids").and_then(Value::as_array) else {
                return admin_error(StatusCode::BAD_REQUEST, "providerAlias and ids[] required");
            };
            if provider_alias.is_empty() {
                return admin_error(StatusCode::BAD_REQUEST, "providerAlias and ids[] required");
            }
            let connection = match open_database_for_admin() {
                Ok(connection) => connection,
                Err(error) => {
                    eprintln!("[api/models/disabled] database unavailable: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            let transaction = match connection.unchecked_transaction() {
                Ok(transaction) => transaction,
                Err(error) => {
                    eprintln!("[api/models/disabled] transaction failed: {error}");
                    return admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Models could not be disabled",
                    );
                }
            };
            let current = transaction
                .query_row(
                    "SELECT value FROM kv WHERE scope = 'disabledModels' AND key = ?1",
                    params![provider_alias],
                    |row| row.get::<_, String>(0),
                )
                .optional();
            let mut merged = match current {
                Ok(Some(value)) => serde_json::from_str::<Vec<Value>>(&value).unwrap_or_default(),
                Ok(None) => Vec::new(),
                Err(error) => {
                    eprintln!("[api/models/disabled] lookup failed: {error}");
                    return admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Models could not be disabled",
                    );
                }
            };
            for id in ids {
                if !merged.contains(id) {
                    merged.push(id.clone());
                }
            }
            if let Err(error) = kv_set(
                &transaction,
                "disabledModels",
                provider_alias,
                &json!(merged),
            ) {
                eprintln!("[api/models/disabled] write failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Models could not be disabled",
                );
            }
            if let Err(error) = transaction.commit() {
                eprintln!("[api/models/disabled] commit failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Models could not be disabled",
                );
            }
            Json(json!({ "success": true })).into_response()
        }
        Method::DELETE => {
            let query = parts.uri.query().unwrap_or_default();
            let values: HashMap<_, _> = url::form_urlencoded::parse(query.as_bytes())
                .into_owned()
                .collect();
            let provider_alias = values
                .get("providerAlias")
                .map(String::as_str)
                .unwrap_or("");
            if provider_alias.is_empty() {
                return admin_error(StatusCode::BAD_REQUEST, "providerAlias required");
            }
            let connection = match open_database_for_admin() {
                Ok(connection) => connection,
                Err(error) => {
                    eprintln!("[api/models/disabled] database unavailable: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            let Some(id) = values.get("id").filter(|id| !id.is_empty()) else {
                if let Err(error) = connection.execute(
                    "DELETE FROM kv WHERE scope = 'disabledModels' AND key = ?1",
                    params![provider_alias],
                ) {
                    eprintln!("[api/models/disabled] clear failed: {error}");
                    return admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Models could not be enabled",
                    );
                }
                return Json(json!({ "success": true })).into_response();
            };
            let transaction = match connection.unchecked_transaction() {
                Ok(transaction) => transaction,
                Err(error) => {
                    eprintln!("[api/models/disabled] transaction failed: {error}");
                    return admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Models could not be enabled",
                    );
                }
            };
            let current = transaction
                .query_row(
                    "SELECT value FROM kv WHERE scope = 'disabledModels' AND key = ?1",
                    params![provider_alias],
                    |row| row.get::<_, String>(0),
                )
                .optional();
            let mut ids = match current {
                Ok(Some(value)) => serde_json::from_str::<Vec<Value>>(&value).unwrap_or_default(),
                Ok(None) => Vec::new(),
                Err(error) => {
                    eprintln!("[api/models/disabled] lookup failed: {error}");
                    return admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Models could not be enabled",
                    );
                }
            };
            ids.retain(|value| value != id);
            let result = if ids.is_empty() {
                transaction.execute(
                    "DELETE FROM kv WHERE scope = 'disabledModels' AND key = ?1",
                    params![provider_alias],
                )
            } else {
                transaction.execute(
                    "INSERT INTO kv(scope, key, value) VALUES('disabledModels', ?1, ?2)
                     ON CONFLICT(scope, key) DO UPDATE SET value = excluded.value",
                    params![provider_alias, json!(ids).to_string()],
                )
            };
            if let Err(error) = result {
                eprintln!("[api/models/disabled] update failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Models could not be enabled",
                );
            }
            if let Err(error) = transaction.commit() {
                eprintln!("[api/models/disabled] commit failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Models could not be enabled",
                );
            }
            Json(json!({ "success": true })).into_response()
        }
        _ => admin_error(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed"),
    }
}

fn generate_uuid_v4() -> Result<String, String> {
    let mut bytes = [0_u8; 16];
    getrandom::getrandom(&mut bytes).map_err(|error| error.to_string())?;
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let value = hex_encode(&bytes);
    Ok(format!(
        "{}-{}-{}-{}-{}",
        &value[..8],
        &value[8..12],
        &value[12..16],
        &value[16..20],
        &value[20..]
    ))
}

fn sanitize_provider_node_base_url(node_type: &str, base_url: &str) -> String {
    let mut base_url = base_url.to_owned();
    if node_type == "anthropic-compatible" {
        if base_url.ends_with('/') {
            base_url.pop();
        }
        if base_url.ends_with("/messages") {
            base_url.truncate(base_url.len() - "/messages".len());
        }
        if !base_url.ends_with("/v1") {
            base_url.push_str("/v1");
        }
    } else if node_type == "custom-embedding" {
        if base_url.ends_with('/') {
            base_url.pop();
        }
        if base_url.ends_with("/embeddings") {
            base_url.truncate(base_url.len() - "/embeddings".len());
        }
    }
    base_url
}

fn provider_node_value(connection: &Connection, id: &str) -> Result<Option<Value>, String> {
    let row = connection
        .query_row(
            "SELECT id, type, name, data, createdAt, updatedAt FROM providerNodes WHERE id = ?1",
            params![id],
            |row| {
                let raw_data = row.get::<_, String>(3)?;
                let mut value = serde_json::from_str::<Value>(&raw_data)
                    .ok()
                    .and_then(|value| value.as_object().cloned())
                    .unwrap_or_default();
                value.insert("id".to_owned(), json!(row.get::<_, String>(0)?));
                value.insert(
                    "type".to_owned(),
                    row.get::<_, Option<String>>(1)?
                        .map_or(Value::Null, Value::String),
                );
                value.insert(
                    "name".to_owned(),
                    row.get::<_, Option<String>>(2)?
                        .map_or(Value::Null, Value::String),
                );
                value.insert("createdAt".to_owned(), json!(row.get::<_, String>(4)?));
                value.insert("updatedAt".to_owned(), json!(row.get::<_, String>(5)?));
                Ok(Value::Object(value))
            },
        )
        .optional()
        .map_err(|error| error.to_string())?;
    Ok(row)
}

fn provider_nodes_list_response(connection: &Connection) -> Result<Value, String> {
    let mut statement = connection
        .prepare("SELECT id FROM providerNodes")
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|error| error.to_string())?;
    let mut nodes = Vec::new();
    for row in rows {
        if let Some(node) =
            provider_node_value(connection, &row.map_err(|error| error.to_string())?)?
        {
            nodes.push(node);
        }
    }
    Ok(json!({ "nodes": nodes }))
}

async fn provider_nodes_collection(request: Request) -> Response {
    if let Err((status, message)) = require_dashboard_auth(&request) {
        return admin_error(status, message);
    }
    let (parts, body) = request.into_parts();
    match parts.method {
        Method::GET => {
            match open_database_for_admin()
                .and_then(|connection| provider_nodes_list_response(&connection))
            {
                Ok(value) => Json(value).into_response(),
                Err(error) => {
                    eprintln!("[api/provider-nodes] list failed: {error}");
                    admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    )
                }
            }
        }
        Method::POST => {
            let payload = match parse_admin_json(body).await {
                Ok(Value::Object(payload)) => Value::Object(payload),
                Ok(_) => {
                    return admin_error(
                        StatusCode::BAD_REQUEST,
                        "Request body must be a JSON object",
                    );
                }
                Err((status, message)) => return admin_error(status, message),
            };
            let name = payload
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if name.is_empty() {
                return admin_error(StatusCode::BAD_REQUEST, "Name is required");
            }
            let prefix = payload
                .get("prefix")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if prefix.is_empty() {
                return admin_error(StatusCode::BAD_REQUEST, "Prefix is required");
            }
            let node_type = payload
                .get("type")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .unwrap_or("openai-compatible");
            let api_type = payload.get("apiType").and_then(Value::as_str);
            let (id_prefix, default_base_url) = match node_type {
                "openai-compatible" => {
                    let Some(api_type) =
                        api_type.filter(|value| matches!(*value, "chat" | "responses"))
                    else {
                        return admin_error(
                            StatusCode::BAD_REQUEST,
                            "Invalid OpenAI compatible API type",
                        );
                    };
                    (
                        format!("openai-compatible-{api_type}-"),
                        "https://api.openai.com/v1",
                    )
                }
                "anthropic-compatible" => (
                    "anthropic-compatible-".to_owned(),
                    "https://api.anthropic.com/v1",
                ),
                "custom-embedding" => ("custom-embedding-".to_owned(), "https://api.openai.com/v1"),
                _ => return admin_error(StatusCode::BAD_REQUEST, "Invalid provider node type"),
            };
            let raw_base_url = match payload.get("baseUrl") {
                None | Some(Value::Null) => default_base_url,
                Some(Value::String(value)) if value.is_empty() => default_base_url,
                Some(Value::String(value)) => value.as_str(),
                Some(_) => {
                    return admin_error(StatusCode::BAD_REQUEST, "Base URL must be a string");
                }
            };
            let base_url = sanitize_provider_node_base_url(node_type, raw_base_url.trim());
            let uuid = match generate_uuid_v4() {
                Ok(uuid) => uuid,
                Err(error) => {
                    eprintln!("[api/provider-nodes] id generation failed: {error}");
                    return admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Provider node could not be created",
                    );
                }
            };
            let id = format!("{id_prefix}{uuid}");
            let now = utc_timestamp();
            let mut data = serde_json::Map::new();
            data.insert("prefix".to_owned(), json!(prefix));
            data.insert("baseUrl".to_owned(), json!(base_url));
            if node_type == "openai-compatible" {
                data.insert("apiType".to_owned(), json!(api_type));
            }
            let connection = match open_database_for_admin() {
                Ok(connection) => connection,
                Err(error) => {
                    eprintln!("[api/provider-nodes] database unavailable: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            if let Err(error) = connection.execute(
                "INSERT INTO providerNodes (id, type, name, data, createdAt, updatedAt)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
                params![id, node_type, name, Value::Object(data).to_string(), now],
            ) {
                eprintln!("[api/provider-nodes] create failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Provider node could not be created",
                );
            }
            match provider_node_value(&connection, &id) {
                Ok(Some(node)) => {
                    (StatusCode::CREATED, Json(json!({ "node": node }))).into_response()
                }
                Ok(None) => admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Created provider node could not be loaded",
                ),
                Err(error) => {
                    eprintln!("[api/provider-nodes] created node read failed: {error}");
                    admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Created provider node could not be loaded",
                    )
                }
            }
        }
        _ => admin_error(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed"),
    }
}

async fn provider_node_resource(request: Request, id: String) -> Response {
    if let Err((status, message)) = require_dashboard_auth(&request) {
        return admin_error(status, message);
    }
    let (parts, body) = request.into_parts();
    match parts.method {
        Method::PUT => {
            let payload = match parse_admin_json(body).await {
                Ok(Value::Object(payload)) => Value::Object(payload),
                Ok(_) => {
                    return admin_error(
                        StatusCode::BAD_REQUEST,
                        "Request body must be a JSON object",
                    );
                }
                Err((status, message)) => return admin_error(status, message),
            };
            let mut connection = match open_database_for_admin() {
                Ok(connection) => connection,
                Err(error) => {
                    eprintln!("[api/provider-nodes/{id}] database unavailable: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            let current = match connection.query_row(
                "SELECT type, data FROM providerNodes WHERE id = ?1",
                params![id],
                |row| Ok((row.get::<_, Option<String>>(0)?, row.get::<_, String>(1)?)),
            ) {
                Ok(current) => current,
                Err(rusqlite::Error::QueryReturnedNoRows) => {
                    return admin_error(StatusCode::NOT_FOUND, "Provider node not found");
                }
                Err(error) => {
                    eprintln!("[api/provider-nodes/{id}] lookup failed: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            let node_type = current.0.as_deref().unwrap_or("");
            let name = payload
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if name.is_empty() {
                return admin_error(StatusCode::BAD_REQUEST, "Name is required");
            }
            let prefix = payload
                .get("prefix")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if prefix.is_empty() {
                return admin_error(StatusCode::BAD_REQUEST, "Prefix is required");
            }
            let api_type = payload.get("apiType").and_then(Value::as_str);
            if node_type == "openai-compatible" && !matches!(api_type, Some("chat" | "responses")) {
                return admin_error(
                    StatusCode::BAD_REQUEST,
                    "Invalid OpenAI compatible API type",
                );
            }
            let base_url = payload
                .get("baseUrl")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if base_url.is_empty() {
                return admin_error(StatusCode::BAD_REQUEST, "Base URL is required");
            }
            let base_url = sanitize_provider_node_base_url(node_type, base_url);
            let mut node_data = serde_json::from_str::<Value>(&current.1)
                .ok()
                .and_then(|value| value.as_object().cloned())
                .unwrap_or_default();
            node_data.insert("prefix".to_owned(), json!(prefix));
            node_data.insert("baseUrl".to_owned(), json!(base_url));
            if node_type == "openai-compatible" {
                node_data.insert("apiType".to_owned(), json!(api_type));
            } else {
                node_data.remove("apiType");
            }
            let now = utc_timestamp();
            let transaction = match connection.transaction() {
                Ok(transaction) => transaction,
                Err(error) => {
                    eprintln!("[api/provider-nodes/{id}] transaction failed: {error}");
                    return admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Provider node could not be updated",
                    );
                }
            };
            if let Err(error) = transaction.execute(
                "UPDATE providerNodes SET name = ?1, data = ?2, updatedAt = ?3 WHERE id = ?4",
                params![name, Value::Object(node_data).to_string(), now, id],
            ) {
                eprintln!("[api/provider-nodes/{id}] update failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Provider node could not be updated",
                );
            }
            let mut statement = match transaction
                .prepare("SELECT id, data FROM providerConnections WHERE provider = ?1")
            {
                Ok(statement) => statement,
                Err(error) => {
                    eprintln!("[api/provider-nodes/{id}] connection query failed: {error}");
                    return admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Provider node connections could not be updated",
                    );
                }
            };
            let rows = match statement.query_map(params![id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            }) {
                Ok(rows) => rows,
                Err(error) => {
                    eprintln!("[api/provider-nodes/{id}] connection query failed: {error}");
                    return admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Provider node connections could not be updated",
                    );
                }
            };
            let mut connection_rows = Vec::new();
            for row in rows {
                match row {
                    Ok(row) => connection_rows.push(row),
                    Err(error) => {
                        eprintln!("[api/provider-nodes/{id}] connection row failed: {error}");
                        return admin_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "Provider node connections could not be updated",
                        );
                    }
                }
            }
            drop(statement);
            for (connection_id, raw_data) in connection_rows {
                let mut connection_data = serde_json::from_str::<Value>(&raw_data)
                    .ok()
                    .and_then(|value| value.as_object().cloned())
                    .unwrap_or_default();
                let provider_specific_data = connection_data
                    .entry("providerSpecificData".to_owned())
                    .or_insert_with(|| Value::Object(serde_json::Map::new()));
                let Some(provider_specific_data) = provider_specific_data.as_object_mut() else {
                    return admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Provider node connections could not be updated",
                    );
                };
                provider_specific_data.insert("prefix".to_owned(), json!(prefix));
                provider_specific_data.insert("baseUrl".to_owned(), json!(base_url));
                provider_specific_data.insert("nodeName".to_owned(), json!(name));
                if node_type == "openai-compatible" {
                    provider_specific_data.insert("apiType".to_owned(), json!(api_type));
                } else {
                    provider_specific_data.remove("apiType");
                }
                if let Err(error) = transaction.execute(
                    "UPDATE providerConnections SET data = ?1, updatedAt = ?2 WHERE id = ?3",
                    params![
                        Value::Object(connection_data).to_string(),
                        now,
                        connection_id
                    ],
                ) {
                    eprintln!("[api/provider-nodes/{id}] connection update failed: {error}");
                    return admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Provider node connections could not be updated",
                    );
                }
            }
            if let Err(error) = transaction.commit() {
                eprintln!("[api/provider-nodes/{id}] transaction commit failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Provider node could not be updated",
                );
            }
            match provider_node_value(&connection, &id) {
                Ok(Some(node)) => Json(json!({ "node": node })).into_response(),
                Ok(None) => admin_error(StatusCode::NOT_FOUND, "Provider node not found"),
                Err(error) => {
                    eprintln!("[api/provider-nodes/{id}] updated node read failed: {error}");
                    admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Provider node could not be loaded",
                    )
                }
            }
        }
        Method::DELETE => {
            let mut connection = match open_database_for_admin() {
                Ok(connection) => connection,
                Err(error) => {
                    eprintln!("[api/provider-nodes/{id}] database unavailable: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            match provider_node_value(&connection, &id) {
                Ok(Some(_)) => {}
                Ok(None) => return admin_error(StatusCode::NOT_FOUND, "Provider node not found"),
                Err(error) => {
                    eprintln!("[api/provider-nodes/{id}] lookup failed: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            let transaction = match connection.transaction() {
                Ok(transaction) => transaction,
                Err(error) => {
                    eprintln!("[api/provider-nodes/{id}] delete transaction failed: {error}");
                    return admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Provider node could not be deleted",
                    );
                }
            };
            if let Err(error) = transaction.execute(
                "DELETE FROM providerConnections WHERE provider = ?1",
                params![id],
            ) {
                eprintln!("[api/provider-nodes/{id}] connection delete failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Provider node could not be deleted",
                );
            }
            if let Err(error) =
                transaction.execute("DELETE FROM providerNodes WHERE id = ?1", params![id])
            {
                eprintln!("[api/provider-nodes/{id}] delete failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Provider node could not be deleted",
                );
            }
            if let Err(error) = transaction.commit() {
                eprintln!("[api/provider-nodes/{id}] delete commit failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Provider node could not be deleted",
                );
            }
            Json(json!({ "success": true })).into_response()
        }
        _ => admin_error(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed"),
    }
}

fn provider_connection_value(connection: &Connection, id: &str) -> Result<Option<Value>, String> {
    let row = connection
        .query_row(
            "SELECT id, provider, authType, name, email, priority, isActive, data, createdAt, updatedAt FROM providerConnections WHERE id = ?1",
            params![id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, Option<i64>>(5)?,
                    row.get::<_, Option<i64>>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, String>(8)?,
                    row.get::<_, String>(9)?,
                ))
            },
        )
        .optional()
        .map_err(|error| error.to_string())?;
    Ok(row.map(
        |(id, provider, auth_type, name, email, priority, active, raw_data, created, updated)| {
            let mut value = serde_json::from_str::<Value>(&raw_data)
                .ok()
                .and_then(|value| value.as_object().cloned())
                .unwrap_or_default();
            for secret in ["apiKey", "accessToken", "refreshToken", "idToken"] {
                value.remove(secret);
            }
            value.insert("id".to_owned(), Value::String(id));
            value.insert("provider".to_owned(), Value::String(provider.clone()));
            value.insert("authType".to_owned(), Value::String(auth_type));
            value.insert(
                "alias".to_owned(),
                model_catalog()
                    .providers
                    .iter()
                    .find(|entry| entry.id == provider)
                    .map(|entry| Value::String(entry.alias.clone()))
                    .unwrap_or(Value::Null),
            );
            value.insert("name".to_owned(), name.map_or(Value::Null, Value::String));
            value.insert("email".to_owned(), email.map_or(Value::Null, Value::String));
            value.insert(
                "priority".to_owned(),
                priority.map_or(Value::Null, Value::from),
            );
            value.insert("isActive".to_owned(), Value::Bool(active.unwrap_or(0) != 0));
            value.insert("createdAt".to_owned(), Value::String(created));
            value.insert("updatedAt".to_owned(), Value::String(updated));
            Value::Object(value)
        },
    ))
}

fn provider_list_response(connection: &Connection) -> Result<Value, String> {
    let mut statement = connection
        .prepare("SELECT id FROM providerConnections ORDER BY provider, priority, createdAt")
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|error| error.to_string())?;
    let mut connections = Vec::new();
    for row in rows {
        let id = row.map_err(|error| error.to_string())?;
        if let Some(value) = provider_connection_value(connection, &id)? {
            connections.push(value);
        }
    }

    let mut providers = Vec::new();
    let mut alias_map = serde_json::Map::new();
    for provider in &model_catalog().providers {
        providers.push(json!({
            "id": provider.id,
            "alias": provider.alias,
            "aliases": provider.aliases,
            "displayName": provider.display_name,
            "noAuth": provider.no_auth,
            "serviceKinds": provider.service_kinds,
        }));
        for alias in std::iter::once(&provider.alias).chain(provider.aliases.iter()) {
            if alias != &provider.id {
                alias_map.insert(alias.clone(), Value::String(provider.id.clone()));
            }
        }
    }
    Ok(json!({
        "connections": connections,
        "aliasMap": Value::Object(alias_map),
        "providers": providers,
    }))
}

fn provider_definition(provider_id: &str) -> Option<&CatalogProvider> {
    let normalized = provider_id.trim();
    let slug = normalized
        .to_ascii_lowercase()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    model_catalog().providers.iter().find(|provider| {
        provider.id == normalized
            || provider.alias == normalized
            || provider.aliases.iter().any(|alias| alias == normalized)
            || provider.id == slug
            || provider.alias == slug
            || provider.aliases.iter().any(|alias| alias == &slug)
            || provider.display_name.eq_ignore_ascii_case(normalized)
    })
}

fn provider_node_definition(
    connection: &Connection,
    provider_id: &str,
) -> Result<Option<(String, String, Value)>, String> {
    connection
        .query_row(
            "SELECT type, name, data FROM providerNodes WHERE id = ?1",
            params![provider_id],
            |row| {
                Ok((
                    row.get::<_, Option<String>>(0)?.unwrap_or_default(),
                    row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()
        .map_err(|error| error.to_string())?
        .map(|(kind, name, raw_data)| {
            serde_json::from_str::<Value>(&raw_data)
                .map(|data| (kind, name, data))
                .map_err(|error| error.to_string())
        })
        .transpose()
}

fn reorder_provider_priorities(connection: &mut Connection, provider: &str) -> Result<(), String> {
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    let mut statement = transaction
        .prepare("SELECT id FROM providerConnections WHERE provider = ?1 ORDER BY priority, updatedAt DESC")
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map(params![provider], |row| row.get::<_, String>(0))
        .map_err(|error| error.to_string())?;
    let mut ids = Vec::new();
    for row in rows {
        ids.push(row.map_err(|error| error.to_string())?);
    }
    drop(statement);
    for (index, id) in ids.iter().enumerate() {
        transaction
            .execute(
                "UPDATE providerConnections SET priority = ?1 WHERE id = ?2",
                params![index as i64 + 1, id],
            )
            .map_err(|error| error.to_string())?;
    }
    transaction.commit().map_err(|error| error.to_string())
}

async fn providers_collection(request: Request) -> Response {
    if let Err((status, message)) = require_dashboard_auth(&request) {
        return admin_error(status, message);
    }
    let (parts, body) = request.into_parts();
    match parts.method {
        Method::GET => {
            match open_database().and_then(|connection| provider_list_response(&connection)) {
                Ok(value) => Json(value).into_response(),
                Err(error) => {
                    eprintln!("[api/providers] database query failed: {error}");
                    admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    )
                }
            }
        }
        Method::POST => {
            let payload = match parse_admin_json(body).await {
                Ok(payload) => payload,
                Err((status, message)) => return admin_error(status, message),
            };
            let mut connection = match open_database_for_admin() {
                Ok(connection) => connection,
                Err(error) => {
                    eprintln!("[api/providers] database unavailable: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            let provider_input = payload
                .get("provider")
                .and_then(Value::as_str)
                .unwrap_or("");
            let catalog_provider = provider_definition(provider_input);
            let provider_node = if catalog_provider.is_none() {
                match provider_node_definition(&connection, provider_input.trim()) {
                    Ok(node) => node,
                    Err(error) => {
                        eprintln!("[api/providers] custom provider lookup failed: {error}");
                        return admin_error(
                            StatusCode::SERVICE_UNAVAILABLE,
                            "Backend database unavailable",
                        );
                    }
                }
            } else {
                None
            };
            if catalog_provider.is_none() && provider_node.is_none() {
                return admin_error(StatusCode::BAD_REQUEST, "Invalid provider");
            }
            let provider_id = catalog_provider
                .map(|provider| provider.id.as_str())
                .unwrap_or_else(|| provider_input.trim());
            if let Some(provider) = catalog_provider
                && provider.category.as_deref() != Some("apikey")
                && !provider.auth_modes.iter().any(|mode| mode == "apikey")
                && !provider.no_auth
            {
                return admin_error(
                    StatusCode::NOT_IMPLEMENTED,
                    "This provider requires a Node-managed auth flow and is not supported by Rust CRUD",
                );
            }
            let api_key = payload.get("apiKey").and_then(Value::as_str).unwrap_or("");
            if api_key.is_empty() && provider_id != "ollama-local" {
                return admin_error(StatusCode::BAD_REQUEST, "API Key is required");
            }
            let fallback_name = catalog_provider
                .map(|provider| provider.display_name.as_str())
                .or_else(|| provider_node.as_ref().map(|(_, name, _)| name.as_str()))
                .unwrap_or(provider_id);
            let name = payload
                .get("name")
                .or_else(|| payload.get("displayName"))
                .and_then(Value::as_str)
                .unwrap_or(fallback_name)
                .trim();
            if name.is_empty() {
                return admin_error(StatusCode::BAD_REQUEST, "Name is required");
            }
            let priority = match payload.get("priority") {
                None | Some(Value::Null) => 1,
                Some(value) => match value.as_i64().filter(|priority| *priority > 0) {
                    Some(priority) => priority,
                    None => {
                        return admin_error(
                            StatusCode::BAD_REQUEST,
                            "priority must be a positive integer",
                        );
                    }
                },
            };
            let mut data = serde_json::Map::new();
            data.insert("apiKey".to_owned(), Value::String(api_key.to_owned()));
            data.insert(
                "testStatus".to_owned(),
                payload
                    .get("testStatus")
                    .cloned()
                    .unwrap_or_else(|| Value::String("unknown".to_owned())),
            );
            for field in ["globalPriority", "defaultModel"] {
                if let Some(value) = payload
                    .get(field)
                    .filter(|value| !value.is_null() && !value.as_str().is_some_and(str::is_empty))
                {
                    data.insert(field.to_owned(), value.clone());
                }
            }
            let provider_specific_data = match payload.get("providerSpecificData") {
                Some(Value::Object(data)) => data.clone(),
                Some(_) => {
                    return admin_error(
                        StatusCode::BAD_REQUEST,
                        "providerSpecificData must be an object",
                    );
                }
                None => serde_json::Map::new(),
            };
            let mut provider_specific_data = provider_specific_data;
            if provider_id == "ollama-local" {
                for field in ["baseUrl", "baseURL", "ollamaHostUrl"] {
                    if let Some(base_url) = payload.get(field).and_then(Value::as_str) {
                        let base_url = base_url.trim();
                        if !base_url.is_empty() {
                            provider_specific_data
                                .insert("baseUrl".to_owned(), Value::String(base_url.to_owned()));
                            break;
                        }
                    }
                }
            }
            if let Some((_, node_name, node_data)) = provider_node.as_ref() {
                for field in ["prefix", "apiType", "baseUrl"] {
                    if let Some(value) = node_data.get(field) {
                        provider_specific_data.insert(field.to_owned(), value.clone());
                    }
                }
                provider_specific_data
                    .insert("nodeName".to_owned(), Value::String(node_name.clone()));
            }
            let connection_proxy_enabled =
                payload.get("connectionProxyEnabled") == Some(&Value::Bool(true));
            let connection_proxy_url = payload
                .get("connectionProxyUrl")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            let connection_no_proxy = payload
                .get("connectionNoProxy")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if connection_proxy_enabled && connection_proxy_url.is_empty() {
                return admin_error(
                    StatusCode::BAD_REQUEST,
                    "Connection proxy URL is required when connection proxy is enabled",
                );
            }
            provider_specific_data.insert(
                "connectionProxyEnabled".to_owned(),
                Value::Bool(connection_proxy_enabled),
            );
            provider_specific_data.insert(
                "connectionProxyUrl".to_owned(),
                Value::String(connection_proxy_url.to_owned()),
            );
            provider_specific_data.insert(
                "connectionNoProxy".to_owned(),
                Value::String(connection_no_proxy.to_owned()),
            );
            let pool_id = match payload.get("proxyPoolId") {
                None | Some(Value::Null) => None,
                Some(Value::String(id)) => {
                    let id = id.trim();
                    (!id.is_empty() && id != "__none__").then_some(id)
                }
                Some(_) => {
                    return admin_error(StatusCode::BAD_REQUEST, "Invalid proxy pool id");
                }
            };
            if let Some(pool_id) = pool_id {
                let pool_exists = match connection.query_row(
                    "SELECT EXISTS(SELECT 1 FROM proxyPools WHERE id = ?1)",
                    params![pool_id],
                    |row| row.get::<_, bool>(0),
                ) {
                    Ok(exists) => exists,
                    Err(error) => {
                        eprintln!("[api/providers] proxy-pool validation failed: {error}");
                        return admin_error(
                            StatusCode::SERVICE_UNAVAILABLE,
                            "Backend database unavailable",
                        );
                    }
                };
                if !pool_exists {
                    return admin_error(StatusCode::BAD_REQUEST, "Proxy pool not found");
                }
                provider_specific_data
                    .insert("proxyPoolId".to_owned(), Value::String(pool_id.to_owned()));
            }
            data.insert(
                "providerSpecificData".to_owned(),
                Value::Object(provider_specific_data),
            );
            let now = utc_timestamp();
            let transaction = match connection.transaction() {
                Ok(transaction) => transaction,
                Err(error) => {
                    eprintln!("[api/providers] create transaction failed: {error}");
                    return admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Provider connection could not be created",
                    );
                }
            };
            let existing_id = match transaction
                .query_row(
                    "SELECT id FROM providerConnections WHERE provider = ?1 AND authType = 'apikey' AND name = ?2 ORDER BY rowid LIMIT 1",
                    params![provider_id, name],
                    |row| row.get::<_, String>(0),
                )
                .optional()
            {
                Ok(id) => id,
                Err(error) => {
                    eprintln!("[api/providers] existing connection lookup failed: {error}");
                    return admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Provider connection could not be created",
                    );
                }
            };
            let connection_id = if let Some(id) = existing_id {
                match transaction.execute(
                    "UPDATE providerConnections SET email = NULL, priority = ?1, isActive = 1, data = ?2, updatedAt = ?3 WHERE id = ?4",
                    params![priority, Value::Object(data).to_string(), now, id],
                ) {
                    Ok(_) => id,
                    Err(error) => {
                        eprintln!("[api/providers] existing connection update failed: {error}");
                        return admin_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "Provider connection could not be created",
                        );
                    }
                }
            } else {
                if let Err(error) = transaction.execute(
                    "INSERT INTO providerConnections (id, provider, authType, name, email, priority, isActive, data, createdAt, updatedAt) VALUES (lower(hex(randomblob(16))), ?1, 'apikey', ?2, NULL, ?3, 1, ?4, ?5, ?5)",
                    params![provider_id, name, priority, Value::Object(data).to_string(), now],
                ) {
                    eprintln!("[api/providers] connection create failed: {error}");
                    let status = if is_sqlite_constraint(&error) {
                        StatusCode::CONFLICT
                    } else {
                        StatusCode::INTERNAL_SERVER_ERROR
                    };
                    return admin_error(status, "Provider connection could not be created");
                }
                match transaction.query_row(
                    "SELECT id FROM providerConnections WHERE provider = ?1 AND authType = 'apikey' AND name = ?2 ORDER BY rowid DESC LIMIT 1",
                    params![provider_id, name],
                    |row| row.get::<_, String>(0),
                ) {
                    Ok(id) => id,
                    Err(error) => {
                        eprintln!("[api/providers] created connection lookup failed: {error}");
                        return admin_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "Provider connection was created but could not be loaded",
                        );
                    }
                }
            };
            if let Err(error) = transaction.commit() {
                eprintln!("[api/providers] create transaction commit failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Provider connection could not be created",
                );
            }
            if let Err(error) = reorder_provider_priorities(&mut connection, provider_id) {
                eprintln!("[api/providers] connection priority reorder failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to reorder connections",
                );
            }
            match provider_connection_value(&connection, &connection_id) {
                Ok(Some(value)) => {
                    (StatusCode::CREATED, Json(json!({"connection": value}))).into_response()
                }
                Ok(None) => admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Created connection not found",
                ),
                Err(error) => {
                    eprintln!("[api/providers] created connection read failed: {error}");
                    admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Failed to load created connection",
                    )
                }
            }
        }
        _ => admin_error(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed"),
    }
}

async fn provider_resource(request: Request, id: String) -> Response {
    if let Err((status, message)) = require_dashboard_auth(&request) {
        return admin_error(status, message);
    }
    let (parts, body) = request.into_parts();
    let mut connection = match open_database_for_admin() {
        Ok(connection) => connection,
        Err(error) => {
            eprintln!("[api/providers/{id}] database unavailable: {error}");
            return admin_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Backend database unavailable",
            );
        }
    };
    match parts.method {
        Method::GET => match provider_connection_value(&connection, &id) {
            Ok(Some(value)) => Json(json!({"connection": value})).into_response(),
            Ok(None) => admin_error(StatusCode::NOT_FOUND, "Connection not found"),
            Err(error) => {
                eprintln!("[api/providers/{id}] read failed: {error}");
                admin_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Backend database unavailable",
                )
            }
        },
        Method::PUT => {
            let payload = match parse_admin_json(body).await {
                Ok(payload) => payload,
                Err((status, message)) => return admin_error(status, message),
            };
            let row = connection
                .query_row(
                    "SELECT provider, authType, name, priority, isActive, data FROM providerConnections WHERE id = ?1",
                    params![id],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, Option<String>>(2)?,
                            row.get::<_, Option<i64>>(3)?,
                            row.get::<_, Option<i64>>(4)?,
                            row.get::<_, String>(5)?,
                        ))
                    },
                )
                .optional();
            let (provider, auth_type, old_name, old_priority, active, raw_data) = match row {
                Ok(Some(row)) => row,
                Ok(None) => return admin_error(StatusCode::NOT_FOUND, "Connection not found"),
                Err(error) => {
                    eprintln!("[api/providers/{id}] connection lookup failed: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            let mut data = serde_json::from_str::<Value>(&raw_data)
                .ok()
                .and_then(|value| value.as_object().cloned())
                .unwrap_or_default();
            for field in [
                "globalPriority",
                "defaultModel",
                "testStatus",
                "lastError",
                "lastErrorAt",
                "projectId",
                "isProjectIdManual",
            ] {
                if let Some(value) = payload.get(field) {
                    data.insert(field.to_owned(), value.clone());
                }
            }
            if payload.get("testStatus").and_then(Value::as_str) == Some("active") {
                data.insert("lastError".to_owned(), Value::Null);
                data.insert("lastErrorAt".to_owned(), Value::Null);
                data.insert("errorCode".to_owned(), Value::Null);
                data.insert("rateLimitedUntil".to_owned(), Value::Null);
                data.insert("backoffLevel".to_owned(), Value::from(0));
                let model_lock_keys: Vec<_> = data
                    .keys()
                    .filter(|key| key.starts_with("modelLock_"))
                    .cloned()
                    .collect();
                for key in model_lock_keys {
                    data.insert(key, Value::Null);
                }
            }
            if auth_type == "apikey"
                && let Some(api_key) = payload.get("apiKey").and_then(Value::as_str)
                && !api_key.is_empty()
            {
                data.insert("apiKey".to_owned(), Value::String(api_key.to_owned()));
            }
            if let Some(incoming) = payload.get("providerSpecificData") {
                let incoming = match incoming {
                    Value::Object(incoming) => incoming.clone(),
                    Value::Null => serde_json::Map::new(),
                    _ => {
                        return admin_error(
                            StatusCode::BAD_REQUEST,
                            "providerSpecificData must be an object",
                        );
                    }
                };
                let existing = data
                    .get("providerSpecificData")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                let mut merged = existing;
                merged.extend(incoming);
                data.insert("providerSpecificData".to_owned(), Value::Object(merged));
            }
            let has_proxy_config = [
                "connectionProxyEnabled",
                "connectionProxyUrl",
                "connectionNoProxy",
            ]
            .iter()
            .any(|field| payload.get(*field).is_some());
            if has_proxy_config {
                let enabled = payload.get("connectionProxyEnabled") == Some(&Value::Bool(true));
                let url = payload
                    .get("connectionProxyUrl")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim();
                let no_proxy = payload
                    .get("connectionNoProxy")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim();
                if enabled && url.is_empty() {
                    return admin_error(
                        StatusCode::BAD_REQUEST,
                        "Connection proxy URL is required when connection proxy is enabled",
                    );
                }
                let provider_specific_data = data
                    .entry("providerSpecificData".to_owned())
                    .or_insert_with(|| Value::Object(serde_json::Map::new()));
                let Some(provider_specific_data) = provider_specific_data.as_object_mut() else {
                    return admin_error(
                        StatusCode::BAD_REQUEST,
                        "providerSpecificData must be an object",
                    );
                };
                provider_specific_data
                    .insert("connectionProxyEnabled".to_owned(), Value::Bool(enabled));
                provider_specific_data.insert(
                    "connectionProxyUrl".to_owned(),
                    Value::String(url.to_owned()),
                );
                provider_specific_data.insert(
                    "connectionNoProxy".to_owned(),
                    Value::String(no_proxy.to_owned()),
                );
            }
            if payload.get("proxyPoolIds").is_some() || payload.get("proxyPoolId").is_some() {
                let (pool_id, pool_ids, rotation_strategy) = if let Some(pool_ids) =
                    payload.get("proxyPoolIds")
                {
                    let Some(pool_ids) = pool_ids.as_array() else {
                        return admin_error(
                            StatusCode::BAD_REQUEST,
                            "proxyPoolIds must be an array",
                        );
                    };
                    let mut unique_pool_ids = HashSet::new();
                    let mut normalized_pool_ids = Vec::new();
                    for value in pool_ids {
                        let Some(pool_id) = value.as_str() else {
                            return admin_error(
                                StatusCode::BAD_REQUEST,
                                "proxyPoolIds must contain only strings",
                            );
                        };
                        let pool_id = pool_id.trim();
                        if !pool_id.is_empty() && unique_pool_ids.insert(pool_id.to_owned()) {
                            normalized_pool_ids.push(pool_id.to_owned());
                        }
                    }
                    let strategy = payload
                        .get("proxyRotationStrategy")
                        .and_then(Value::as_str)
                        .unwrap_or("none")
                        .trim()
                        .to_ascii_lowercase();
                    if !["none", "fill-first", "round-robin", "random", "smart"]
                        .contains(&strategy.as_str())
                    {
                        return admin_error(
                            StatusCode::BAD_REQUEST,
                            "Invalid proxy rotation strategy",
                        );
                    }
                    (None, normalized_pool_ids, Some(strategy))
                } else {
                    let pool_id = match payload.get("proxyPoolId") {
                        Some(Value::String(id))
                            if !id.trim().is_empty() && id.trim() != "__none__" =>
                        {
                            Some(id.trim().to_owned())
                        }
                        Some(Value::Null | Value::String(_)) => None,
                        Some(_) => {
                            return admin_error(StatusCode::BAD_REQUEST, "Invalid proxy pool id");
                        }
                        None => None,
                    };
                    (pool_id, Vec::new(), None)
                };
                for pool_id in pool_id.iter().chain(pool_ids.iter()) {
                    let exists = match connection.query_row(
                        "SELECT EXISTS(SELECT 1 FROM proxyPools WHERE id = ?1)",
                        params![pool_id],
                        |row| row.get::<_, bool>(0),
                    ) {
                        Ok(exists) => exists,
                        Err(error) => {
                            eprintln!("[api/providers/{id}] proxy-pool validation failed: {error}");
                            return admin_error(
                                StatusCode::SERVICE_UNAVAILABLE,
                                "Backend database unavailable",
                            );
                        }
                    };
                    if !exists {
                        return admin_error(StatusCode::BAD_REQUEST, "Proxy pool not found");
                    }
                }
                let provider_specific_data = data
                    .entry("providerSpecificData".to_owned())
                    .or_insert_with(|| Value::Object(serde_json::Map::new()));
                let Some(provider_specific_data) = provider_specific_data.as_object_mut() else {
                    return admin_error(
                        StatusCode::BAD_REQUEST,
                        "providerSpecificData must be an object",
                    );
                };
                provider_specific_data.remove("proxyPoolId");
                provider_specific_data.remove("proxyPoolIds");
                provider_specific_data.remove("proxyRotationStrategy");
                if let Some(pool_id) = pool_id {
                    provider_specific_data.insert("proxyPoolId".to_owned(), Value::String(pool_id));
                }
                if !pool_ids.is_empty() {
                    provider_specific_data.insert(
                        "proxyPoolIds".to_owned(),
                        Value::Array(pool_ids.into_iter().map(Value::String).collect()),
                    );
                    if let Some(strategy) = rotation_strategy {
                        provider_specific_data
                            .insert("proxyRotationStrategy".to_owned(), Value::String(strategy));
                    }
                }
            }
            let name = payload
                .get("name")
                .and_then(Value::as_str)
                .map(str::trim)
                .map(str::to_owned)
                .or(old_name);
            let priority = match payload.get("priority") {
                Some(value) => match value.as_i64().filter(|priority| *priority > 0) {
                    Some(priority) => Some(priority),
                    None => {
                        return admin_error(
                            StatusCode::BAD_REQUEST,
                            "priority must be a positive integer",
                        );
                    }
                },
                None => old_priority,
            };
            let active = match payload.get("isActive") {
                Some(Value::Bool(active)) => {
                    if *active {
                        1
                    } else {
                        0
                    }
                }
                Some(_) => {
                    return admin_error(StatusCode::BAD_REQUEST, "isActive must be a boolean");
                }
                None => active.unwrap_or(0),
            };
            let result = connection.execute(
                "UPDATE providerConnections SET name = ?1, priority = ?2, isActive = ?3, data = ?4, updatedAt = ?5 WHERE id = ?6",
                params![name, priority, active, Value::Object(data).to_string(), utc_timestamp(), id],
            );
            if let Err(error) = result {
                eprintln!("[api/providers/{id}] update failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to update connection",
                );
            }
            if payload.get("priority").is_some()
                && let Err(error) = reorder_provider_priorities(&mut connection, &provider)
            {
                eprintln!("[api/providers/{id}] priority reorder failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to reorder connections",
                );
            }
            match provider_connection_value(&connection, &id) {
                Ok(Some(value)) => Json(json!({"connection": value})).into_response(),
                Ok(None) => admin_error(StatusCode::NOT_FOUND, "Connection not found"),
                Err(error) => {
                    eprintln!("[api/providers/{id}] updated connection read failed: {error}");
                    admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Failed to load updated connection",
                    )
                }
            }
        }
        Method::DELETE => {
            let provider = connection.query_row(
                "SELECT provider FROM providerConnections WHERE id = ?1",
                params![id],
                |row| row.get::<_, String>(0),
            );
            let provider = match provider {
                Ok(provider) => provider,
                Err(rusqlite::Error::QueryReturnedNoRows) => {
                    return admin_error(StatusCode::NOT_FOUND, "Connection not found");
                }
                Err(error) => {
                    eprintln!("[api/providers/{id}] connection lookup failed: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            if let Err(error) =
                connection.execute("DELETE FROM providerConnections WHERE id = ?1", params![id])
            {
                eprintln!("[api/providers/{id}] delete failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to delete connection",
                );
            }
            if let Err(error) = reorder_provider_priorities(&mut connection, &provider) {
                eprintln!("[api/providers/{id}] priority reorder failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to reorder connections",
                );
            }
            Json(json!({"message": "Connection deleted successfully"})).into_response()
        }
        _ => admin_error(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed"),
    }
}

fn combo_value_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Value> {
    let models: String = row.get("models")?;
    let models = serde_json::from_str::<Value>(&models).unwrap_or_else(|_| json!([]));
    Ok(json!({
        "id": row.get::<_, String>("id")?,
        "name": row.get::<_, String>("name")?,
        "kind": row.get::<_, Option<String>>("kind")?,
        "models": models,
        "context_length": row.get::<_, Option<i64>>("context_length")?,
        "createdAt": row.get::<_, String>("createdAt")?,
        "updatedAt": row.get::<_, String>("updatedAt")?,
    }))
}

fn combo_name_valid(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

fn combo_context_length(value: Option<&Value>) -> Result<Option<i64>, &'static str> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if value.is_empty() => Ok(None),
        Some(value) => {
            let length = value
                .as_i64()
                .or_else(|| value.as_str().and_then(|raw| raw.parse::<i64>().ok()));
            length
                .filter(|length| (1..=2_000_000).contains(length))
                .map(Some)
                .ok_or("context_length must be a positive integer not exceeding 2000000")
        }
    }
}

fn utc_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let milliseconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let seconds = (milliseconds / 1_000) as i64;
    let days = seconds.div_euclid(86_400);
    let seconds_of_day = seconds.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        seconds_of_day / 3_600,
        seconds_of_day % 3_600 / 60,
        seconds_of_day % 60,
        milliseconds % 1_000
    )
}

fn load_combo(connection: &Connection, id: &str) -> Result<Option<Value>, String> {
    connection
        .query_row(
            "SELECT id, name, kind, models, context_length, createdAt, updatedAt FROM combos WHERE id = ?1",
            params![id],
            combo_value_from_row,
        )
        .optional()
        .map_err(|error| error.to_string())
}

async fn combos_collection(request: Request) -> Response {
    if let Err((status, message)) = require_dashboard_auth(&request) {
        return admin_error(status, message);
    }
    let (parts, body) = request.into_parts();
    match parts.method {
        Method::GET => {
            let connection = match open_database() {
                Ok(connection) => connection,
                Err(error) => {
                    eprintln!("[api/combos] database unavailable: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            let mut statement = match connection.prepare(
                "SELECT id, name, kind, models, context_length, createdAt, updatedAt FROM combos ORDER BY createdAt ASC",
            ) {
                Ok(statement) => statement,
                Err(error) => {
                    eprintln!("[api/combos] query prepare failed: {error}");
                    return admin_error(StatusCode::SERVICE_UNAVAILABLE, "Backend database unavailable");
                }
            };
            let rows = match statement.query_map([], combo_value_from_row) {
                Ok(rows) => rows,
                Err(error) => {
                    eprintln!("[api/combos] query failed: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            let mut combos = Vec::new();
            for row in rows {
                match row {
                    Ok(combo) => combos.push(combo),
                    Err(error) => {
                        eprintln!("[api/combos] row read failed: {error}");
                        return admin_error(
                            StatusCode::SERVICE_UNAVAILABLE,
                            "Backend database unavailable",
                        );
                    }
                }
            }
            Json(json!({"combos": combos})).into_response()
        }
        Method::POST => {
            let payload = match parse_admin_json(body).await {
                Ok(payload) => payload,
                Err((status, message)) => return admin_error(status, message),
            };
            let Some(name) = payload.get("name").and_then(Value::as_str) else {
                return admin_error(StatusCode::BAD_REQUEST, "Name is required");
            };
            if name.is_empty() {
                return admin_error(StatusCode::BAD_REQUEST, "Name is required");
            }
            if !combo_name_valid(name) {
                return admin_error(
                    StatusCode::BAD_REQUEST,
                    "Name can only contain letters, numbers, -, _ and .",
                );
            }
            let models = payload.get("models").cloned().unwrap_or_else(|| json!([]));
            if !models.is_array() {
                return admin_error(StatusCode::BAD_REQUEST, "models must be an array");
            }
            if payload
                .get("kind")
                .is_some_and(|kind| !kind.is_null() && !kind.is_string())
            {
                return admin_error(StatusCode::BAD_REQUEST, "kind must be a string or null");
            }
            let context_length = match combo_context_length(payload.get("context_length")) {
                Ok(value) => value,
                Err(error) => return admin_error(StatusCode::BAD_REQUEST, error),
            };
            let connection = match open_database_for_admin() {
                Ok(connection) => connection,
                Err(error) => {
                    eprintln!("[api/combos] database unavailable: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            let now = utc_timestamp();
            let result = connection.execute(
                "INSERT INTO combos (id, name, kind, models, context_length, createdAt, updatedAt) VALUES (lower(hex(randomblob(16))), ?1, ?2, ?3, ?4, ?5, ?5)",
                params![
                    name,
                    payload
                        .get("kind")
                        .and_then(Value::as_str)
                        .filter(|kind| !kind.is_empty()),
                    models.to_string(),
                    context_length,
                    now
                ],
            );
            if let Err(error) = result {
                eprintln!("[api/combos] create failed: {error}");
                let status = if is_sqlite_constraint(&error) {
                    StatusCode::BAD_REQUEST
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                };
                let message = if status == StatusCode::BAD_REQUEST {
                    "Combo name already exists"
                } else {
                    "Failed to create combo"
                };
                return admin_error(status, message);
            }
            let id = match connection.query_row(
                "SELECT id FROM combos WHERE name = ?1",
                params![name],
                |row| row.get::<_, String>(0),
            ) {
                Ok(id) => id,
                Err(error) => {
                    eprintln!("[api/combos] created combo lookup failed: {error}");
                    return admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Failed to load created combo",
                    );
                }
            };
            match load_combo(&connection, &id) {
                Ok(Some(combo)) => (StatusCode::CREATED, Json(combo)).into_response(),
                Ok(None) => {
                    admin_error(StatusCode::INTERNAL_SERVER_ERROR, "Created combo not found")
                }
                Err(error) => {
                    eprintln!("[api/combos] created combo read failed: {error}");
                    admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Failed to load created combo",
                    )
                }
            }
        }
        _ => admin_error(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed"),
    }
}

async fn combo_resource(request: Request, id: String) -> Response {
    if let Err((status, message)) = require_dashboard_auth(&request) {
        return admin_error(status, message);
    }
    let (parts, body) = request.into_parts();
    let connection = match open_database_for_admin() {
        Ok(connection) => connection,
        Err(error) => {
            eprintln!("[api/combos/{id}] database unavailable: {error}");
            return admin_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Backend database unavailable",
            );
        }
    };
    match parts.method {
        Method::GET => match load_combo(&connection, &id) {
            Ok(Some(combo)) => Json(combo).into_response(),
            Ok(None) => admin_error(StatusCode::NOT_FOUND, "Combo not found"),
            Err(error) => {
                eprintln!("[api/combos/{id}] read failed: {error}");
                admin_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Backend database unavailable",
                )
            }
        },
        Method::PUT => {
            let payload = match parse_admin_json(body).await {
                Ok(payload) => payload,
                Err((status, message)) => return admin_error(status, message),
            };
            let Some(existing) = (match load_combo(&connection, &id) {
                Ok(combo) => combo,
                Err(error) => {
                    eprintln!("[api/combos/{id}] read failed: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            }) else {
                return admin_error(StatusCode::NOT_FOUND, "Combo not found");
            };
            let name = payload
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_else(|| existing["name"].as_str().unwrap_or(""));
            if !combo_name_valid(name) {
                return admin_error(
                    StatusCode::BAD_REQUEST,
                    "Name can only contain letters, numbers, -, _ and .",
                );
            }
            let name_exists = match connection.query_row(
                "SELECT EXISTS(SELECT 1 FROM combos WHERE name = ?1 AND id <> ?2)",
                params![name, id],
                |row| row.get::<_, bool>(0),
            ) {
                Ok(exists) => exists,
                Err(error) => {
                    eprintln!("[api/combos/{id}] duplicate-name check failed: {error}");
                    return admin_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            if name_exists {
                return admin_error(StatusCode::BAD_REQUEST, "Combo name already exists");
            }
            if payload
                .get("models")
                .is_some_and(|models| !models.is_array())
            {
                return admin_error(StatusCode::BAD_REQUEST, "models must be an array");
            }
            let context_length = if payload.get("context_length").is_some() {
                match combo_context_length(payload.get("context_length")) {
                    Ok(value) => value,
                    Err(error) => return admin_error(StatusCode::BAD_REQUEST, error),
                }
            } else {
                existing["context_length"].as_i64()
            };
            let models = payload.get("models").unwrap_or(&existing["models"]);
            let kind = match payload.get("kind") {
                Some(Value::Null) => None,
                Some(value) => match value.as_str() {
                    Some(kind) => Some(kind),
                    None => {
                        return admin_error(
                            StatusCode::BAD_REQUEST,
                            "kind must be a string or null",
                        );
                    }
                },
                None => existing["kind"].as_str(),
            };
            let result = connection.execute(
                "UPDATE combos SET name = ?1, kind = ?2, models = ?3, context_length = ?4, updatedAt = ?5 WHERE id = ?6",
                params![name, kind, models.to_string(), context_length, utc_timestamp(), id],
            );
            if let Err(error) = result {
                eprintln!("[api/combos/{id}] update failed: {error}");
                return admin_error(StatusCode::INTERNAL_SERVER_ERROR, "Failed to update combo");
            }
            match load_combo(&connection, &id) {
                Ok(Some(combo)) => Json(combo).into_response(),
                Ok(None) => admin_error(StatusCode::NOT_FOUND, "Combo not found"),
                Err(error) => {
                    eprintln!("[api/combos/{id}] updated combo read failed: {error}");
                    admin_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Failed to load updated combo",
                    )
                }
            }
        }
        Method::DELETE => match connection.execute("DELETE FROM combos WHERE id = ?1", params![id])
        {
            Ok(0) => admin_error(StatusCode::NOT_FOUND, "Combo not found"),
            Ok(_) => Json(json!({"success": true})).into_response(),
            Err(error) => {
                eprintln!("[api/combos/{id}] delete failed: {error}");
                admin_error(StatusCode::INTERNAL_SERVER_ERROR, "Failed to delete combo")
            }
        },
        _ => admin_error(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed"),
    }
}

fn resolve_model_alias(connection: &Connection, requested_model: &str) -> Result<String, String> {
    let aliases = read_kv_scope(connection, "modelAliases")?;
    let mut current = requested_model.to_owned();
    let mut seen = HashSet::new();
    for _ in 0..16 {
        if !seen.insert(current.clone()) {
            return Err("Model alias cycle detected".to_owned());
        }
        let Some(target) = aliases.get(&current).and_then(Value::as_str) else {
            return Ok(current);
        };
        current = target.to_owned();
    }
    Err("Model alias chain is too long".to_owned())
}

fn combo_models_for_chat(
    connection: &Connection,
    model: &str,
) -> Result<Option<(String, Vec<String>, String)>, String> {
    let name = if let Some(name) = model.strip_prefix("combo/") {
        name
    } else if !model.contains('/') {
        model
    } else {
        return Ok(None);
    };
    let Some((canonical_name, raw_models)) = connection
        .query_row(
            "SELECT name, models FROM combos WHERE name = ?1",
            params![name],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(|error| error.to_string())?
    else {
        return Ok(None);
    };
    let models = serde_json::from_str::<Vec<String>>(&raw_models)
        .map_err(|error| format!("Combo model list is invalid: {error}"))?
        .into_iter()
        .filter(|model| !model.trim().is_empty())
        .collect::<Vec<_>>();
    let settings = connection
        .query_row("SELECT data FROM settings WHERE id = 1", [], |row| {
            row.get::<_, String>(0)
        })
        .optional()
        .map_err(|error| error.to_string())?
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .unwrap_or(Value::Null);
    let strategy = settings
        .get("comboStrategies")
        .and_then(|strategies| strategies.get(&canonical_name))
        .and_then(|settings| settings.get("fallbackStrategy"))
        .and_then(Value::as_str)
        .or_else(|| settings.get("comboStrategy").and_then(Value::as_str))
        .unwrap_or("fallback")
        .to_owned();
    Ok(Some((canonical_name, models, strategy)))
}

fn combo_rotation_index(name: &str, length: usize) -> usize {
    static ROTATION: OnceLock<StdMutex<HashMap<String, usize>>> = OnceLock::new();
    let Ok(mut rotation) = ROTATION
        .get_or_init(|| StdMutex::new(HashMap::new()))
        .lock()
    else {
        return 0;
    };
    let index = rotation.get(name).copied().unwrap_or_default() % length;
    rotation.insert(name.to_owned(), (index + 1) % length);
    index
}

fn chat_provider_id(
    connection: &Connection,
    output_alias: &str,
) -> Result<Option<(String, bool)>, String> {
    if let Some(provider) = model_catalog()
        .providers
        .iter()
        .find(|provider| provider.alias == output_alias || provider.id == output_alias)
    {
        return Ok(Some((provider.id.clone(), provider.id == "openai")));
    }

    let mut statement = connection
        .prepare("SELECT id, type, data FROM providerNodes")
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                row.get::<_, String>(2)?,
            ))
        })
        .map_err(|error| error.to_string())?;
    for row in rows {
        let (id, kind, raw_data) = row.map_err(|error| error.to_string())?;
        let data = serde_json::from_str::<Value>(&raw_data).unwrap_or(Value::Null);
        let prefix = data.get("prefix").and_then(Value::as_str).unwrap_or(&id);
        if prefix == output_alias || id == output_alias {
            let is_openai_chat = kind == "openai-compatible"
                && data.get("apiType").and_then(Value::as_str) == Some("chat");
            return Ok(Some((id, is_openai_chat)));
        }
    }
    Ok(None)
}

fn load_openai_connection(
    connection: &Connection,
    provider_id: &str,
) -> Result<(String, String, String), String> {
    let (connection_id, raw_data) = connection
        .query_row(
            "SELECT id, data FROM providerConnections WHERE provider = ?1 AND isActive = 1 ORDER BY priority ASC LIMIT 1",
            params![provider_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "No active OpenAI-compatible provider connection".to_owned())?;
    let data: Value = serde_json::from_str(&raw_data).map_err(|error| error.to_string())?;
    let api_key = data
        .get("accessToken")
        .or_else(|| data.get("apiKey"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Provider connection has no API credential".to_owned())?
        .to_owned();
    let configured_base = data
        .get("providerSpecificData")
        .and_then(|value| value.get("baseUrl"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(if provider_id == "openai" {
            "https://api.openai.com/v1"
        } else {
            ""
        });
    if configured_base.is_empty() {
        return Err("Custom OpenAI-compatible provider has no base URL".to_owned());
    }
    let url = if configured_base.ends_with("/chat/completions") {
        configured_base.to_owned()
    } else {
        format!("{}/chat/completions", configured_base.trim_end_matches('/'))
    };
    let parsed: reqwest::Url = url
        .parse()
        .map_err(|error| format!("Invalid OpenAI base URL: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err("OpenAI base URL must use HTTP or HTTPS".to_owned());
    }
    Ok((connection_id, api_key, url))
}

fn add_usage_counter(
    counters: &mut Value,
    key: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
    cached_tokens: u64,
    cost: f64,
    meta: Value,
) {
    if !counters.is_object() {
        *counters = json!({});
    }
    let entry = counters
        .as_object_mut()
        .expect("usage counters are initialized as objects")
        .entry(key.to_owned())
        .or_insert_with(|| {
            json!({
                "requests": 0,
                "promptTokens": 0,
                "completionTokens": 0,
                "cachedTokens": 0,
                "cost": 0.0
            })
        });
    for (field, amount) in [
        ("requests", 1),
        ("promptTokens", prompt_tokens),
        ("completionTokens", completion_tokens),
        ("cachedTokens", cached_tokens),
    ] {
        let current = entry.get(field).and_then(Value::as_u64).unwrap_or_default();
        entry[field] = Value::from(current.saturating_add(amount));
    }
    let current_cost = entry
        .get("cost")
        .and_then(Value::as_f64)
        .unwrap_or_default();
    entry["cost"] = json!(current_cost + cost);
    if let Some(fields) = meta.as_object() {
        for (key, value) in fields {
            entry[key] = value.clone();
        }
    }
}

fn extract_sse_usage(pending: &mut Vec<u8>, flush: bool) -> Option<Value> {
    let mut latest = None;
    loop {
        let delimiter = pending
            .windows(2)
            .position(|window| window == b"\n\n")
            .map(|index| (index, 2))
            .or_else(|| {
                pending
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|index| (index, 4))
            });
        let Some((index, delimiter_len)) = delimiter else {
            break;
        };
        let event = pending.drain(..index + delimiter_len).collect::<Vec<_>>();
        if let Some(usage) = usage_from_sse_event(&event) {
            latest = Some(usage);
        }
    }
    if flush && !pending.is_empty() {
        if let Some(usage) = usage_from_sse_event(pending) {
            latest = Some(usage);
        }
        pending.clear();
    }
    if pending.len() > 1024 * 1024 {
        pending.clear();
    }
    latest
}

fn mask_usage_api_key(key: &str) -> String {
    let chars = key.chars().collect::<Vec<_>>();
    if chars.len() <= 12 {
        return format!("{}***", chars.first().copied().unwrap_or('*'));
    }
    format!(
        "{}***{}",
        chars.iter().take(8).collect::<String>(),
        chars.iter().skip(chars.len() - 4).collect::<String>()
    )
}

fn usage_from_sse_event(event: &[u8]) -> Option<Value> {
    let text = String::from_utf8_lossy(event);
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            continue;
        }
        if let Ok(payload) = serde_json::from_str::<Value>(data)
            && let Some(usage) = payload.get("usage")
            && (usage.get("prompt_tokens").is_some()
                || usage.get("input_tokens").is_some()
                || usage.get("completion_tokens").is_some()
                || usage.get("output_tokens").is_some())
        {
            return Some(usage.clone());
        }
    }
    None
}

fn record_chat_usage(
    provider: &str,
    model: &str,
    connection_id: &str,
    api_key: Option<&str>,
    endpoint: &str,
    usage: &Value,
) -> Result<(), String> {
    let prompt_tokens = usage
        .get("prompt_tokens")
        .or_else(|| usage.get("input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let completion_tokens = usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let cached_tokens = usage
        .get("cached_tokens")
        .or_else(|| usage.get("cache_read_input_tokens"))
        .or_else(|| {
            usage
                .get("prompt_tokens_details")
                .and_then(|details| details.get("cached_tokens"))
        })
        .and_then(Value::as_u64)
        .unwrap_or_default();
    if prompt_tokens == 0 && completion_tokens == 0 {
        return Ok(());
    }
    let mut connection = open_database_for_writes()?;
    let cost = model_pricing(&connection, provider, model)?
        .as_ref()
        .map(|pricing| calculate_chat_cost(usage, pricing))
        .unwrap_or_default();
    persist_chat_usage(
        &mut connection,
        provider,
        model,
        connection_id,
        api_key,
        endpoint,
        usage,
        prompt_tokens,
        completion_tokens,
        cached_tokens,
        cost,
    )
}

#[allow(clippy::too_many_arguments)]
fn persist_chat_usage(
    connection: &mut Connection,
    provider: &str,
    model: &str,
    connection_id: &str,
    api_key: Option<&str>,
    endpoint: &str,
    usage: &Value,
    prompt_tokens: u64,
    completion_tokens: u64,
    cached_tokens: u64,
    cost: f64,
) -> Result<(), String> {
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    let timestamp = utc_timestamp();
    let api_key_value = api_key.filter(|value| !value.is_empty());
    transaction
        .execute(
            "INSERT INTO usageHistory(timestamp, provider, model, connectionId, apiKey, endpoint, promptTokens, completionTokens, cost, status, tokens, meta) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'ok', ?10, '{}')",
            params![
                timestamp,
                provider,
                model,
                connection_id,
                api_key_value,
                endpoint,
                prompt_tokens,
                completion_tokens,
                cost,
                usage.to_string()
            ],
        )
        .map_err(|error| error.to_string())?;

    let date_key: String = transaction
        .query_row("SELECT date(?1, 'localtime')", params![timestamp], |row| {
            row.get(0)
        })
        .map_err(|error| error.to_string())?;
    let raw_day: Option<String> = transaction
        .query_row(
            "SELECT data FROM usageDaily WHERE dateKey = ?1",
            params![date_key],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| error.to_string())?;
    let mut day = raw_day
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .unwrap_or_else(|| {
            json!({
                "requests": 0,
                "promptTokens": 0,
                "completionTokens": 0,
                "cachedTokens": 0,
                "cost": 0.0,
                "byProvider": {},
                "byModel": {},
                "byAccount": {},
                "byApiKey": {},
                "byEndpoint": {}
            })
        });
    for (field, amount) in [
        ("requests", 1),
        ("promptTokens", prompt_tokens),
        ("completionTokens", completion_tokens),
        ("cachedTokens", cached_tokens),
    ] {
        let current = day.get(field).and_then(Value::as_u64).unwrap_or_default();
        day[field] = Value::from(current.saturating_add(amount));
    }
    let total_cost = day.get("cost").and_then(Value::as_f64).unwrap_or_default();
    day["cost"] = json!(total_cost + cost);
    let model_key = format!("{model}|{provider}");
    let api_key_string = api_key_value.unwrap_or("local-no-key");
    let api_key_model_key = format!("{api_key_string}|{model}|{provider}");
    let endpoint_model_key = format!("{endpoint}|{model}|{provider}");
    let meta = json!({"rawModel": model, "provider": provider});
    add_usage_counter(
        &mut day["byProvider"],
        provider,
        prompt_tokens,
        completion_tokens,
        cached_tokens,
        cost,
        json!({}),
    );
    add_usage_counter(
        &mut day["byModel"],
        &model_key,
        prompt_tokens,
        completion_tokens,
        cached_tokens,
        cost,
        meta.clone(),
    );
    add_usage_counter(
        &mut day["byAccount"],
        connection_id,
        prompt_tokens,
        completion_tokens,
        cached_tokens,
        cost,
        meta,
    );
    add_usage_counter(
        &mut day["byApiKey"],
        &api_key_model_key,
        prompt_tokens,
        completion_tokens,
        cached_tokens,
        cost,
        json!({"rawModel": model, "provider": provider, "apiKey": api_key_value}),
    );
    add_usage_counter(
        &mut day["byEndpoint"],
        &endpoint_model_key,
        prompt_tokens,
        completion_tokens,
        cached_tokens,
        cost,
        json!({"endpoint": endpoint, "rawModel": model, "provider": provider}),
    );
    transaction
        .execute(
            "INSERT INTO usageDaily(dateKey, data) VALUES(?1, ?2) ON CONFLICT(dateKey) DO UPDATE SET data = excluded.data",
            params![date_key, day.to_string()],
        )
        .map_err(|error| error.to_string())?;
    transaction
        .execute(
            "INSERT INTO _meta(key, value) VALUES('totalRequestsLifetime', '1') ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1",
            [],
        )
        .map_err(|error| error.to_string())?;
    transaction.commit().map_err(|error| error.to_string())
}

fn chat_error(status: StatusCode, message: &str) -> Response {
    api_error(status, message)
}

fn chat_rate_limit_error(retry_after: u64) -> Response {
    let mut response = chat_error(
        StatusCode::TOO_MANY_REQUESTS,
        "API key requests-per-minute limit exceeded",
    );
    if let Ok(value) = axum::http::HeaderValue::from_str(&retry_after.to_string()) {
        response.headers_mut().insert("retry-after", value);
    }
    response
}

async fn finish_chat_upstream(
    upstream: reqwest::Response,
    provider: &str,
    model: &str,
    connection_id: &str,
    api_key: Option<&str>,
) -> Response {
    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let is_event_stream = upstream
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.to_ascii_lowercase().contains("text/event-stream"));
    let upstream_headers: Vec<_> = [
        "content-type",
        "cache-control",
        "retry-after",
        "x-request-id",
    ]
    .into_iter()
    .filter_map(|name| {
        upstream
            .headers()
            .get(name)
            .map(|value| (name, value.clone()))
    })
    .collect();
    let mut response = if is_event_stream {
        let stream = upstream.bytes_stream();
        let provider = provider.to_owned();
        let model = model.to_owned();
        let connection_id = connection_id.to_owned();
        let api_key = api_key.map(str::to_owned);
        let body_stream = futures_util::stream::unfold(
            (stream, Vec::new(), None::<Value>),
            move |(mut stream, mut pending, mut usage)| {
                let provider = provider.clone();
                let model = model.clone();
                let connection_id = connection_id.clone();
                let api_key = api_key.clone();
                async move {
                    match stream.next().await {
                        Some(Ok(chunk)) => {
                            pending.extend_from_slice(&chunk);
                            if let Some(found) = extract_sse_usage(&mut pending, false) {
                                usage = Some(found);
                            }
                            Some((Ok::<_, std::io::Error>(chunk), (stream, pending, usage)))
                        }
                        Some(Err(error)) => {
                            Some((Err(std::io::Error::other(error)), (stream, pending, usage)))
                        }
                        None => {
                            if let Some(found) = extract_sse_usage(&mut pending, true) {
                                usage = Some(found);
                            }
                            if let Some(usage) = usage {
                                let _ = tokio::task::spawn_blocking(move || {
                                    if let Err(error) = record_chat_usage(
                                        &provider,
                                        &model,
                                        &connection_id,
                                        api_key.as_deref(),
                                        "/v1/chat/completions",
                                        &usage,
                                    ) {
                                        eprintln!(
                                            "[v1/chat/completions] streaming usage write failed: {error}"
                                        );
                                    }
                                })
                                .await;
                            }
                            None
                        }
                    }
                }
            },
        );
        Response::new(Body::from_stream(body_stream))
    } else {
        match upstream.bytes().await {
            Ok(bytes) => {
                if status.is_success()
                    && let Ok(body) = serde_json::from_slice::<Value>(&bytes)
                    && let Some(usage) = body.get("usage")
                    && let Err(error) = record_chat_usage(
                        provider,
                        model,
                        connection_id,
                        api_key,
                        "/v1/chat/completions",
                        usage,
                    )
                {
                    eprintln!("[v1/chat/completions] usage write failed: {error}");
                }
                Response::new(Body::from(bytes))
            }
            Err(error) => {
                eprintln!("[v1/chat/completions] upstream response read failed: {error}");
                return chat_error(StatusCode::BAD_GATEWAY, "OpenAI upstream response failed");
            }
        }
    };
    *response.status_mut() = status;
    for (header_name, value) in upstream_headers {
        response.headers_mut().insert(header_name, value);
    }
    response.headers_mut().insert(
        "access-control-allow-origin",
        axum::http::HeaderValue::from_static("*"),
    );
    response
}

async fn chat_completions(
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    request: Request,
) -> Response {
    let api_key = extract_api_key(&request);
    let cli_token = extract_cli_token(&request).map(str::to_owned);
    let is_local = is_local_request(&request, peer);
    let (parts, body) = request.into_parts();
    let body = match to_bytes(body, MAX_CHAT_BODY_BYTES).await {
        Ok(body) => body,
        Err(error) => {
            let message = if error
                .to_string()
                .to_ascii_lowercase()
                .contains("length limit")
            {
                "Request body too large"
            } else {
                "Unable to read request body"
            };
            let status = if message == "Request body too large" {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_REQUEST
            };
            return chat_error(status, message);
        }
    };
    let mut payload: Value = match serde_json::from_slice::<Value>(&body) {
        Ok(payload) if payload.is_object() => payload,
        _ => return chat_error(StatusCode::BAD_REQUEST, "Invalid JSON body"),
    };
    let requested_model = match payload.get("model").and_then(Value::as_str) {
        Some(model) if !model.trim().is_empty() => model.to_owned(),
        _ => return chat_error(StatusCode::BAD_REQUEST, "Missing model"),
    };

    let connection = match open_database() {
        Ok(connection) => connection,
        Err(error) => {
            eprintln!("[v1/chat/completions] database unavailable: {error}");
            return chat_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Backend database unavailable",
            );
        }
    };
    let requested_model = match resolve_model_alias(&connection, &requested_model) {
        Ok(model) => model,
        Err(error) => {
            eprintln!("[v1/chat/completions] model alias resolution failed: {error}");
            return chat_error(StatusCode::BAD_REQUEST, "Invalid model alias");
        }
    };

    let models =
        match read_v1_models_from_connection(&connection, api_key.clone(), cli_token, is_local) {
            Ok(models) => models,
            Err((status, message)) => return chat_error(status, &message),
        };
    let (combo_name, mut candidate_models, combo_strategy) =
        match combo_models_for_chat(&connection, &requested_model) {
            Ok(Some((name, candidates, strategy))) => {
                if !models
                    .data
                    .iter()
                    .any(|entry| entry.id == format!("combo/{name}"))
                {
                    return chat_error(
                        StatusCode::FORBIDDEN,
                        "Combo is not allowed for this API key",
                    );
                }
                (Some(name), candidates, strategy)
            }
            Ok(None) => (None, vec![requested_model.clone()], "fallback".to_owned()),
            Err(error) => {
                eprintln!("[v1/chat/completions] combo lookup failed: {error}");
                return chat_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Backend database unavailable",
                );
            }
        };
    if candidate_models.is_empty() {
        return chat_error(StatusCode::NOT_FOUND, "Combo has no configured models");
    }
    if combo_strategy == "round-robin" && candidate_models.len() > 1 {
        let start =
            combo_rotation_index(combo_name.as_deref().unwrap_or(""), candidate_models.len());
        candidate_models.rotate_left(start);
    }
    for candidate in &mut candidate_models {
        match resolve_model_alias(&connection, candidate) {
            Ok(model) => *candidate = model,
            Err(error) => {
                eprintln!("[v1/chat/completions] combo model alias resolution failed: {error}");
                return chat_error(StatusCode::BAD_REQUEST, "Invalid model alias");
            }
        }
    }
    let is_combo = combo_name.is_some();
    if !is_combo
        && !models
            .data
            .iter()
            .any(|entry| entry.id == candidate_models[0])
    {
        return chat_error(StatusCode::NOT_FOUND, "Model is not available");
    }
    if is_combo
        && !models
            .data
            .iter()
            .any(|entry| entry.id == candidate_models[0])
    {
        candidate_models.retain(|candidate| models.data.iter().any(|entry| entry.id == *candidate));
    }
    if is_combo && candidate_models.is_empty() {
        return chat_error(
            StatusCode::FORBIDDEN,
            "Combo has no models available to this API key",
        );
    }
    if payload.get("messages").and_then(Value::as_array).is_none() {
        return chat_error(StatusCode::BAD_REQUEST, "messages must be an array");
    }
    let require_api_key = match setting_bool(&connection, "requireApiKey") {
        Ok(Some(value)) => value,
        Ok(None) => env::var("REQUIRE_API_KEY").as_deref() == Ok("true"),
        Err(error) => {
            eprintln!("[v1/chat/completions] settings lookup failed: {error}");
            return chat_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Backend database unavailable",
            );
        }
    };
    if require_api_key && let Some(api_key) = api_key.as_deref() {
        let limits = match load_api_key_limits(&connection, api_key) {
            Ok(Some(limits)) => limits,
            Ok(None) => return chat_error(StatusCode::UNAUTHORIZED, "Invalid API key"),
            Err(error) => {
                eprintln!("[v1/chat/completions] API key limits lookup failed: {error}");
                return chat_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Backend database unavailable",
                );
            }
        };
        if limits.has_ip_whitelist {
            return chat_error(
                StatusCode::FORBIDDEN,
                "Rust chat IP-whitelisted keys require a trusted client-IP proxy configuration",
            );
        }
        if let Some(max_input_tokens) = limits.max_input_tokens {
            let estimated_tokens = payload.to_string().len().div_ceil(4) as u64;
            if estimated_tokens > max_input_tokens {
                return chat_error(
                    StatusCode::BAD_REQUEST,
                    "Estimated input size exceeds this API key's limit",
                );
            }
        }
        if let Some(max_output_tokens) = limits.max_output_tokens {
            let fields = ["max_tokens", "max_completion_tokens", "max_output_tokens"];
            let mut applied = false;
            for field in fields {
                if let Some(requested) = payload.get(field).and_then(Value::as_u64) {
                    if let Some(object) = payload.as_object_mut() {
                        object.insert(
                            field.to_owned(),
                            Value::from(requested.min(max_output_tokens)),
                        );
                    }
                    applied = true;
                }
            }
            if !applied && let Some(object) = payload.as_object_mut() {
                object.insert("max_tokens".to_owned(), Value::from(max_output_tokens));
            }
        }
        if let Some(limit) = limits.requests_per_minute {
            let writable = match open_database_for_writes() {
                Ok(connection) => connection,
                Err(error) => {
                    eprintln!("[v1/chat/completions] cannot open rate-limit database: {error}");
                    return chat_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            };
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            match consume_api_key_request(&writable, &limits.id, limit, now_ms) {
                Ok((true, _)) => {}
                Ok((false, retry_after)) => return chat_rate_limit_error(retry_after),
                Err(error) => {
                    eprintln!("[v1/chat/completions] rate-limit update failed: {error}");
                    return chat_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Backend database unavailable",
                    );
                }
            }
        }
    }
    let client = reqwest::Client::new();
    let mut last_upstream = None;
    let mut last_error = None;
    let mut supported_candidates = 0;
    for (candidate_index, candidate) in candidate_models.iter().enumerate() {
        let Some((model_provider, upstream_model)) = candidate.split_once('/') else {
            if !is_combo {
                return chat_error(
                    StatusCode::BAD_REQUEST,
                    "Expected model format provider/<model>",
                );
            }
            last_error = Some(chat_error(
                StatusCode::BAD_REQUEST,
                "Invalid combo model format",
            ));
            continue;
        };
        let available = models.data.iter().find(|entry| entry.id == *candidate);
        let Some(available) = available else {
            if !is_combo {
                return chat_error(StatusCode::NOT_FOUND, "Model is not available");
            }
            continue;
        };
        if model_provider != available.owned_by {
            if !is_combo {
                return chat_error(StatusCode::BAD_REQUEST, "Model provider does not match");
            }
            continue;
        }
        let provider_id = match chat_provider_id(&connection, model_provider) {
            Ok(Some((provider_id, true))) => provider_id,
            Ok(Some((_, false))) | Ok(None) => {
                if !is_combo {
                    return chat_error(
                        StatusCode::NOT_IMPLEMENTED,
                        "Rust chat supports OpenAI-compatible providers only; this provider still uses Node",
                    );
                }
                continue;
            }
            Err(error) => {
                eprintln!("[v1/chat/completions] provider lookup failed: {error}");
                return chat_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Backend database unavailable",
                );
            }
        };
        supported_candidates += 1;
        let (connection_id, upstream_key, upstream_url) =
            match load_openai_connection(&connection, &provider_id) {
                Ok(credentials) => credentials,
                Err(error) => {
                    eprintln!("[v1/chat/completions] provider connection unavailable: {error}");
                    last_error = Some(chat_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "OpenAI-compatible provider is not configured",
                    ));
                    continue;
                }
            };
        let mut candidate_payload = payload.clone();
        if let Some(object) = candidate_payload.as_object_mut() {
            object.insert("model".to_owned(), Value::String(upstream_model.to_owned()));
        }
        let mut upstream_request = client
            .post(upstream_url)
            .bearer_auth(upstream_key)
            .header("content-type", "application/json")
            .body(candidate_payload.to_string());
        if let Some(accept) = parts.headers.get("accept") {
            upstream_request = upstream_request.header("accept", accept);
        }
        let upstream = match upstream_request.send().await {
            Ok(response) => response,
            Err(error) => {
                eprintln!("[v1/chat/completions] upstream request failed: {error}");
                last_error = Some(chat_error(
                    StatusCode::BAD_GATEWAY,
                    "OpenAI upstream request failed",
                ));
                continue;
            }
        };
        let status = upstream.status();
        let fallback_status =
            matches!(status.as_u16(), 401 | 402 | 403 | 404 | 429) || status.is_server_error();
        if is_combo && fallback_status && candidate_index + 1 < candidate_models.len() {
            last_upstream = Some((
                upstream,
                model_provider.to_owned(),
                upstream_model.to_owned(),
                connection_id,
            ));
            continue;
        }
        return finish_chat_upstream(
            upstream,
            model_provider,
            upstream_model,
            &connection_id,
            api_key.as_deref(),
        )
        .await;
    }
    if let Some((upstream, provider, model, connection_id)) = last_upstream {
        return finish_chat_upstream(
            upstream,
            &provider,
            &model,
            &connection_id,
            api_key.as_deref(),
        )
        .await;
    }
    if let Some(error) = last_error {
        return error;
    }
    if supported_candidates == 0 {
        return chat_error(
            StatusCode::NOT_IMPLEMENTED,
            "Rust chat supports OpenAI-compatible providers only; this provider still uses Node",
        );
    }
    chat_error(
        StatusCode::SERVICE_UNAVAILABLE,
        "All combo models are unavailable",
    )
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "ok": true }))
}

async fn ollama_tags() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "models": [
            {
                "name": "llama3.2",
                "modified_at": "2025-12-26T00:00:00Z",
                "size": 2000000000_u64,
                "digest": "abc123def456",
                "details": {
                    "format": "gguf",
                    "family": "llama",
                    "parameter_size": "3B",
                    "quantization_level": "Q4_K_M"
                }
            },
            {
                "name": "qwen2.5",
                "modified_at": "2025-12-26T00:00:00Z",
                "size": 4000000000_u64,
                "digest": "def456abc123",
                "details": {
                    "format": "gguf",
                    "family": "qwen",
                    "parameter_size": "7B",
                    "quantization_level": "Q4_K_M"
                }
            }
        ]
    }))
}

fn query_param(request: &Request, name: &str) -> Option<String> {
    request
        .uri()
        .query()
        .into_iter()
        .flat_map(|query| url::form_urlencoded::parse(query.as_bytes()))
        .find_map(|(key, value)| (key == name).then(|| value.into_owned()))
}

fn usage_database(request: &Request) -> Result<Connection, (StatusCode, &'static str)> {
    require_dashboard_auth(request)?;
    open_database().map_err(|error| {
        eprintln!("[api/usage] database unavailable: {error}");
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Backend database unavailable",
        )
    })
}

async fn usage_history(request: Request) -> Response {
    let connection = match usage_database(&request) {
        Ok(connection) => connection,
        Err((status, message)) => return admin_error(status, message),
    };
    let mut sql = String::from(
        "SELECT timestamp, provider, model, connectionId, apiKey, endpoint, cost, status, tokens FROM usageHistory",
    );
    let mut conditions = Vec::new();
    let mut values = Vec::<String>::new();
    for (query_name, column) in [
        ("provider", "provider"),
        ("model", "model"),
        ("startDate", "timestamp"),
        ("endDate", "timestamp"),
    ] {
        if let Some(value) = query_param(&request, query_name).filter(|value| !value.is_empty()) {
            values.push(value);
            let comparison = if query_name == "endDate" {
                "<="
            } else if query_name == "startDate" {
                ">="
            } else {
                "="
            };
            conditions.push(format!("{column} {comparison} ?{}", values.len()));
        }
    }
    if !conditions.is_empty() {
        sql.push_str(" WHERE ");
        sql.push_str(&conditions.join(" AND "));
    }
    sql.push_str(" ORDER BY id ASC");
    let mut statement = match connection.prepare(&sql) {
        Ok(statement) => statement,
        Err(error) => {
            eprintln!("[api/usage/history] query prepare failed: {error}");
            return admin_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to fetch usage history",
            );
        }
    };
    let rows = match statement.query_map(rusqlite::params_from_iter(values.iter()), |row| {
        let api_key: Option<String> = row.get(4)?;
        let raw_tokens: Option<String> = row.get(8)?;
        Ok(json!({
            "timestamp": row.get::<_, String>(0)?,
            "provider": row.get::<_, Option<String>>(1)?,
            "model": row.get::<_, Option<String>>(2)?,
            "connectionId": row.get::<_, Option<String>>(3)?,
            "apiKeyMasked": api_key.as_deref().map(mask_usage_api_key),
            "endpoint": row.get::<_, Option<String>>(5)?,
            "cost": row.get::<_, Option<f64>>(6)?.unwrap_or_default(),
            "status": row.get::<_, Option<String>>(7)?,
            "tokens": raw_tokens.and_then(|raw| serde_json::from_str::<Value>(&raw).ok()).unwrap_or(json!({}))
        }))
    }) {
        Ok(rows) => rows,
        Err(error) => {
            eprintln!("[api/usage/history] query failed: {error}");
            return admin_error(StatusCode::INTERNAL_SERVER_ERROR, "Failed to fetch usage history");
        }
    };
    let mut history = Vec::new();
    for row in rows {
        match row {
            Ok(row) => history.push(row),
            Err(error) => {
                eprintln!("[api/usage/history] row read failed: {error}");
                return admin_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to fetch usage history",
                );
            }
        }
    }
    Json(history).into_response()
}

fn valid_usage_period(period: &str) -> bool {
    matches!(period, "today" | "24h" | "7d" | "30d" | "60d" | "all")
}

async fn usage_chart(request: Request) -> Response {
    let connection = match usage_database(&request) {
        Ok(connection) => connection,
        Err((status, message)) => return admin_error(status, message),
    };
    let period = query_param(&request, "period").unwrap_or_else(|| "7d".to_owned());
    if !valid_usage_period(&period) {
        return admin_error(StatusCode::BAD_REQUEST, "Invalid period");
    }
    let mut sql = if period == "today" || period == "24h" {
        "SELECT strftime('%Y-%m-%d %H', timestamp, 'localtime') AS bucket, SUM(promptTokens + completionTokens), SUM(cost), COUNT(*) FROM usageHistory WHERE ".to_owned()
    } else {
        "SELECT dateKey, json_extract(data, '$.promptTokens') + json_extract(data, '$.completionTokens'), json_extract(data, '$.cost'), json_extract(data, '$.requests') FROM usageDaily WHERE ".to_owned()
    };
    match period.as_str() {
        "today" => sql.push_str("date(timestamp, 'localtime') = date('now', 'localtime') "),
        "24h" => sql.push_str("timestamp >= strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-24 hours') "),
        "7d" => sql.push_str("dateKey >= date('now', 'localtime', '-6 day') "),
        "30d" => sql.push_str("dateKey >= date('now', 'localtime', '-29 day') "),
        "60d" => sql.push_str("dateKey >= date('now', 'localtime', '-59 day') "),
        "all" => sql.push_str("1 = 1 "),
        _ => return admin_error(StatusCode::BAD_REQUEST, "Invalid period"),
    }
    sql.push_str("GROUP BY 1 ORDER BY 1");
    let rows = match connection.prepare(&sql).and_then(|mut statement| {
        statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<u64>>(1)?.unwrap_or_default(),
                    row.get::<_, Option<f64>>(2)?.unwrap_or_default(),
                    row.get::<_, Option<u64>>(3)?.unwrap_or_default(),
                ))
            })?
            .collect::<Result<Vec<_>, _>>()
    }) {
        Ok(rows) => rows,
        Err(error) => {
            eprintln!("[api/usage/chart] query failed: {error}");
            return admin_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to fetch chart data",
            );
        }
    };
    let data = rows
        .into_iter()
        .map(|(key, tokens, cost, requests)| {
            let label = if period == "today" || period == "24h" {
                key.rsplit(' ')
                    .next()
                    .map(|hour| format!("{hour}:00"))
                    .unwrap_or(key)
            } else {
                let parts = key.split('-').collect::<Vec<_>>();
                if parts.len() == 3 {
                    let month = parts[1].parse::<usize>().unwrap_or(0);
                    let day = parts[2].parse::<u8>().unwrap_or(0);
                    let month_name = [
                        "", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                        "Nov", "Dec",
                    ]
                    .get(month)
                    .copied()
                    .unwrap_or("");
                    format!("{month_name} {day}")
                } else {
                    key
                }
            };
            json!({"label": label, "tokens": tokens, "cost": cost, "requests": requests})
        })
        .collect::<Vec<_>>();
    Json(data).into_response()
}

async fn usage_stats(request: Request) -> Response {
    let connection = match usage_database(&request) {
        Ok(connection) => connection,
        Err((status, message)) => return admin_error(status, message),
    };
    let period = query_param(&request, "period").unwrap_or_else(|| "7d".to_owned());
    if !valid_usage_period(&period) {
        return admin_error(StatusCode::BAD_REQUEST, "Invalid period");
    }
    let cutoff = match period.as_str() {
        "today" => Some("date(timestamp, 'localtime') = date('now', 'localtime')".to_owned()),
        "24h" => Some("timestamp >= strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-24 hours')".to_owned()),
        "7d" => Some(
            "timestamp >= strftime('%Y-%m-%dT00:00:00.000Z', 'now', '-6 day', 'localtime')"
                .to_owned(),
        ),
        "30d" => Some(
            "timestamp >= strftime('%Y-%m-%dT00:00:00.000Z', 'now', '-29 day', 'localtime')"
                .to_owned(),
        ),
        "60d" => Some(
            "timestamp >= strftime('%Y-%m-%dT00:00:00.000Z', 'now', '-59 day', 'localtime')"
                .to_owned(),
        ),
        "all" => None,
        _ => return admin_error(StatusCode::BAD_REQUEST, "Invalid period"),
    };
    let mut sql = String::from(
        "SELECT timestamp, provider, model, connectionId, apiKey, endpoint, promptTokens, completionTokens, cost, status, tokens FROM usageHistory",
    );
    if let Some(cutoff) = cutoff {
        sql.push_str(" WHERE ");
        sql.push_str(&cutoff);
    }
    sql.push_str(" ORDER BY id ASC");
    let rows = match connection.prepare(&sql).and_then(|mut statement| {
        statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, Option<u64>>(6)?.unwrap_or_default(),
                    row.get::<_, Option<u64>>(7)?.unwrap_or_default(),
                    row.get::<_, Option<f64>>(8)?.unwrap_or_default(),
                    row.get::<_, Option<String>>(9)?
                        .unwrap_or_else(|| "ok".to_owned()),
                    row.get::<_, Option<String>>(10)?.unwrap_or_default(),
                ))
            })?
            .collect::<Result<Vec<_>, _>>()
    }) {
        Ok(rows) => rows,
        Err(error) => {
            eprintln!("[api/usage/stats] query failed: {error}");
            return admin_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to fetch usage stats",
            );
        }
    };
    let mut stats = json!({
        "totalRequests": 0,
        "totalPromptTokens": 0,
        "totalCompletionTokens": 0,
        "totalCachedTokens": 0,
        "totalCost": 0.0,
        "byProvider": {},
        "byModel": {},
        "byAccount": {},
        "byApiKey": {},
        "byEndpoint": {},
        "last10Minutes": [],
        "pending": {"byModel": {}, "byAccount": {}},
        "activeRequests": [],
        "recentRequests": [],
        "errorProvider": ""
    });
    let mut recent = rows.iter().rev().take(100).collect::<Vec<_>>();
    let mut seen_recent = HashSet::new();
    let mut recent_requests = Vec::new();
    for row in recent.drain(..) {
        let tokens: Value = serde_json::from_str(&row.10).unwrap_or_else(|_| json!({}));
        let prompt = tokens
            .get("prompt_tokens")
            .or_else(|| tokens.get("input_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(row.6);
        let completion = tokens
            .get("completion_tokens")
            .or_else(|| tokens.get("output_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(row.7);
        if prompt == 0 && completion == 0 {
            continue;
        }
        let minute = row.0.get(..16).unwrap_or(&row.0);
        let key = format!("{}|{}|{prompt}|{completion}|{minute}", row.2, row.1);
        if !seen_recent.insert(key) {
            continue;
        }
        recent_requests.push(json!({
            "timestamp": row.0,
            "model": row.2,
            "provider": row.1,
            "promptTokens": prompt,
            "completionTokens": completion,
            "cachedTokens": tokens.get("cached_tokens").or_else(|| tokens.get("cache_read_input_tokens")).and_then(Value::as_u64).unwrap_or_default(),
            "status": row.9
        }));
        if recent_requests.len() == 20 {
            break;
        }
    }
    stats["recentRequests"] = json!(recent_requests);
    for row in &rows {
        let tokens: Value = serde_json::from_str(&row.10).unwrap_or_else(|_| json!({}));
        let cached = tokens
            .get("cached_tokens")
            .or_else(|| tokens.get("cache_read_input_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or_default();
        stats["totalRequests"] = json!(stats["totalRequests"].as_u64().unwrap_or_default() + 1);
        stats["totalPromptTokens"] =
            json!(stats["totalPromptTokens"].as_u64().unwrap_or_default() + row.6);
        stats["totalCompletionTokens"] =
            json!(stats["totalCompletionTokens"].as_u64().unwrap_or_default() + row.7);
        stats["totalCachedTokens"] =
            json!(stats["totalCachedTokens"].as_u64().unwrap_or_default() + cached);
        stats["totalCost"] = json!(stats["totalCost"].as_f64().unwrap_or_default() + row.8);
        let model_key = format!("{} ({})", row.2, row.1);
        let api_key = row.4.as_deref().map(mask_usage_api_key);
        let api_key_key = api_key
            .as_deref()
            .map(|value| format!("{value}|{}|{}", row.2, row.1))
            .unwrap_or_else(|| "local-no-key".to_owned());
        let endpoint = row.5.as_deref().unwrap_or("Unknown");
        let endpoint_key = format!("{endpoint}|{}|{}", row.2, row.1);
        add_usage_counter(
            &mut stats["byProvider"],
            &row.1,
            row.6,
            row.7,
            cached,
            row.8,
            json!({}),
        );
        add_usage_counter(
            &mut stats["byModel"],
            &model_key,
            row.6,
            row.7,
            cached,
            row.8,
            json!({
                "rawModel": row.2, "provider": row.1, "lastUsed": row.0
            }),
        );
        if let Some(connection_id) = row.3.as_deref() {
            add_usage_counter(
                &mut stats["byAccount"],
                connection_id,
                row.6,
                row.7,
                cached,
                row.8,
                json!({
                    "rawModel": row.2, "provider": row.1, "connectionId": connection_id,
                    "accountName": connection_id, "lastUsed": row.0
                }),
            );
        }
        add_usage_counter(
            &mut stats["byApiKey"],
            &api_key_key,
            row.6,
            row.7,
            cached,
            row.8,
            json!({
                "rawModel": row.2, "provider": row.1, "apiKeyMasked": api_key,
                "keyName": api_key.as_deref().unwrap_or("Local (No API Key)"),
                "apiKeyKey": api_key.as_deref().unwrap_or("local-no-key"), "lastUsed": row.0
            }),
        );
        add_usage_counter(
            &mut stats["byEndpoint"],
            &endpoint_key,
            row.6,
            row.7,
            cached,
            row.8,
            json!({
                "endpoint": endpoint, "rawModel": row.2, "provider": row.1, "lastUsed": row.0
            }),
        );
    }
    Json(stats).into_response()
}

#[derive(Default)]
struct VersionCache {
    value: Option<String>,
    fetched_at: Option<Instant>,
    last_failure_at: Option<Instant>,
}

#[derive(serde::Deserialize)]
struct NpmVersion {
    version: Option<String>,
}

#[derive(serde::Serialize)]
struct VersionResponse {
    #[serde(rename = "currentVersion")]
    current_version: String,
    #[serde(rename = "latestVersion")]
    latest_version: Option<String>,
    #[serde(rename = "hasUpdate")]
    has_update: bool,
    #[serde(rename = "buildId")]
    build_id: Option<String>,
}

async fn fetch_latest_version() -> Result<String, String> {
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(4))
        .build()
        .map_err(|error| error.to_string())?
        .get(NPM_VERSION_URL)
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| error.to_string())?;
    let release = response
        .json::<NpmVersion>()
        .await
        .map_err(|error| error.to_string())?;

    release
        .version
        .filter(|version| !version.is_empty())
        .ok_or_else(|| "npm response did not contain a version".to_owned())
}

async fn latest_version_cached() -> Option<String> {
    static CACHE: OnceLock<Mutex<VersionCache>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(VersionCache::default()))
        .lock()
        .await;
    let now = Instant::now();

    if cache
        .fetched_at
        .is_some_and(|fetched_at| now.duration_since(fetched_at) < VERSION_CACHE_TTL)
    {
        return cache.value.clone();
    }
    if cache
        .last_failure_at
        .is_some_and(|failed_at| now.duration_since(failed_at) < VERSION_FAILURE_BACKOFF)
    {
        return cache.value.clone();
    }

    match fetch_latest_version().await {
        Ok(version) => {
            cache.value = Some(version);
            cache.fetched_at = Some(Instant::now());
            cache.last_failure_at = None;
        }
        Err(error) => {
            eprintln!("[version] npm lookup failed: {error}");
            cache.last_failure_at = Some(Instant::now());
        }
    }

    cache.value.clone()
}

fn compare_versions(a: &str, b: &str) -> Ordering {
    for (left, right) in a.split('.').zip(b.split('.')).take(3) {
        if let (Ok(left), Ok(right)) = (left.parse::<u64>(), right.parse::<u64>()) {
            match left.cmp(&right) {
                Ordering::Equal => {}
                ordering => return ordering,
            }
        }
    }
    Ordering::Equal
}

fn version_response(
    current_version: String,
    latest_version: Option<String>,
    build_id: Option<String>,
) -> VersionResponse {
    let has_update = latest_version
        .as_deref()
        .is_some_and(|latest| compare_versions(latest, &current_version) == Ordering::Greater);
    VersionResponse {
        current_version,
        latest_version,
        has_update,
        build_id,
    }
}

async fn version() -> Json<VersionResponse> {
    let skip_update_check = env::var("VANSROUTER_SKIP_UPDATE_CHECK").as_deref() == Ok("1")
        || env::var("VANROUTER_SKIP_UPDATE_CHECK").as_deref() == Ok("1");
    let latest_version = if skip_update_check {
        None
    } else {
        latest_version_cached().await
    };
    let current_version =
        env::var("VANSROUTER_VERSION").unwrap_or_else(|_| env!("CARGO_PKG_VERSION").to_owned());
    let build_id = env::var("RELEASE_BUILD_ID").ok();

    Json(version_response(current_version, latest_version, build_id))
}

async fn provider_resource_route(Path(id): Path<String>, request: Request) -> Response {
    provider_resource(request, id).await
}

async fn combo_resource_route(Path(id): Path<String>, request: Request) -> Response {
    combo_resource(request, id).await
}

async fn api_key_resource_route(Path(id): Path<String>, request: Request) -> Response {
    api_key_resource(request, id).await
}

async fn provider_node_resource_route(Path(id): Path<String>, request: Request) -> Response {
    provider_node_resource(request, id).await
}

async fn model_aliases_route_handler(request: Request) -> Response {
    model_aliases_route(request).await
}

async fn custom_models_route_handler(request: Request) -> Response {
    custom_models_route(request).await
}

async fn disabled_models_route_handler(request: Request) -> Response {
    disabled_models_route(request).await
}

fn app() -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/health", get(health))
        .route("/api/auth/login", axum::routing::post(auth::login))
        .route("/api/auth/logout", axum::routing::post(auth::logout))
        .route("/api/auth/status", get(auth::status))
        .route("/api/auth/check", get(auth::check))
        .route("/api/auth/oidc/start", get(oidc::start))
        .route("/api/auth/oidc/callback", get(oidc::callback))
        .route("/api/auth/oidc/test", axum::routing::post(oidc::test))
        .route(
            "/api/auth/reset-password",
            axum::routing::post(auth::reset_password),
        )
        .route("/api/tags", get(ollama_tags))
        .route("/api/version", get(version))
        .route("/api/usage/history", get(usage_history))
        .route("/api/usage/stats", get(usage_stats))
        .route("/api/usage/chart", get(usage_chart))
        .route("/api/providers", any(providers_collection))
        .route("/api/providers/{id}", any(provider_resource_route))
        .route("/api/keys", any(api_keys_collection))
        .route("/api/keys/{id}", any(api_key_resource_route))
        .route("/api/provider-nodes", any(provider_nodes_collection))
        .route(
            "/api/provider-nodes/{id}",
            any(provider_node_resource_route),
        )
        .route("/api/models/alias", any(model_aliases_route_handler))
        .route("/api/models/custom", any(custom_models_route_handler))
        .route("/api/models/disabled", any(disabled_models_route_handler))
        .route("/api/combos", any(combos_collection))
        .route("/api/combos/{id}", any(combo_resource_route))
        .route("/v1/models", get(v1_models))
        .route(
            "/v1/chat/completions",
            axum::routing::post(chat_completions),
        )
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([
                    Method::GET,
                    Method::POST,
                    Method::PUT,
                    Method::DELETE,
                    Method::OPTIONS,
                ])
                .allow_headers(Any),
        )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port = env::var("PORT")
        .unwrap_or_else(|_| "8080".to_owned())
        .parse::<u16>()?;
    let address = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(address).await?;

    println!("Rust backend listening on {address}");
    axum::serve(
        listener,
        app().into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode},
    };
    use jsonwebtoken::{EncodingKey, Header, encode};
    use rusqlite::Connection;
    use tower::ServiceExt;

    fn models_test_database(allowed_providers: &str, allowed_combos: &str) -> Connection {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE settings (id INTEGER PRIMARY KEY, data TEXT NOT NULL);
                 CREATE TABLE apiKeys (
                   id TEXT NOT NULL,
                   key TEXT PRIMARY KEY,
                   isActive INTEGER NOT NULL,
                   allowedProviders TEXT,
                   allowedCombos TEXT,
                   allowedKinds TEXT,
                   requestsPerMinute INTEGER,
                   maxInputTokens INTEGER,
                   maxOutputTokens INTEGER,
                   ipWhitelist TEXT
                 );
                 CREATE TABLE apiKeyRateLimits (
                   apiKeyId TEXT PRIMARY KEY,
                   windowStart INTEGER NOT NULL,
                   requestCount INTEGER NOT NULL
                 );
                 CREATE TABLE providerConnections (
                   provider TEXT NOT NULL,
                   isActive INTEGER NOT NULL,
                   data TEXT NOT NULL
                 );
                 CREATE TABLE providerNodes (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                 CREATE TABLE combos (
                   name TEXT NOT NULL,
                   kind TEXT,
                   createdAt TEXT NOT NULL
                 );
                 CREATE TABLE kv (scope TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL);
                 INSERT INTO settings (id, data) VALUES (1, '{\"requireApiKey\":true}');
                 INSERT INTO providerConnections (provider, isActive, data)
                   VALUES ('openai', 1, '{}');
                 INSERT INTO combos (name, kind, createdAt) VALUES
                   ('good', NULL, '2026-01-01'),
                   ('blocked', NULL, '2026-01-02');",
            )
            .unwrap_or_else(|error| panic!("test schema setup failed: {error}"));
        connection
            .execute(
                "INSERT INTO apiKeys
                 (id, key, isActive, allowedProviders, allowedCombos, allowedKinds)
                 VALUES ('test-key-id', 'test-key', 1, ?1, ?2, '[\"llm\"]')",
                params![allowed_providers, allowed_combos],
            )
            .unwrap();
        connection
    }

    #[tokio::test]
    async fn api_health_returns_ok_json() {
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/api/health")
                    .header("origin", "https://dashboard.example")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("access-control-allow-origin")
                .unwrap(),
            "*"
        );
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).unwrap(),
            serde_json::json!({ "ok": true })
        );
    }

    #[test]
    fn api_key_generation_matches_machine_key_format_and_hmac() {
        let secret = b"test-api-key-secret";
        let machine_id = "0123456789abcdef";
        let key = generate_api_key(machine_id, secret).unwrap();
        let parts: Vec<_> = key.split('-').collect();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0], "sk");
        assert_eq!(parts[1], machine_id);
        assert_eq!(parts[2].len(), 6);
        assert!(
            parts[2]
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        );
        let expected_crc = hmac_sha256_hex(secret, format!("{machine_id}{}", parts[2]).as_bytes());
        assert_eq!(parts[3], &expected_crc[..8]);
        assert_eq!(
            hmac_sha256_hex(b"key", b"The quick brown fox jumps over the lazy dog"),
            "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8"
        );
    }

    #[test]
    fn api_key_limits_and_ip_whitelists_are_normalized() {
        assert_eq!(
            api_key_limit(Some(&json!("1200")), "maxInputTokens").unwrap(),
            Some(1200)
        );
        assert_eq!(
            api_key_limit(Some(&json!("")), "maxInputTokens").unwrap(),
            None
        );
        assert!(api_key_limit(Some(&json!(0)), "requestsPerMinute").is_err());
        assert!(api_key_limit(Some(&json!(1.5)), "maxOutputTokens").is_err());

        assert_eq!(
            normalize_ip_whitelist(Some(&json!("192.0.2.4\n2001:0db8::1, ::ffff:192.0.2.4")))
                .unwrap(),
            Some(vec!["192.0.2.4".to_owned(), "2001:db8::1".to_owned()])
        );
        assert!(normalize_ip_whitelist(Some(&json!("example.com"))).is_err());
        assert!(!json_truthy(&Value::Null));
        assert!(!json_truthy(&json!(0)));
        assert!(!json_truthy(&json!("")));
        assert!(json_truthy(&json!("false")));
        assert!(json_truthy(&json!([])));
    }

    #[test]
    fn provider_node_base_urls_match_node_route_normalization() {
        assert_eq!(
            sanitize_provider_node_base_url(
                "anthropic-compatible",
                "https://gateway.example/messages/"
            ),
            "https://gateway.example/v1"
        );
        assert_eq!(
            sanitize_provider_node_base_url("anthropic-compatible", "https://gateway.example/v1"),
            "https://gateway.example/v1"
        );
        assert_eq!(
            sanitize_provider_node_base_url(
                "custom-embedding",
                "https://gateway.example/embeddings/"
            ),
            "https://gateway.example"
        );
        assert_eq!(
            sanitize_provider_node_base_url("openai-compatible", "https://gateway.example/v1/"),
            "https://gateway.example/v1/"
        );
    }

    #[tokio::test]
    async fn api_errors_use_openai_compatible_error_types() {
        for (status, expected_type) in [
            (StatusCode::UNAUTHORIZED, "authentication_error"),
            (StatusCode::FORBIDDEN, "permission_error"),
            (StatusCode::BAD_REQUEST, "invalid_request_error"),
            (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error"),
        ] {
            let response = api_error(status, "test error");
            assert_eq!(response.status(), status);
            let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(value["error"]["type"], expected_type);
        }
    }

    #[test]
    fn version_response_matches_dashboard_contract() {
        let response = version_response(
            "1.2.3".to_owned(),
            Some("1.2.4".to_owned()),
            Some("build-123".to_owned()),
        );
        let value = serde_json::to_value(response).unwrap();

        assert_eq!(
            value,
            serde_json::json!({
                "currentVersion": "1.2.3",
                "latestVersion": "1.2.4",
                "hasUpdate": true,
                "buildId": "build-123"
            })
        );
    }

    #[test]
    fn dashboard_auth_accepts_only_valid_unexpired_hs256_tokens() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let claims = json!({
            "authenticated": true,
            "iat": now,
            "exp": now + 60,
        });
        let token = encode(
            &Header::new(Algorithm::HS256),
            &claims,
            &EncodingKey::from_secret(b"test-secret"),
        )
        .unwrap();
        assert!(dashboard_token_valid(&token, "test-secret"));
        assert!(!dashboard_token_valid(&token, "wrong-secret"));

        let expired = encode(
            &Header::new(Algorithm::HS256),
            &json!({"authenticated":true,"iat":now-120,"exp":now-60}),
            &EncodingKey::from_secret(b"test-secret"),
        )
        .unwrap();
        assert!(!dashboard_token_valid(&expired, "test-secret"));

        let not_yet_valid = encode(
            &Header::new(Algorithm::HS256),
            &json!({"authenticated":true,"iat":now,"nbf":now+60,"exp":now+120}),
            &EncodingKey::from_secret(b"test-secret"),
        )
        .unwrap();
        assert!(!dashboard_token_valid(&not_yet_valid, "test-secret"));
    }

    #[test]
    fn rust_data_directory_matches_node_production_fallback_rules() {
        let persistent = PathBuf::from("/home/user/.9router");
        assert_eq!(
            resolve_data_dir(
                Some("/tmp/9router-data-smoke"),
                persistent.clone(),
                true,
                false,
                false,
                false,
            ),
            persistent
        );
        assert_eq!(
            resolve_data_dir(
                Some("/tmp/9router-data-smoke"),
                PathBuf::from("/home/user/.9router"),
                true,
                true,
                false,
                false,
            ),
            PathBuf::from("/tmp/9router-data-smoke")
        );
        assert_eq!(
            resolve_data_dir(
                None,
                PathBuf::from("/home/user/.9router"),
                false,
                false,
                false,
                false,
            ),
            PathBuf::from("/home/user/.9router")
        );
        assert_eq!(
            resolve_data_dir(
                Some("/var/lib/9router"),
                PathBuf::from("/home/user/.9router"),
                true,
                false,
                true,
                false,
            ),
            PathBuf::from("/home/user/.9router")
        );
    }

    #[test]
    fn combo_validation_and_timestamps_match_database_contract() {
        assert!(combo_name_valid("primary.v2_test-1"));
        assert!(!combo_name_valid("bad/name"));
        assert_eq!(combo_context_length(Some(&json!(8192))), Ok(Some(8192)));
        assert_eq!(combo_context_length(Some(&json!("8192"))), Ok(Some(8192)));
        assert_eq!(combo_context_length(Some(&Value::Null)), Ok(None));
        assert_eq!(combo_context_length(Some(&json!(""))), Ok(None));
        assert!(combo_context_length(Some(&json!(2_000_001))).is_err());
        assert!(
            regex_timestamp(&utc_timestamp()),
            "timestamps must sort and parse like Node ISO timestamps"
        );
    }

    #[test]
    fn chat_usage_updates_history_daily_summary_and_lifetime_counter() {
        let mut connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE usageHistory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    connectionId TEXT,
                    apiKey TEXT,
                    endpoint TEXT,
                    promptTokens INTEGER DEFAULT 0,
                    completionTokens INTEGER DEFAULT 0,
                    cost REAL DEFAULT 0,
                    status TEXT,
                    tokens TEXT,
                    meta TEXT
                 );
                 CREATE TABLE usageDaily (dateKey TEXT PRIMARY KEY, data TEXT NOT NULL);
                 CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
            )
            .unwrap();
        let usage = json!({
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 40}
        });

        persist_chat_usage(
            &mut connection,
            "openai",
            "gpt-4.1",
            "connection-1",
            Some("sk-client"),
            "/v1/chat/completions",
            &usage,
            120,
            30,
            40,
            0.0006,
        )
        .unwrap();

        let history: (String, i64, i64, f64, String) = connection
            .query_row(
                "SELECT provider, promptTokens, completionTokens, cost, tokens FROM usageHistory",
                [],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(history.0, "openai");
        assert_eq!((history.1, history.2), (120, 30));
        assert!((history.3 - 0.0006).abs() < f64::EPSILON);
        assert_eq!(serde_json::from_str::<Value>(&history.4).unwrap(), usage);

        let (day_data, total): (String, String) = connection
            .query_row(
                "SELECT usageDaily.data, _meta.value FROM usageDaily, _meta WHERE _meta.key = 'totalRequestsLifetime'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        let day: Value = serde_json::from_str(&day_data).unwrap();
        assert_eq!(day["requests"], 1);
        assert_eq!(day["promptTokens"], 120);
        assert_eq!(day["completionTokens"], 30);
        assert_eq!(day["cachedTokens"], 40);
        assert_eq!(day["cost"], 0.0006);
        assert_eq!(day["byProvider"]["openai"]["requests"], 1);
        assert_eq!(day["byAccount"]["connection-1"]["rawModel"], "gpt-4.1");
        assert_eq!(total, "1");
    }

    #[test]
    fn chat_cost_uses_canonical_and_custom_pricing() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE kv (scope TEXT, key TEXT, value TEXT, PRIMARY KEY(scope, key));
                 INSERT INTO kv(scope, key, value) VALUES('pricing', 'openai', '{\"gpt-4.1\":{\"input\":4.0}}');",
            )
            .unwrap();
        let usage = json!({
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 40}
        });
        let canonical = model_pricing(&connection, "anthropic", "gpt-4.1")
            .unwrap()
            .unwrap();
        assert_eq!(calculate_chat_cost(&usage, &canonical), 0.00055);
        let custom = model_pricing(&connection, "openai", "gpt-4.1")
            .unwrap()
            .unwrap();
        assert_eq!(custom["input"], 4.0);
        assert_eq!(custom["output"], 10.0);
        assert_eq!(calculate_chat_cost(&usage, &custom), 0.00067);
    }

    #[test]
    fn streaming_usage_parser_handles_chunk_boundaries_and_crlf_events() {
        let mut pending = b"data: {\"choices\":[]}\r\n\r\n".to_vec();
        assert!(extract_sse_usage(&mut pending, false).is_none());
        pending.extend_from_slice(b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2}}\r\n\r\n");
        let usage = extract_sse_usage(&mut pending, false).unwrap();
        assert_eq!(usage["prompt_tokens"], 4);
        assert_eq!(usage["completion_tokens"], 2);

        pending
            .extend_from_slice(b"data: {\"usage\":{\"prompt_tokens\":8,\"completion_tokens\":3}}");
        assert_eq!(
            extract_sse_usage(&mut pending, true).unwrap()["prompt_tokens"],
            8
        );
    }

    #[test]
    fn combo_chat_loads_members_and_uses_combo_specific_strategy() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                r#"CREATE TABLE combos (name TEXT, models TEXT);
                   CREATE TABLE settings (id INTEGER PRIMARY KEY, data TEXT);
                   INSERT INTO combos (name, models) VALUES ('chat-test', '["openai/a", "node/b"]');
                   INSERT INTO settings (id, data) VALUES (1, '{"comboStrategy":"fallback","comboStrategies":{"chat-test":{"fallbackStrategy":"round-robin"}}}');"#,
            )
            .unwrap();

        let (name, models, strategy) = combo_models_for_chat(&connection, "combo/chat-test")
            .unwrap()
            .unwrap();
        assert_eq!(name, "chat-test");
        assert_eq!(models, ["openai/a", "node/b"]);
        assert_eq!(strategy, "round-robin");
        assert!(
            combo_models_for_chat(&connection, "openai/gpt-4.1")
                .unwrap()
                .is_none()
        );
    }

    fn regex_timestamp(value: &str) -> bool {
        let bytes = value.as_bytes();
        bytes.len() == 24
            && bytes[4] == b'-'
            && bytes[7] == b'-'
            && bytes[10] == b'T'
            && bytes[13] == b':'
            && bytes[16] == b':'
            && bytes[19] == b'.'
            && bytes[23] == b'Z'
    }

    #[test]
    fn provider_connection_responses_never_include_credentials() {
        let db = Connection::open_in_memory().unwrap();
        db.execute_batch(
            "CREATE TABLE providerConnections (
               id TEXT PRIMARY KEY, provider TEXT NOT NULL, authType TEXT NOT NULL,
               name TEXT, email TEXT, priority INTEGER, isActive INTEGER,
               data TEXT NOT NULL, createdAt TEXT NOT NULL, updatedAt TEXT NOT NULL
             );
             INSERT INTO providerConnections
               VALUES ('secret-row', 'openai', 'apikey', 'OpenAI', NULL, 1, 1,
                 '{\"apiKey\":\"gateway-provider-secret\",\"accessToken\":\"oauth-secret\",\"providerSpecificData\":{}}',
                 '2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z');",
        )
        .unwrap();

        let response = provider_connection_value(&db, "secret-row")
            .unwrap()
            .unwrap();
        let serialized = response.to_string();
        assert!(!serialized.contains("gateway-provider-secret"));
        assert!(!serialized.contains("oauth-secret"));
        assert_eq!(response["provider"], "openai");
        assert_eq!(response["isActive"], true);
    }

    #[test]
    fn version_comparison_checks_numeric_segments() {
        assert_eq!(compare_versions("1.10.0", "1.9.9"), Ordering::Greater);
        assert_eq!(compare_versions("1.2.3", "1.2.3"), Ordering::Equal);
        assert_eq!(compare_versions("1.2.3", "1.2.4"), Ordering::Less);
    }

    #[test]
    fn null_acl_allows_and_empty_acl_denies() {
        let unrestricted = ApiKeyAcl::default();
        assert!(kind_allowed(&unrestricted, "llm"));
        assert!(combo_allowed(&unrestricted, "primary"));

        let restricted = ApiKeyAcl {
            allowed_providers: Some(vec![]),
            allowed_combos: Some(vec![]),
            allowed_kinds: Some(vec![]),
        };
        assert!(!kind_allowed(&restricted, "llm"));
        assert!(!combo_allowed(&restricted, "primary"));
    }

    #[test]
    fn v1_models_requires_valid_key_and_applies_acl_filters() {
        let db = models_test_database("[\"openai\"]", "[\"good\"]");
        assert_eq!(
            read_v1_models_from_connection(&db, None, None, true)
                .unwrap_err()
                .0,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            read_v1_models_from_connection(&db, Some("invalid".to_owned()), None, true)
                .unwrap_err()
                .0,
            StatusCode::UNAUTHORIZED
        );

        let models =
            read_v1_models_from_connection(&db, Some("test-key".to_owned()), None, true).unwrap();
        assert!(models.data.iter().any(|model| model.owned_by == "openai"));
        assert!(models.data.iter().any(|model| model.id == "combo/good"));
        assert!(!models.data.iter().any(|model| model.id == "combo/blocked"));
    }

    #[test]
    fn remote_api_gate_requires_key_unless_explicitly_opened() {
        assert!(public_api_access_allowed(true, false, false, true, false));
        assert!(public_api_access_allowed(false, true, false, true, false));
        assert!(public_api_access_allowed(false, false, true, true, false));
        assert!(!public_api_access_allowed(
            false, false, false, false, false
        ));
        assert!(public_api_access_allowed(false, false, false, false, true));
        assert!(!public_api_access_allowed(false, false, false, true, true));
    }

    #[test]
    fn api_key_rate_limit_is_atomic_and_resets_each_minute() {
        let db = models_test_database("[\"openai\"]", "[\"good\"]");
        assert_eq!(
            consume_api_key_request(&db, "key-id", 1, 60_001).unwrap(),
            (true, 0)
        );
        let (allowed, retry_after) = consume_api_key_request(&db, "key-id", 1, 60_500).unwrap();
        assert!(!allowed);
        assert_eq!(retry_after, 60);
        assert_eq!(
            consume_api_key_request(&db, "key-id", 1, 120_000).unwrap(),
            (true, 0)
        );
    }

    #[test]
    fn local_access_is_not_inferred_through_forwarding_headers() {
        let local_peer = "127.0.0.1:8080".parse().unwrap();
        let remote_peer = "203.0.113.10:8080".parse().unwrap();
        let direct_local_request = Request::builder().body(Body::empty()).unwrap();
        assert!(is_local_request(&direct_local_request, local_peer));
        assert!(!is_local_request(&direct_local_request, remote_peer));

        let forwarded_request = Request::builder()
            .header("x-forwarded-for", "203.0.113.20")
            .body(Body::empty())
            .unwrap();
        assert!(!is_local_request(&forwarded_request, local_peer));

        let remote_origin_request = Request::builder()
            .header("origin", "https://dashboard.example")
            .body(Body::empty())
            .unwrap();
        assert!(!is_local_request(&remote_origin_request, local_peer));
    }

    #[test]
    fn provider_lookup_normalizes_names_aliases_and_custom_nodes() {
        assert_eq!(
            provider_definition(" OpenAI ").map(|provider| provider.id.as_str()),
            Some("openai")
        );
        assert_eq!(
            provider_definition("aai").map(|provider| provider.id.as_str()),
            Some("assemblyai")
        );

        let db = Connection::open_in_memory().unwrap();
        db.execute_batch(
            "CREATE TABLE providerNodes (
               id TEXT PRIMARY KEY, type TEXT, name TEXT, data TEXT NOT NULL
             );
             INSERT INTO providerNodes VALUES (
               'openai-compatible-test', 'openai-compatible', 'Test Node',
               '{\"prefix\":\"demo\",\"apiType\":\"chat\",\"baseUrl\":\"https://api.example/v1\"}'
             );",
        )
        .unwrap();
        assert_eq!(
            provider_node_definition(&db, "openai-compatible-test").unwrap(),
            Some((
                "openai-compatible".to_owned(),
                "Test Node".to_owned(),
                json!({
                    "prefix": "demo",
                    "apiType": "chat",
                    "baseUrl": "https://api.example/v1"
                })
            ))
        );
        assert_eq!(
            chat_provider_id(&db, "demo").unwrap(),
            Some(("openai-compatible-test".to_owned(), true))
        );
    }

    #[test]
    fn bundled_catalog_contains_provider_models() {
        let catalog = model_catalog();
        assert!(!catalog.providers.is_empty());
        let openai = catalog
            .providers
            .iter()
            .find(|provider| provider.id == "openai")
            .expect("OpenAI provider must be in bundled catalog");
        assert_eq!(openai.alias, "openai");
        assert_eq!(openai.category.as_deref(), Some("apikey"));
        assert_eq!(openai.display_name, "OpenAI");
        assert!(openai.auth_modes.iter().any(|mode| mode == "apikey"));
        assert!(!openai.models.is_empty());
        assert_eq!(
            provider_definition("aai").map(|provider| provider.id.as_str()),
            Some("assemblyai")
        );
    }

    #[tokio::test]
    async fn ollama_tags_matches_existing_response() {
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/api/tags")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(
            value,
            serde_json::json!({
                "models": [
                    {
                        "name": "llama3.2",
                        "modified_at": "2025-12-26T00:00:00Z",
                        "size": 2000000000_u64,
                        "digest": "abc123def456",
                        "details": {
                            "format": "gguf",
                            "family": "llama",
                            "parameter_size": "3B",
                            "quantization_level": "Q4_K_M"
                        }
                    },
                    {
                        "name": "qwen2.5",
                        "modified_at": "2025-12-26T00:00:00Z",
                        "size": 4000000000_u64,
                        "digest": "def456abc123",
                        "details": {
                            "format": "gguf",
                            "family": "qwen",
                            "parameter_size": "7B",
                            "quantization_level": "Q4_K_M"
                        }
                    }
                ]
            })
        );
    }
}
