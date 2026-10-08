import { beforeEach, describe, expect, it, vi } from "vitest";

const { resetComboRotation, invalidateAllowedModelsCache } = vi.hoisted(() => ({
  resetComboRotation: vi.fn(),
  invalidateAllowedModelsCache: vi.fn(),
}));

vi.mock("open-sse/services/combo.js", () => ({ resetComboRotation }));
vi.mock("@/sse/services/allowedModels.js", () => ({ invalidateAllowedModelsCache }));

import {
  isRustBackendEnabled,
  isRustChatModelSupported,
  proxyRustAdminRequest,
  proxyRustAuthRequest,
  proxyRustChatRequest,
} from "@/lib/rustBackendProxy.js";

describe("proxyRustAdminRequest combo cache invalidation", () => {
  beforeEach(() => {
    vi.stubEnv("NEXT_PUBLIC_RUST_API_URL", "https://rust.example/");
    vi.stubGlobal("fetch", vi.fn(async () => new Response('{"success":true}', { status: 200 })));
    resetComboRotation.mockClear();
    invalidateAllowedModelsCache.mockClear();
  });

  describe("proxyRustAuthRequest", () => {
    beforeEach(() => {
      vi.stubEnv("NEXT_PUBLIC_RUST_API_URL", "https://rust.example/");
      vi.stubGlobal("fetch", vi.fn(async () => new Response('{"success":true}', {
        status: 200,
        headers: [
          ["content-type", "application/json"],
          ["set-cookie", "auth_token=signed-token; Path=/; HttpOnly; SameSite=Lax; Max-Age=86400; Secure"],
        ],
      })));
    });

    it("uses one RUST_BE flag to enable Rust routing", () => {
      vi.stubEnv("RUST_BE", "true");
      expect(isRustBackendEnabled()).toBe(true);
      vi.stubEnv("RUST_BE", "false");
      expect(isRustBackendEnabled()).toBe(false);
    });

    it("routes only chat models Rust currently supports", () => {
      expect(isRustChatModelSupported("openai/gpt-4.1")).toBe(true);
      expect(isRustChatModelSupported("anthropic/claude-sonnet")).toBe(false);
      expect(isRustChatModelSupported("gpt-4.1")).toBe(false);
    });

    it("maps the Next chat API path to Rust's OpenAI chat endpoint", async () => {
      vi.stubEnv("NEXT_PUBLIC_RUST_API_URL", "https://rust.example/");
      const request = new Request("https://dashboard.example/api/v1/chat/completions?trace=1", {
        method: "POST",
        body: JSON.stringify({ model: "openai/gpt-4.1" }),
      });

      const response = await proxyRustChatRequest(request);

      expect(response.status).toBe(200);
      expect(fetch.mock.calls.at(-1)[0]).toBe("https://rust.example/v1/chat/completions?trace=1");
    });

    it("forwards request cookies and preserves Rust session cookies", async () => {
      const request = new Request("https://dashboard.example/api/auth/login", {
        method: "POST",
        headers: { cookie: "password_change=bootstrap", "content-type": "application/json" },
        body: JSON.stringify({ password: "correct-horse" }),
      });

      const response = await proxyRustAuthRequest(request);
      const [, options] = fetch.mock.calls[0];

      expect(response.status).toBe(200);
      expect(options.headers.get("cookie")).toBe("password_change=bootstrap");
      expect(response.headers.get("set-cookie")).toContain("auth_token=signed-token");
    });

    it("preserves Rust OIDC redirects through the Next.js proxy", async () => {
      vi.stubGlobal("fetch", vi.fn(async () => new Response(null, {
        status: 302,
        headers: { location: "https://issuer.example/authorize?state=opaque" },
      })));
      const request = new Request("https://dashboard.example/api/auth/oidc/start", {
        headers: { "sec-fetch-mode": "navigate", "sec-fetch-dest": "document" },
      });

      const response = await proxyRustAuthRequest(request);

      expect(response.status).toBe(302);
      expect(response.headers.get("location")).toBe("https://issuer.example/authorize?state=opaque");
    });
  });

  it.each([
    ["POST", "/api/combos"],
    ["PUT", "/api/combos/combo-id"],
    ["DELETE", "/api/combos/combo-id"],
  ])("invalidates local combo state after successful %s %s", async (method, path) => {
    const request = new Request(`https://dashboard.example${path}?source=test`, {
      method,
      headers: { cookie: "auth_token=session", "x-9r-cli-token": "cli-token" },
      ...(method === "DELETE" ? {} : { body: "{}" }),
    });

    const response = await proxyRustAdminRequest(request);

    expect(response.status).toBe(200);
    expect(resetComboRotation).toHaveBeenCalledOnce();
    expect(invalidateAllowedModelsCache).toHaveBeenCalledOnce();
    const [url, options] = fetch.mock.calls[0];
    expect(url).toBe(`https://rust.example${path}?source=test`);
    expect(options.headers.get("cookie")).toBe("auth_token=session");
    expect(options.headers.get("x-9r-cli-token")).toBe("cli-token");
  });

  it("does not invalidate local caches when Rust rejects a mutation", async () => {
    fetch.mockResolvedValueOnce(new Response('{"error":"invalid"}', { status: 400 }));
    const response = await proxyRustAdminRequest(new Request("https://dashboard.example/api/combos", {
      method: "POST",
      body: "{}",
    }));

    expect(response.status).toBe(400);
    expect(resetComboRotation).not.toHaveBeenCalled();
    expect(invalidateAllowedModelsCache).not.toHaveBeenCalled();
  });

  it("does not invalidate local caches for reads or unrelated routes", async () => {
    await proxyRustAdminRequest(new Request("https://dashboard.example/api/combos"));
    await proxyRustAdminRequest(new Request("https://dashboard.example/api/keys?active=true"));
    await proxyRustAdminRequest(new Request("https://dashboard.example/api/providers", {
      method: "POST",
      body: "{}",
    }));
    await proxyRustAdminRequest(new Request("https://dashboard.example/api/provider-nodes", {
      method: "POST",
      body: "{}",
    }));

    expect(resetComboRotation).not.toHaveBeenCalled();
    expect(invalidateAllowedModelsCache).not.toHaveBeenCalled();
  });

  it.each([
    ["PUT", "/api/models/alias"],
    ["POST", "/api/models/custom"],
    ["DELETE", "/api/models/disabled?providerAlias=openai&id=model-a"],
  ])("invalidates allowed-model cache after successful %s %s", async (method, path) => {
    const response = await proxyRustAdminRequest(new Request(`https://dashboard.example${path}`, {
      method,
      ...(method === "DELETE" ? {} : { body: "{}" }),
    }));

    expect(response.status).toBe(200);
    expect(invalidateAllowedModelsCache).toHaveBeenCalledOnce();
    expect(resetComboRotation).not.toHaveBeenCalled();
  });
});
