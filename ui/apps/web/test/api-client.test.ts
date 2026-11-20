import { afterEach, describe, expect, mock, test } from "bun:test";

import { apiFetch } from "@/lib/api/api-client";
import { getHealth } from "@/lib/api/health-client";

const realFetch = globalThis.fetch;

/** Answer every `fetch` with `response`, recording what was asked for. */
function respondWith(response: Response | Error) {
  const fake = mock(
    async (_input: string | URL | Request, _init?: RequestInit) => {
      if (response instanceof Error) {
        throw response;
      }
      return response;
    },
  );
  globalThis.fetch = fake as unknown as typeof fetch;
  return fake;
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  globalThis.fetch = realFetch;
});

describe("apiFetch", () => {
  test("a bare GET carries no JSON content type, a body does", async () => {
    const fake = respondWith(json({}));
    await apiFetch("/x");
    await apiFetch("/x", { method: "POST", body: JSON.stringify({ a: 1 }) });
    const headersOf = (call: number) =>
      fake.mock.calls[call]?.[1]?.headers as Record<string, string>;
    expect(headersOf(0)["Content-Type"]).toBeUndefined();
    expect(headersOf(1)["Content-Type"]).toBe("application/json");
  });

  test("a failure reports the envelope's message and code", async () => {
    respondWith(json({ message: "Nope", code: 4001 }, 400));
    const { data, error } = await apiFetch("/x");
    expect(data).toBeNull();
    expect(error?.message).toBe("Nope");
    expect(error?.code).toBe(4001);
  });
});

describe("getHealth", () => {
  test("asks GET /health/check through the proxy prefix", async () => {
    const fake = respondWith(
      json({ data: { status: "ok", version: "0.0.1" } }),
    );
    const { data } = await getHealth();
    expect(fake.mock.calls[0]?.[0]).toBe("/api/health/check");
    expect(fake.mock.calls[0]?.[1]?.method).toBeUndefined();
    expect(data).toEqual({ status: "ok", version: "0.0.1" });
  });
});
