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
  test("sends the path under the API base, /api by default", async () => {
    const fake = respondWith(json({ data: {} }));
    await apiFetch("/health/check");
    expect(fake.mock.calls[0]?.[0]).toBe("/api/health/check");
  });

  test("unwraps the project envelope", async () => {
    respondWith(json({ data: { status: "ok" } }));
    expect(await apiFetch("/x")).toEqual({
      data: { status: "ok" },
      error: null,
    });
  });

  test("takes a body without an envelope as it is", async () => {
    respondWith(json({ status: "ok" }));
    expect((await apiFetch("/x")).data).toEqual({ status: "ok" });
  });

  test("a bare GET carries no JSON content type, a body does", async () => {
    const fake = respondWith(json({}));
    await apiFetch("/x");
    await apiFetch("/x", { method: "POST", body: JSON.stringify({ a: 1 }) });
    const headersOf = (call: number) =>
      fake.mock.calls[call]?.[1]?.headers as Record<string, string>;
    expect(headersOf(0)["Content-Type"]).toBeUndefined();
    expect(headersOf(1)["Content-Type"]).toBe("application/json");
  });

  test("a 204 or an empty body is a success with no data, not a network error", async () => {
    respondWith(new Response(null, { status: 204 }));
    expect(await apiFetch("/x")).toEqual({ data: null, error: null });
    respondWith(new Response("", { status: 200 }));
    expect(await apiFetch("/x")).toEqual({ data: null, error: null });
  });

  test("a failure reports the envelope's message and code", async () => {
    respondWith(json({ message: "Nope", code: 4001 }, 400));
    const { data, error } = await apiFetch("/x");
    expect(data).toBeNull();
    expect(error?.message).toBe("Nope");
    expect(error?.code).toBe(4001);
  });

  test("a failure in FastAPI's own shape reports its detail", async () => {
    respondWith(json({ detail: "Not Found" }, 404));
    const { error } = await apiFetch("/x");
    expect(error?.message).toBe("Not Found");
    expect(error?.code).toBe(404);
  });

  test("a failure without a JSON body still says something", async () => {
    respondWith(new Response("<html>", { status: 502 }));
    expect((await apiFetch("/x")).error?.message).toBe("Request failed (502)");
  });

  test("never throws: an unreachable API is a network error", async () => {
    respondWith(new TypeError("fetch failed"));
    expect(await apiFetch("/x")).toEqual({
      data: null,
      error: { message: "Network error" },
    });
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
