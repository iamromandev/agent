import { describe, expect, test } from "bun:test";

import { apiBaseUrlFrom, clientEnvSchema } from "@/config/env-schema";

describe("the API base URL", () => {
  test("a build without one talks to /api on its own origin", () => {
    const parsed = clientEnvSchema.parse({});
    expect(parsed.VITE_API_BASE_URL).toBeUndefined();
    expect(apiBaseUrlFrom(parsed.VITE_API_BASE_URL)).toBe("/api");
  });

  test("a blank value counts as absent", () => {
    const parsed = clientEnvSchema.parse({ VITE_API_BASE_URL: "  " });
    expect(parsed.VITE_API_BASE_URL).toBeUndefined();
  });

  test("a full URL is kept, without a trailing slash", () => {
    const parsed = clientEnvSchema.parse({
      VITE_API_BASE_URL: "https://api.example.com/",
    });
    expect(apiBaseUrlFrom(parsed.VITE_API_BASE_URL)).toBe(
      "https://api.example.com",
    );
  });

  test("a value that is neither a path nor an http(s) URL is refused", () => {
    expect(
      clientEnvSchema.safeParse({ VITE_API_BASE_URL: "api.example.com" })
        .success,
    ).toBe(false);
  });
});
