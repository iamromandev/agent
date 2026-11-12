import { z } from "zod";

/**
 * The build's public configuration, as a schema of its own so it can be tested
 * without a Vite build around it.
 *
 * A blank value counts as absent, so an empty `.env` line means "use the
 * default" rather than a startup error.
 */
export const clientEnvSchema = z.object({
  VITE_API_BASE_URL: z.preprocess(
    (value) =>
      typeof value === "string" && value.trim() === "" ? undefined : value,
    z
      .string()
      .trim()
      .regex(
        /^(\/|https?:\/\/)/,
        "must be a path starting with / or an http(s) URL",
      )
      .optional(),
  ),
});

export type ClientEnv = z.infer<typeof clientEnvSchema>;

/** Where the API answers by default: `/api` on the page's own origin. */
export const DEFAULT_API_BASE_URL = "/api";

/**
 * The base every API path is appended to, without a trailing slash — so
 * `${base}${path}` never doubles one up.
 */
export function apiBaseUrlFrom(value: string | undefined): string {
  return (value ?? DEFAULT_API_BASE_URL).replace(/\/+$/, "");
}
