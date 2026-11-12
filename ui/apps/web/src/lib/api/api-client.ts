import { getApiBaseUrl } from "@/config/env";

/**
 * The one place in the app that calls `fetch`.
 *
 * Every API call goes through `apiFetch`, so the base URL, the headers and the
 * shape of a failure are decided once. A call never throws: it resolves to
 * `{ data, error }` with exactly one of the two set (or neither, for a 204).
 */

export interface ApiViolation {
  field?: string | null;
  description?: string | null;
}

export interface ApiErrorDetail {
  subject?: string | null;
  description?: string | null;
  fields?: unknown[] | null;
  violations?: ApiViolation[] | null;
}

export interface ApiError {
  message: string;
  code?: number | null;
  details?: ApiErrorDetail[] | null;
}

export interface ApiResult<T> {
  data: T | null;
  error: ApiError | null;
}

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {},
): Promise<ApiResult<T>> {
  try {
    const headers: Record<string, string> = { Accept: "application/json" };
    // Only a request that carries a body says what it is: a bare GET with a
    // JSON content type is a non-simple request, and cross-origin that costs
    // a preflight for nothing.
    if (typeof options.body === "string") {
      headers["Content-Type"] = "application/json";
    }
    const res = await fetch(`${getApiBaseUrl()}${path}`, {
      ...options,
      headers: {
        ...headers,
        ...(options.headers as Record<string, string> | undefined),
      },
    });
    if (!res.ok) {
      let message = res.statusText || `Request failed (${res.status})`;
      let code: number | null = res.status;
      let details: ApiErrorDetail[] | null = null;
      try {
        const body = await res.json();
        if (typeof body?.message === "string") {
          message = body.message;
        } else if (typeof body?.detail === "string") {
          // FastAPI's own errors (`HTTPException`, an unknown route) answer
          // `{detail: "..."}` rather than the project envelope.
          message = body.detail;
        }
        if (typeof body?.code === "number") {
          code = body.code;
        }
        if (Array.isArray(body?.details)) {
          details = body.details as ApiErrorDetail[];
        }
      } catch {
        /* non-JSON error body */
      }
      return { data: null, error: { message, code, details } };
    }
    if (res.status === 204) {
      return { data: null, error: null };
    }
    // Read as text first: an empty body makes `res.json()` reject, which would
    // otherwise be reported as a network error on a call that succeeded.
    const text = await res.text();
    if (!text) {
      return { data: null, error: null };
    }
    const body = JSON.parse(text);
    // The project envelope is `{data: ...}`; a bare body is taken as it is.
    return { data: (body?.data ?? body) as T, error: null };
  } catch {
    return { data: null, error: { message: "Network error" } };
  }
}
