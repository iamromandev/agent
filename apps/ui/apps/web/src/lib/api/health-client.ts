import { apiFetch, type ApiResult } from "@/lib/api/api-client";

/**
 * What `GET /health/check` reports. Every field is optional: the page shows
 * whatever the API sends, so the endpoint can grow without the UI breaking.
 */
export interface HealthCheck {
  status?: string;
  version?: string;
  environment?: string | null;
  host?: string | null;
  port?: number | null;
  uptime?: string | null;
}

export async function getHealth(): Promise<ApiResult<HealthCheck>> {
  return apiFetch<HealthCheck>("/health/check");
}
