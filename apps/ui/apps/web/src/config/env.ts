import { z } from "zod";

import { apiBaseUrlFrom, clientEnvSchema } from "@/config/env-schema";

function validateEnv<T extends z.ZodTypeAny>(
  schema: T,
  values: Record<string, string | undefined>,
  context: string,
): z.infer<T> {
  const result = schema.safeParse(values);
  if (!result.success) {
    console.error(`❌ Invalid ${context} environment variables:`);
    result.error.issues.forEach((issue) => {
      console.error(`  ${issue.path.join(".")}: ${issue.message}`);
    });
    throw new Error(`Invalid ${context} environment variables`);
  }
  return result.data;
}

export const clientEnv = validateEnv(
  clientEnvSchema,
  {
    VITE_API_BASE_URL: import.meta.env.VITE_API_BASE_URL,
  },
  "client",
);

export const env = {
  apiBaseUrl: apiBaseUrlFrom(clientEnv.VITE_API_BASE_URL),
} as const;

/** Where every call in `src/lib/api/*` is sent. */
export function getApiBaseUrl(): string {
  return env.apiBaseUrl;
}
