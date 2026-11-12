import { useQuery } from "@tanstack/react-query";

import { getHealth, type HealthCheck } from "@/lib/api/health-client";
import { queryKeys } from "@/lib/query/keys";

/** The API's health, re-checked every 30 seconds while the page is open. */
export function useHealth() {
  return useQuery<HealthCheck>({
    queryKey: queryKeys.health(),
    queryFn: async () => {
      const { data, error } = await getHealth();
      if (error) {
        throw new Error(error.message);
      }
      return data ?? {};
    },
    refetchInterval: 30_000,
    retry: false,
  });
}
