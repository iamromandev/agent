/**
 * Every query key in the app comes from here, under one root, so a cache can
 * be dropped by prefix. A page that builds its own literal key opts out of
 * that, silently, so it should not.
 */
const ROOT = "agent" as const;

export const queryKeys = {
  all: [ROOT] as const,
  health: () => [ROOT, "health"] as const,
} as const;
