import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  // Every variable, not only `VITE_*`: `AGENT_API_ORIGIN` configures this dev
  // server alone and must never reach the bundle, which only `VITE_*` values do.
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.AGENT_API_ORIGIN?.trim() || "http://localhost:8000";
  // `/api/*`, not the API's own paths. The API serves `/health/*` and so on at
  // its root and the prefix is stripped on the way out, so nothing changes on
  // its side — but the SPA keeps every other path for its own routes, and no
  // page can ever collide with an endpoint. `getApiBaseUrl` in
  // `src/config/env.ts` is the other half: by default it points the app at
  // `/api` on this origin.
  const rewrite = (path: string) => path.replace(/^\/api/, "");

  return {
    plugins: [tailwindcss(), react()],
    resolve: {
      tsconfigPaths: true,
    },
    server: {
      port: 3000,
      host: true,
      proxy: {
        "/api/": {
          target,
          rewrite,
          changeOrigin: true,
        },
      },
    },
    build: {
      outDir: "dist",
      sourcemap: true,
      target: "baseline-widely-available",
    },
  };
});
