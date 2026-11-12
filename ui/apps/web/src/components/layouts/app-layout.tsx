import { Bot, Moon, Sun } from "lucide-react";
import { Helmet } from "react-helmet-async";
import { Link, Outlet } from "react-router-dom";

import { Button } from "@/components/core/button";
import { useTheme } from "@/components/layouts/theme-provider";
import { siteConfig } from "@/config/site-config";

export function AppLayout(): JSX.Element {
  const { resolvedTheme, setTheme } = useTheme();
  const next = resolvedTheme === "dark" ? "light" : "dark";

  return (
    <>
      <Helmet
        titleTemplate={`%s | ${siteConfig.name}`}
        defaultTitle={siteConfig.name}
      />
      <div className="flex min-h-screen flex-col">
        <header className="border-b border-border/50 px-6 py-4">
          <div className="mx-auto flex max-w-6xl items-center justify-between">
            <Link to="/" className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
                <Bot className="h-5 w-5 text-primary-foreground" />
              </div>
              <span className="text-lg font-bold tracking-tight">
                {siteConfig.name}
              </span>
            </Link>
            <Button
              variant="ghost"
              size="icon"
              aria-label={`Switch to ${next} theme`}
              onClick={() => setTheme(next)}
            >
              {resolvedTheme === "dark" ? <Sun /> : <Moon />}
            </Button>
          </div>
        </header>
        <main className="flex flex-1 flex-col">
          <Outlet />
        </main>
      </div>
    </>
  );
}
