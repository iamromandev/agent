import { Helmet } from "react-helmet-async";

import { HealthStatus } from "@/components/features/health/health-status";
import { siteConfig } from "@/config/site-config";

export function HomePage(): JSX.Element {
  return (
    <>
      <Helmet>
        <title>Home</title>
        <meta name="description" content={siteConfig.description} />
      </Helmet>
      <section className="flex flex-1 flex-col items-center justify-center gap-8 px-6 py-16 text-center">
        <div className="space-y-3">
          <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl">
            {siteConfig.name}
          </h1>
          <p className="text-muted-foreground">{siteConfig.description}</p>
        </div>
        <HealthStatus />
      </section>
    </>
  );
}
