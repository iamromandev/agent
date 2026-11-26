import { RefreshCw } from "lucide-react";

import { Badge } from "@/components/core/badge";
import { Button } from "@/components/core/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/core/card";
import { Spinner } from "@/components/core/spinner";
import { useHealth } from "@/hook/use-health";
import type { HealthCheck } from "@/lib/api/health-client";

const FIELDS: Array<{ key: keyof HealthCheck; label: string }> = [
  { key: "status", label: "Status" },
  { key: "version", label: "Version" },
  { key: "environment", label: "Environment" },
  { key: "uptime", label: "Uptime" },
  { key: "host", label: "Host" },
  { key: "port", label: "Port" },
];

export function HealthStatus(): JSX.Element {
  const { data, error, isPending, isFetching, refetch } = useHealth();

  return (
    <Card className="w-full max-w-md">
      <CardHeader className="flex-row items-start justify-between space-y-0">
        <div className="space-y-1.5">
          <CardTitle>API health</CardTitle>
          <CardDescription>
            <code>GET /health/check</code>
          </CardDescription>
        </div>
        {isPending ? (
          <Spinner size="sm" />
        ) : error ? (
          <Badge variant="destructive">Unreachable</Badge>
        ) : (
          <Badge variant="success">Healthy</Badge>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        {error ? (
          <p className="text-sm text-destructive">{error.message}</p>
        ) : data ? (
          <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm">
            {FIELDS.filter(({ key }) => data[key] != null).map(
              ({ key, label }) => (
                <div key={key} className="contents">
                  <dt className="text-muted-foreground">{label}</dt>
                  <dd className="font-medium">{String(data[key])}</dd>
                </div>
              ),
            )}
          </dl>
        ) : null}
        <Button
          variant="outline"
          size="sm"
          loading={isFetching}
          onClick={() => void refetch()}
        >
          <RefreshCw />
          Check again
        </Button>
      </CardContent>
    </Card>
  );
}
