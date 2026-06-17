/**
 * Tiga kartu metrik: Response Time, Active Model, API Status.
 */

import { Activity, Brain, Server } from "lucide-react";
import { Card } from "./ui";
import type { ApiStatus } from "../lib/types";

type MetricsRowProps = {
  responseTime: string;
  activeMode: string;
  apiStatus: ApiStatus;
};

export function MetricsRow({
  responseTime,
  activeMode,
  apiStatus,
}: MetricsRowProps) {
  return (
    <div className="grid gap-6 md:grid-cols-3">
      {/* Kartu Metrik 1: Latensi Response Time */}
      <Card>
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-neutral-100">
            <Activity size={18} />
          </div>
          <div>
            <p className="text-sm text-neutral-500">Response Time</p>
            <p className="text-xl font-semibold">{responseTime}</p>
          </div>
        </div>
      </Card>

      {/* Kartu Metrik 2: Model yang Sedang Aktif */}
      <Card>
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-neutral-100">
            <Brain size={18} />
          </div>
          <div>
            <p className="text-sm text-neutral-500">Active Model</p>
            <p className="text-xl font-semibold">
              {activeMode === "kata" ? "GRU Kata" : "Static RF"}
            </p>
          </div>
        </div>
      </Card>

      {/* Kartu Metrik 3: Status Konektivitas Backend */}
      <Card>
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-neutral-100">
            <Server size={18} />
          </div>
          <div>
            <p className="text-sm text-neutral-500">API Status</p>
            <p className="text-xl font-semibold">{apiStatus}</p>
          </div>
        </div>
      </Card>
    </div>
  );
}
