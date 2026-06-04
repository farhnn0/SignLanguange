/**
 * Komponen UI primitif yang dipakai ulang di seluruh halaman:
 * Card, Badge, dan ConfidenceBar.
 */

import { cn } from "../lib/utils";

export function Card({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-neutral-200 bg-white p-5 shadow-sm",
        className
      )}
    >
      {children}
    </div>
  );
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: "neutral" | "success" | "warning" | "danger";
}) {
  const toneClass = {
    neutral: "border-neutral-200 bg-neutral-50 text-neutral-700",
    success: "border-green-200 bg-green-50 text-green-700",
    warning: "border-yellow-200 bg-yellow-50 text-yellow-800",
    danger: "border-red-200 bg-red-50 text-red-700",
  }[tone];

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium",
        toneClass
      )}
    >
      {children}
    </span>
  );
}

export function ConfidenceBar({ value }: { value: number }) {
  const safeValue = Math.max(0, Math.min(100, value));

  const barColor =
    safeValue >= 85
      ? "bg-green-600"
      : safeValue >= 60
      ? "bg-yellow-600"
      : "bg-red-600";

  const label =
    safeValue >= 85
      ? "High confidence"
      : safeValue >= 60
      ? "Medium confidence"
      : "Low confidence";

  return (
    <div className="space-y-3">
      <div className="flex items-end justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-neutral-500">Confidence</p>
          <p className="mt-1 text-4xl font-bold tracking-tight text-neutral-950">
            {safeValue}%
          </p>
        </div>

        <Badge
          tone={
            safeValue >= 85 ? "success" : safeValue >= 60 ? "warning" : "danger"
          }
        >
          {label}
        </Badge>
      </div>

      <div className="h-2.5 overflow-hidden rounded-full bg-neutral-100">
        <div
          className={cn("h-full rounded-full", barColor)}
          style={{ width: `${safeValue}%` }}
        />
      </div>
    </div>
  );
}
