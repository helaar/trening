import type { RiskCounts } from "../api/coach"

export type Severity = "none" | "low" | "moderate" | "high"

export function severityFor(counts: RiskCounts): Severity {
  if (counts.high > 0) return "high"
  if (counts.moderate > 0) return "moderate"
  if (counts.low > 0) return "low"
  return "none"
}

export function combinedSeverity(
  performance: RiskCounts,
  restitution: RiskCounts
): Severity {
  const order: Severity[] = ["none", "low", "moderate", "high"]
  const a = severityFor(performance)
  const b = severityFor(restitution)
  return order[Math.max(order.indexOf(a), order.indexOf(b))]
}

export const severityBadgeStyle: Record<Severity, string> = {
  none: "",
  low: "border-yellow-200 bg-yellow-50 text-yellow-800",
  moderate: "border-orange-200 bg-orange-50 text-orange-800",
  high: "border-red-200 bg-red-50 text-red-800",
}

export const severityRowStyle: Record<Severity, string> = {
  none: "",
  low: "bg-yellow-50",
  moderate: "bg-orange-50",
  high: "bg-red-50",
}

export const recoveryQualityTextStyle: Record<string, string> = {
  good: "text-green-700",
  adequate: "text-yellow-700",
  concerning: "text-red-700",
}
