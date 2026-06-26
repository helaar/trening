import { useQuery } from "@tanstack/react-query"
import { Link } from "@tanstack/react-router"
import { Loader2 } from "lucide-react"
import { fetchCoachRoster } from "../api/coach"
import type { RiskCounts } from "../api/coach"
import { cn } from "../lib/utils"

const qualityStyle: Record<string, string> = {
  good: "text-green-700",
  adequate: "text-yellow-700",
  concerning: "text-red-700",
}

function RiskBadge({ label, counts }: { label: string; counts: RiskCounts }) {
  const total = counts.low + counts.moderate + counts.high
  if (total === 0) {
    return <span className="text-xs text-muted-foreground">{label}: none</span>
  }
  const severity = counts.high > 0 ? "high" : counts.moderate > 0 ? "moderate" : "low"
  const severityStyle: Record<string, string> = {
    low: "border-yellow-200 bg-yellow-50 text-yellow-800",
    moderate: "border-orange-200 bg-orange-50 text-orange-800",
    high: "border-red-200 bg-red-50 text-red-800",
  }
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs", severityStyle[severity])}>
      {label}: {counts.high > 0 && `${counts.high} high`}
      {counts.high > 0 && (counts.moderate > 0 || counts.low > 0) && ", "}
      {counts.moderate > 0 && `${counts.moderate} moderate`}
      {counts.moderate > 0 && counts.low > 0 && ", "}
      {counts.low > 0 && `${counts.low} low`}
    </span>
  )
}

export function CoachOverviewPage() {
  const { data: roster, isLoading, error } = useQuery({
    queryKey: ["coach-roster"],
    queryFn: () => fetchCoachRoster(),
  })

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center p-4">
        <p className="text-sm text-destructive">Could not load roster.</p>
      </div>
    )
  }

  return (
    <div className="h-full overflow-auto p-6">
      <h1 className="mb-4 text-xl font-semibold">Coach Overview</h1>
      <div className="overflow-hidden rounded-md border">
        <table className="w-full text-sm">
          <thead className="bg-muted/50 text-left text-xs uppercase text-muted-foreground">
            <tr>
              <th className="px-4 py-2">Athlete</th>
              <th className="px-4 py-2">Last activity</th>
              <th className="px-4 py-2">Readiness</th>
              <th className="px-4 py-2">HRV</th>
              <th className="px-4 py-2">Performance risk</th>
              <th className="px-4 py-2">Restitution risk</th>
              <th className="px-4 py-2">Recovery quality</th>
            </tr>
          </thead>
          <tbody>
            {roster?.map((athlete) => (
              <tr key={athlete.athlete_id} className="border-t hover:bg-accent/50">
                <td className="px-4 py-3">
                  <Link
                    to="/coach/athletes/$athleteId"
                    params={{ athleteId: String(athlete.athlete_id) }}
                    className="font-medium text-foreground hover:underline"
                  >
                    {athlete.name ?? athlete.athlete_id}
                  </Link>
                </td>
                <td className="px-4 py-3">{athlete.last_activity_date ?? "—"}</td>
                <td className="px-4 py-3">{athlete.latest_readiness ?? "—"}</td>
                <td className="px-4 py-3">{athlete.latest_hrv ?? "—"}</td>
                <td className="px-4 py-3">
                  <RiskBadge label="Performance" counts={athlete.performance_risk} />
                </td>
                <td className="px-4 py-3">
                  <RiskBadge label="Restitution" counts={athlete.restitution_risk} />
                </td>
                <td className="px-4 py-3">
                  {athlete.overall_recovery_quality ? (
                    <span className={qualityStyle[athlete.overall_recovery_quality]}>
                      {athlete.overall_recovery_quality}
                    </span>
                  ) : (
                    "—"
                  )}
                </td>
              </tr>
            ))}
            {roster?.length === 0 && (
              <tr>
                <td colSpan={7} className="px-4 py-6 text-center text-muted-foreground">
                  No athletes in your roster.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
