import { WeekIntensityMini } from "./WeekIntensityMini"
import type { WeekTotals } from "./weekTotals"

function formatDuration(totalMinutes: number): string {
  const totalSeconds = Math.round(totalMinutes * 60)
  const h = Math.floor(totalSeconds / 3600)
  const m = Math.floor((totalSeconds % 3600) / 60)
  return `${h}:${String(m).padStart(2, "0")}`
}

interface WeekSummaryProps {
  athleteId: number
  weekStart: string
  weekEnd: string
  totals: WeekTotals
}

// Sits below WeekCategorySelect in the calendar's leftmost column: a compact
// intensity bar plus planned-vs-actual duration/load and actual distance.
export function WeekSummary({ athleteId, weekStart, weekEnd, totals }: WeekSummaryProps) {
  const { plannedDurationMin, actualDurationMin, plannedTss, actualTss, actualDistanceKm } =
    totals

  return (
    <div className="w-full space-y-1 border-t border-dashed border-border pt-1 text-[10px] leading-tight text-muted-foreground">
      <WeekIntensityMini athleteId={athleteId} weekStart={weekStart} weekEnd={weekEnd} />
      <div className="flex items-center justify-between gap-1">
        <span>Dur</span>
        <span className="text-foreground">
          {formatDuration(plannedDurationMin)} / {formatDuration(actualDurationMin)}
        </span>
      </div>
      <div className="flex items-center justify-between gap-1">
        <span>Load</span>
        <span className="text-foreground">
          {Math.round(plannedTss)} / {Math.round(actualTss)}
        </span>
      </div>
      <div className="flex items-center justify-between gap-1">
        <span>Dist</span>
        <span className="text-foreground">{actualDistanceKm.toFixed(1)}km</span>
      </div>
    </div>
  )
}
