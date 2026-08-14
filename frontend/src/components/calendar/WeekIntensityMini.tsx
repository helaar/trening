import { useQuery } from "@tanstack/react-query"
import { fetchWeeklyIntensity } from "../../api/tasks"

const BANDS = [
  { key: "low_pct", label: "Easy", color: "bg-emerald-500" },
  { key: "moderate_pct", label: "Moderate", color: "bg-amber-400" },
  { key: "high_pct", label: "Hard", color: "bg-violet-500" },
] as const

interface WeekIntensityMiniProps {
  athleteId: number
  weekStart: string
  weekEnd: string
}

// Compact easy/moderate/hard distribution bar for a Mon-Sun calendar week, reusing
// the same trailing-7-day endpoint WeeklyIntensityBar uses in the day-detail panel
// (queried with the week's Sunday so the trailing window aligns to Mon-Sun).
export function WeekIntensityMini({ athleteId, weekStart, weekEnd }: WeekIntensityMiniProps) {
  const { data } = useQuery({
    queryKey: ["weekly-intensity", athleteId, weekStart],
    queryFn: () => fetchWeeklyIntensity(athleteId, weekEnd),
  })

  if (!data || data.low_pct == null) return null

  const segments = BANDS.map((b) => ({
    ...b,
    pct: (data[b.key] as number | null) ?? 0,
  }))

  return (
    <div
      className="flex h-1.5 w-full overflow-hidden rounded bg-gray-100"
      title={segments.map((s) => `${s.label} ${s.pct.toFixed(0)}%`).join(" · ")}
    >
      {segments.map((s) => (
        <div key={s.key} className={s.color} style={{ width: `${s.pct}%` }} />
      ))}
    </div>
  )
}
