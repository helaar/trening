import type { WeeklyAssessment, WeeklyPhilosophyStatus } from "../api/tasks"

const STATUS_STYLE: Record<WeeklyPhilosophyStatus, string> = {
  polarized: "border-emerald-200 bg-emerald-50 text-emerald-800",
  mild_drift: "border-amber-200 bg-amber-50 text-amber-800",
  gray_zone_week: "border-red-200 bg-red-50 text-red-800",
  insufficient_data: "border-gray-200 bg-gray-50 text-gray-600",
}

const STATUS_LABEL: Record<WeeklyPhilosophyStatus, string> = {
  polarized: "Polarized",
  mild_drift: "Mild drift",
  gray_zone_week: "Gray-zone week",
  insufficient_data: "Insufficient data",
}

const BANDS = [
  { key: "low", label: "Easy", color: "bg-emerald-500" },
  { key: "moderate", label: "Moderate", color: "bg-amber-400" },
  { key: "high", label: "Hard", color: "bg-violet-500" },
] as const

function StatusBadge({ status }: { status: WeeklyPhilosophyStatus }) {
  return (
    <span
      className={`shrink-0 rounded-full border px-2 py-0.5 text-xs font-medium ${STATUS_STYLE[status]}`}
    >
      {STATUS_LABEL[status]}
    </span>
  )
}

export function WeeklyIntensityBar({ a }: { a: WeeklyAssessment }) {
  const window = a.window.replace("..", " → ")

  if (a.status === "insufficient_data" || a.low_pct == null) {
    return (
      <div className="rounded-md border bg-gray-50 px-4 py-3">
        <div className="flex items-center gap-2">
          <h4 className="text-sm font-semibold text-gray-800">This week's intensity</h4>
          <StatusBadge status={a.status} />
          <span className="ml-auto text-xs text-gray-400">{window}</span>
        </div>
        <p className="mt-1 text-xs text-gray-500">
          {a.description ||
            "Not enough classified endurance training this week to judge polarization."}
        </p>
      </div>
    )
  }

  const segments = [
    { ...BANDS[0], pct: a.low_pct ?? 0 },
    { ...BANDS[1], pct: a.moderate_pct ?? 0 },
    { ...BANDS[2], pct: a.high_pct ?? 0 },
  ]
  const hours = (a.classified_minutes / 60).toFixed(1)

  return (
    <div className="space-y-2 rounded-md border px-4 py-3">
      <div className="flex items-center gap-2">
        <h4 className="text-sm font-semibold text-gray-800">This week's intensity</h4>
        <StatusBadge status={a.status} />
        <span className="ml-auto text-xs text-gray-400">{window}</span>
      </div>

      {/* Stacked distribution bar with a marker at the 80%-easy polarized target. */}
      <div className="relative">
        <div className="flex h-4 w-full overflow-hidden rounded bg-gray-100">
          {segments.map((s) => (
            <div
              key={s.key}
              className={s.color}
              style={{ width: `${s.pct}%` }}
              title={`${s.label} ${s.pct.toFixed(0)}%`}
            />
          ))}
        </div>
        <div
          className="pointer-events-none absolute inset-y-0 left-[80%] w-px bg-gray-700/70"
          title="80% easy target"
        />
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-600">
        {segments.map((s) => (
          <span key={s.key} className="inline-flex items-center gap-1">
            <span className={`inline-block h-2.5 w-2.5 rounded-sm ${s.color}`} />
            {s.label} {s.pct.toFixed(0)}%
          </span>
        ))}
        <span className="ml-auto text-gray-400">
          {hours} h
          {a.weekly_tss != null ? ` · ${Math.round(a.weekly_tss)} TSS` : ""} · {a.session_count}{" "}
          sessions
        </span>
      </div>

      <p className="text-xs leading-relaxed text-gray-600">{a.description}</p>
      <p className="text-[10px] text-gray-400">Line marks the 80% easy target.</p>
    </div>
  )
}
