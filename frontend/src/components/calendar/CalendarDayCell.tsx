import { Brain, AlertTriangle } from "lucide-react"
import { cn } from "../../lib/utils"
import type { FeedDay } from "../../api/feed"

export const SPORT_COLORS: Record<string, string> = {
  cycling: "bg-blue-500",
  running: "bg-green-500",
  strength: "bg-orange-500",
  skiing_cross: "bg-cyan-500",
  skiing_alpine: "bg-indigo-500",
  day_off: "bg-gray-400",
  other: "bg-purple-500",
}

export const SPORT_COLORS_MUTED: Record<string, string> = {
  cycling: "bg-blue-300",
  running: "bg-green-300",
  strength: "bg-orange-300",
  skiing_cross: "bg-cyan-300",
  skiing_alpine: "bg-indigo-300",
  day_off: "bg-gray-300",
  other: "bg-purple-300",
}

interface MissingFlags {
  hasMissingWorkout: boolean
  hasMissingRecovery: boolean
  hasMissingAssessment: boolean
}

function getMissingFlags(day: FeedDay): MissingFlags {
  const hasPlannedTraining = day.plans.some((p) => p.sport !== "day_off")
  const hasMissingWorkout = hasPlannedTraining && day.workouts.length === 0
  const hasMissingRecovery = day.workouts.length > 0 && !day.restitution
  const hasMissingAssessment =
    day.workouts.length > 0 && day.activity_assessments.length === 0
  return { hasMissingWorkout, hasMissingRecovery, hasMissingAssessment }
}

function getTotalTss(day: FeedDay): number {
  return day.workouts.reduce((sum, w) => {
    const tss = w.metrics?.training_stress_score
    return sum + (tss ?? 0)
  }, 0)
}

interface CalendarDayCellProps {
  day: FeedDay | null
  date: string
  isToday: boolean
  isSelected: boolean
  maxTss?: number
  onClick: () => void
}

export function CalendarDayCell({
  day,
  date,
  isToday,
  isSelected,
  maxTss = 150,
  onClick,
}: CalendarDayCellProps) {
  const dayNum = parseInt(date.split("-")[2], 10)
  const flags = day ? getMissingFlags(day) : null
  const hasWarning =
    flags && (flags.hasMissingWorkout || flags.hasMissingRecovery || flags.hasMissingAssessment)
  const tss = day ? getTotalTss(day) : 0
  const tssBarWidth = tss > 0 ? Math.min((tss / maxTss) * 100, 100) : 0

  return (
    <button
      onClick={onClick}
      className={cn(
        "relative flex flex-col w-full rounded-md border text-left transition-colors",
        "hover:bg-accent hover:border-accent-foreground/20",
        "min-h-[80px] p-1.5",
        isSelected && "border-primary ring-1 ring-primary",
        isToday && !isSelected && "border-primary/50 bg-primary/5",
        !isToday && !isSelected && "border-border",
        !day && "opacity-40"
      )}
    >
      {/* Day number */}
      <span
        className={cn(
          "text-xs font-medium leading-none",
          isToday
            ? "flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground"
            : "text-foreground"
        )}
      >
        {dayNum}
      </span>

      {day && (
        <>
          {/* Activity labels */}
          <div className="mt-1 space-y-0.5">
            {day.plans.map((plan, i) => (
              <div key={`plan-label-${i}`} className="flex items-center gap-1">
                <span
                  className={cn(
                    "h-1.5 w-1.5 shrink-0 rounded-full opacity-60",
                    SPORT_COLORS_MUTED[plan.sport] ?? "bg-gray-300"
                  )}
                />
                <span className="truncate text-xs text-muted-foreground">
                  {plan.name}
                  {plan.estimated_duration_min ? ` · ${plan.estimated_duration_min}m` : ""}
                </span>
              </div>
              ))}
              {day.workouts.map((w, i) => (
                <div key={`workout-label-${i}`} className="flex items-center gap-1">
                  <span
                    className={cn(
                      "h-1.5 w-1.5 shrink-0 rounded-full",
                      SPORT_COLORS[w.session.category] ?? "bg-gray-500"
                    )}
                  />
                  <span className="truncate text-xs">
                    {w.session.name ?? w.session.sport}
                    {w.metrics?.training_stress_score
                      ? ` · ${Math.round(w.metrics.training_stress_score)} TSS`
                      : ""}
                  </span>
                </div>
              ))}
            </div>

          {/* TSS load bar */}
          {tssBarWidth > 0 && (
            <div className="absolute bottom-0 left-0 right-0 h-1 rounded-b-md overflow-hidden bg-muted">
              <div
                className="h-full bg-primary/40 transition-all"
                style={{ width: `${tssBarWidth}%` }}
              />
            </div>
          )}

          {/* Icons row */}
          <div className="mt-auto flex items-center gap-1 pt-1">
            {day.has_analysis && (
              <Brain className="h-3 w-3 text-muted-foreground" aria-label="AI analysis available" />
            )}
            {hasWarning && (
              <AlertTriangle
                className="h-3 w-3 text-amber-500"
                aria-label="Missing data"
              />
            )}
          </div>
        </>
      )}
    </button>
  )
}
