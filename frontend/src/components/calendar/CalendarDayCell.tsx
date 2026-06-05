import { Brain, AlertTriangle, Plus } from "lucide-react"
import { cn } from "../../lib/utils"
import type { FeedDay } from "../../api/feed"
import { SPORT_COLORS, SPORT_COLORS_MUTED } from "./sportColors"

interface MissingFlags {
  hasMissingWorkout: boolean
  hasMissingRecovery: boolean
  hasMissingAssessment: boolean
}

function getMissingFlags(day: FeedDay, date: string, today: string): MissingFlags {
  // Only flag past days — future dates simply haven't happened yet
  if (date >= today) return { hasMissingWorkout: false, hasMissingRecovery: false, hasMissingAssessment: false }

  const hasPlannedTraining = day.plans.some((p) => p.sport !== "day_off")
  // Manual notes (e.g. "Day off") are not real workouts for warning purposes
  const realWorkouts = day.workouts.filter((w) => !w.session.manual)
  const hasMissingWorkout = hasPlannedTraining && realWorkouts.length === 0
  const hasMissingRecovery = realWorkouts.length > 0 && !day.restitution
  const hasMissingAssessment = realWorkouts.length > 0 && day.activity_assessments.length === 0
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
  onClick: () => void
  onAddPlan?: (e: React.MouseEvent) => void
}

const TSS_SCALE = 500

export function CalendarDayCell({
  day,
  date,
  isToday,
  isSelected,
  onClick,
  onAddPlan,
}: CalendarDayCellProps) {
  const today = new Date().toISOString().split("T")[0]
  const dayNum = parseInt(date.split("-")[2], 10)
  const flags = day ? getMissingFlags(day, date, today) : null
  const hasWarning =
    flags && (flags.hasMissingWorkout || flags.hasMissingRecovery || flags.hasMissingAssessment)
  const tss = day ? getTotalTss(day) : 0
  const tssBarWidth = tss > 0 ? Math.min((tss / TSS_SCALE) * 100, 100) : 0

  return (
    <button
      onClick={onClick}
      title={
        hasWarning && flags
          ? "⚠ Warning:\n" +
            [
              flags.hasMissingWorkout && "Planned workout not recorded",
              flags.hasMissingRecovery && "No health/recovery data",
              flags.hasMissingAssessment && "No self-assessment",
            ]
              .filter(Boolean)
              .join("\n")
          : undefined
      }
      className={cn(
        "group relative flex flex-col w-full rounded-md border text-left transition-colors",
        "hover:bg-accent hover:border-accent-foreground/20",
        "min-h-[80px] p-1.5",
        isSelected && "border-primary ring-1 ring-primary",
        isToday && !isSelected && "border-primary/50 bg-primary/5",
        !isToday && !isSelected && "border-border",
        !day && "opacity-40"
      )}
    >
      {/* Day number + add plan button */}
      <div className="flex items-center justify-between">
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
        {onAddPlan && (
          <button
            onClick={(e) => { e.stopPropagation(); onAddPlan(e) }}
            className="opacity-0 group-hover:opacity-100 transition-opacity rounded p-0.5 text-muted-foreground hover:text-foreground hover:bg-muted"
            aria-label="Add plan"
            title="Add plan"
          >
            <Plus className="h-3 w-3" />
          </button>
        )}
      </div>

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
              <AlertTriangle className="h-3 w-3 text-amber-500" aria-label="Missing data" />
            )}
          </div>
        </>
      )}
    </button>
  )
}
