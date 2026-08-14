import { CheckCircle2, ExternalLink } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"
import type { PlannedActivity, Sport } from "../api/plans"
import type { WorkoutAnalysis } from "../api/workouts"
import { mondayOf } from "../lib/utils"

function sportLabel(sport: Sport): string {
  const labels: Record<Sport, string> = {
    cycling: "Cycling",
    running: "Running",
    strength: "Strength",
    skiing_cross: "XC Skiing",
    skiing_alpine: "Alpine Skiing",
    day_off: "Day off",
    other: "Other",
  }
  return labels[sport]
}

interface Props {
  plan: PlannedActivity
  matchedWorkout?: WorkoutAnalysis
}

function CompletedBadge() {
  return (
    <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border font-medium bg-emerald-100 text-emerald-700 border-emerald-300">
      <CheckCircle2 className="h-3 w-3" /> Completed
    </span>
  )
}

function SeasonGoalBadge() {
  return (
    <span className="text-xs px-2 py-0.5 rounded-full border font-medium bg-purple-100 text-purple-700 border-purple-300">
      🎯 Season goal
    </span>
  )
}

function RaceBadge({ priority }: { priority: "A" | "B" | "C" | null }) {
  return (
    <span className="text-xs px-2 py-0.5 rounded-full border font-medium bg-amber-100 text-amber-700 border-amber-300">
      🏆 Race{priority ? ` (${priority})` : ""}
    </span>
  )
}

export function PlanCard({ plan, matchedWorkout }: Props) {
  const isSeasonGoal = plan.labels.includes("seasongoal")
  const isRace = plan.labels.includes("race")
  const visibleLabels = plan.labels.filter((l) => l !== "seasongoal" && l !== "race")
  const actualDurationMin = matchedWorkout ? Math.round(matchedWorkout.session.duration_sec / 60) : null
  const actualTss = matchedWorkout?.metrics.training_stress_score ?? null

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="text-base">{plan.name}</CardTitle>
          <div className="flex shrink-0 items-center gap-1.5">
            <span className="rounded-full bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground">
              {sportLabel(plan.sport)}
            </span>
            <a
              href={`https://intervals.icu/?w=${mondayOf(plan.date)}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground"
              aria-label="Open week in Intervals.icu"
              title="Open week in Intervals.icu"
            >
              <ExternalLink className="h-3.5 w-3.5" />
            </a>
          </div>
        </div>

        {(isSeasonGoal || isRace || plan.matched_activity_id) && (
          <div className="flex flex-wrap gap-1.5">
            {plan.matched_activity_id && <CompletedBadge />}
            {isSeasonGoal && <SeasonGoalBadge />}
            {isRace && <RaceBadge priority={plan.race_priority} />}
          </div>
        )}

        <div className="flex flex-wrap gap-2 text-sm text-muted-foreground">
          {plan.estimated_duration_min && (
            <span>
              {plan.estimated_duration_min} min
              {actualDurationMin !== null && ` (actual: ${actualDurationMin} min)`}
            </span>
          )}
          {plan.estimated_tss && (
            <span>
              est. TSS {plan.estimated_tss}
              {actualTss !== null && ` (actual: ${Math.round(actualTss)})`}
            </span>
          )}
        </div>

        {visibleLabels.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pt-1">
            {visibleLabels.map((label) => (
              <span
                key={label}
                className="rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground"
              >
                {label}
              </span>
            ))}
          </div>
        )}
      </CardHeader>

      {(plan.description || plan.purpose) && (
        <CardContent className="space-y-2 border-t pt-3 text-sm">
          {plan.description && <p className="text-foreground">{plan.description}</p>}
          {plan.purpose && (
            <p className="italic text-muted-foreground">{plan.purpose}</p>
          )}
        </CardContent>
      )}
    </Card>
  )
}
