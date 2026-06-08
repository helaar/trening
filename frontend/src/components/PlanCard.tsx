import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"
import type { PlannedActivity, Sport } from "../api/plans"

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
}

function SeasonGoalBadge() {
  return (
    <span className="text-xs px-2 py-0.5 rounded-full border font-medium bg-purple-100 text-purple-700 border-purple-300">
      🎯 Season goal
    </span>
  )
}

function RaceBadge() {
  return (
    <span className="text-xs px-2 py-0.5 rounded-full border font-medium bg-amber-100 text-amber-700 border-amber-300">
      🏆 Race
    </span>
  )
}

export function PlanCard({ plan }: Props) {
  const isSeasonGoal = plan.labels.includes("seasongoal")
  const isRace = plan.labels.includes("race")
  const visibleLabels = plan.labels.filter((l) => l !== "seasongoal" && l !== "race")

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="text-base">{plan.name}</CardTitle>
          <span className="shrink-0 rounded-full bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground">
            {sportLabel(plan.sport)}
          </span>
        </div>

        {(isSeasonGoal || isRace) && (
          <div className="flex flex-wrap gap-1.5">
            {isSeasonGoal && <SeasonGoalBadge />}
            {isRace && <RaceBadge />}
          </div>
        )}

        <div className="flex flex-wrap gap-2 text-sm text-muted-foreground">
          {plan.estimated_duration_min && <span>{plan.estimated_duration_min} min</span>}
          {plan.estimated_tss && <span>est. TSS {plan.estimated_tss}</span>}
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
