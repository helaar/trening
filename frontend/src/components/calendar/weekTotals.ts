import type { FeedDay } from "../../api/feed"

export interface WeekTotals {
  plannedDurationMin: number
  actualDurationMin: number
  plannedTss: number
  actualTss: number
  actualDistanceKm: number
}

// Sums over all workouts regardless of `manual`, consistent with the per-day
// TSS total CalendarDayCell already computes.
export function computeWeekTotals(days: Array<FeedDay | null | undefined>): WeekTotals {
  const totals: WeekTotals = {
    plannedDurationMin: 0,
    actualDurationMin: 0,
    plannedTss: 0,
    actualTss: 0,
    actualDistanceKm: 0,
  }

  for (const day of days) {
    if (!day) continue
    for (const plan of day.plans) {
      totals.plannedDurationMin += plan.estimated_duration_min ?? 0
      totals.plannedTss += plan.estimated_tss ?? 0
    }
    for (const workout of day.workouts) {
      totals.actualDurationMin += workout.session.duration_sec / 60
      totals.actualTss += workout.metrics?.training_stress_score ?? 0
      totals.actualDistanceKm += workout.session.distance_km ?? 0
    }
  }

  return totals
}
