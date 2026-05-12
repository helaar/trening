import { apiFetch } from "./client"
import type { Restitution, ActivityAssessment } from "./dailyEntry"
import type { PlannedActivity } from "./plans"
import type { WorkoutAnalysis } from "./workouts"

export interface FeedDay {
  date: string
  workouts: WorkoutAnalysis[]
  plans: PlannedActivity[]
  restitution: Restitution | null
  activity_assessments: ActivityAssessment[]
}

export function fetchFeed(athleteId: number, start: string, end: string): Promise<FeedDay[]> {
  return apiFetch<FeedDay[]>(
    `/api/v1/athlete/${athleteId}/feed?start=${start}&end=${end}`
  )
}
