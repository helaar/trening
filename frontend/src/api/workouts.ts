import { apiFetch } from "./client"

export interface WorkoutMetrics {
  power: { mean: number | null; max: number | null } | null
  heart_rate: { mean: number | null; max: number | null } | null
  normalized_power: number | null
  training_stress_score: number | null
  intensity_factor: number | null
}

export interface SessionInfo {
  name: string | null
  sport: string
  category: "cycling" | "running" | "skiing" | "strength" | "other"
  start_time: string | null
  duration_sec: number
  distance_km: number
  commute: "yes, marked by athlete" | "yes, detected" | "no"
  tags: string[]
  manual?: boolean
}

export interface WorkoutAnalysis {
  activity_id: number | null
  session: SessionInfo
  metrics: WorkoutMetrics
  has_power_data: boolean
  has_heart_rate_data: boolean
}

export function deleteWorkout(athleteId: number, activityId: number): Promise<void> {
  return apiFetch<void>(`/api/v1/athlete/${athleteId}/activities/${activityId}`, {
    method: "DELETE",
  })
}

export function updateNote(athleteId: number, activityId: number, text: string): Promise<WorkoutAnalysis> {
  return apiFetch<WorkoutAnalysis>(`/api/v1/athlete/${athleteId}/activities/${activityId}/note`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  })
}

export function createNote(athleteId: number, date: string, text: string): Promise<WorkoutAnalysis> {
  return apiFetch<WorkoutAnalysis>(`/api/v1/athlete/${athleteId}/workouts/note`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ date, text }),
  })
}

export function fetchDetailedWorkouts(
  athleteId: number,
  date: string,
  refresh = false
): Promise<WorkoutAnalysis[]> {
  const params = refresh ? `&refresh=true` : ""
  return apiFetch<WorkoutAnalysis[]>(
    `/api/v1/athlete/${athleteId}/workouts/detailed?date=${date}${params}`
  )
}
