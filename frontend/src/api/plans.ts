import { apiFetch } from "./client"

export type Sport = "cycling" | "running" | "strength" | "skiing_cross" | "skiing_alpine" | "day_off" | "other"

export interface PlannedActivity {
  id: string
  athlete_id: number
  date: string
  sport: Sport
  name: string
  description?: string
  purpose?: string
  labels: string[]
  estimated_duration_min?: number
  estimated_tss?: number
  created_at: string
  updated_at: string
}

export interface PlannedActivityRequest {
  date: string
  sport: Sport
  name: string
  description?: string
  purpose?: string
  labels?: string[]
  estimated_duration_min?: number
  estimated_tss?: number
}

export function fetchPlansForDate(athleteId: number, date: string): Promise<PlannedActivity[]> {
  return apiFetch<PlannedActivity[]>(`/api/v1/athlete/${athleteId}/plans?date=${date}`)
}

export function fetchPlansForRange(
  athleteId: number,
  start: string,
  end: string
): Promise<PlannedActivity[]> {
  return apiFetch<PlannedActivity[]>(
    `/api/v1/athlete/${athleteId}/plans?start=${start}&end=${end}`
  )
}

export function createPlan(
  athleteId: number,
  plan: PlannedActivityRequest
): Promise<PlannedActivity> {
  return apiFetch<PlannedActivity>(`/api/v1/athlete/${athleteId}/plans`, {
    method: "POST",
    body: JSON.stringify(plan),
  })
}

export function updatePlan(
  athleteId: number,
  planId: string,
  plan: PlannedActivityRequest
): Promise<PlannedActivity> {
  return apiFetch<PlannedActivity>(`/api/v1/athlete/${athleteId}/plans/${planId}`, {
    method: "PUT",
    body: JSON.stringify(plan),
  })
}

export function deletePlan(athleteId: number, planId: string): Promise<void> {
  return apiFetch<void>(`/api/v1/athlete/${athleteId}/plans/${planId}`, {
    method: "DELETE",
  })
}
