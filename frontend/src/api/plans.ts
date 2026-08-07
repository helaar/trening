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
  external_reference?: string | null
  race_priority: "A" | "B" | "C" | null
  created_at: string
  updated_at: string
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
