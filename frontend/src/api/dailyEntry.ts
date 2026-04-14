import { apiFetch } from "./client"

export interface Restitution {
  sleep_hours?: number
  sleep_quality?: number
  hrv?: number
  resting_hr?: number
  readiness?: number
  comment?: string
}

export interface ActivityAssessment {
  activity_id: number
  activity_name: string
  rpe: number
  notes?: string
}

export interface DailyEntryRequest {
  date: string
  restitution?: Restitution
  activity_assessments: ActivityAssessment[]
}

export interface DailyEntry extends DailyEntryRequest {
  athlete_id: number
  created_at: string
  updated_at: string
}

export async function fetchDailyEntry(
  athleteId: number,
  date: string
): Promise<DailyEntry | null> {
  return apiFetch<DailyEntry | null>(`/api/v1/athlete/${athleteId}/daily-entry?date=${date}`)
}

export function saveDailyEntry(
  athleteId: number,
  entry: DailyEntryRequest
): Promise<DailyEntry> {
  return apiFetch<DailyEntry>(`/api/v1/athlete/${athleteId}/daily-entry`, {
    method: "POST",
    body: JSON.stringify(entry),
  })
}
