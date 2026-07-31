import { apiFetch } from "./client"

export interface FitnessPoint {
  date: string
  ctl: number | null
  atl: number | null
  tsb: number | null
  ramp_rate: number | null
}

export function fetchFitnessSeries(athleteId: number, days = 90): Promise<FitnessPoint[]> {
  return apiFetch<FitnessPoint[]>(`/api/v1/athlete/${athleteId}/fitness?days=${days}`)
}
