import { apiFetch } from "./client"
import type { FeedDay } from "./feed"
import type { StoredAnalysis } from "./tasks"

export interface RosterAthlete {
  athlete_id: number
  name: string | null
  profile_picture: string | null
}

export interface CoachProfile {
  coach_id: number
  display_name: string | null
  roster: RosterAthlete[]
}

export interface RiskCounts {
  low: number
  moderate: number
  high: number
}

export interface AthleteStatus {
  athlete_id: number
  name: string | null
  profile_picture: string | null
  last_activity_date: string | null
  latest_readiness: number | null
  latest_hrv: number | null
  performance_risk: RiskCounts
  restitution_risk: RiskCounts
  overall_recovery_quality: string | null
}

export function fetchCoachProfile(): Promise<CoachProfile> {
  return apiFetch<CoachProfile>("/api/v1/coach/me")
}

export function fetchCoachRoster(windowDays = 14): Promise<AthleteStatus[]> {
  return apiFetch<AthleteStatus[]>(`/api/v1/coach/roster?window_days=${windowDays}`)
}

export function fetchCoachAthleteFeed(
  athleteId: number,
  start: string,
  end: string
): Promise<FeedDay[]> {
  return apiFetch<FeedDay[]>(
    `/api/v1/coach/athletes/${athleteId}/feed?start=${start}&end=${end}`
  )
}

export function fetchCoachAthleteDailyAnalysis(
  athleteId: number,
  date: string
): Promise<StoredAnalysis | null> {
  return apiFetch<StoredAnalysis | null>(
    `/api/v1/coach/athletes/${athleteId}/daily-analysis?date=${date}`
  )
}
