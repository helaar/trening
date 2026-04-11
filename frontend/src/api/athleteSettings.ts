import { apiFetch } from "./client"

export interface ZoneDefinition {
  name: string
  min: number
  max: number | null
}

export interface SportSettings {
  ftp: number
  measured_activity: string | null
  measured_date: string | null
  power_zones: ZoneDefinition[]
}

export interface HeartRateSettings {
  max: number
  lt: number
  measured_activity: string | null
  measured_date: string | null
  hr_zones: ZoneDefinition[]
}

export interface AthleteSettings {
  cycling: SportSettings | null
  running: SportSettings | null
  heart_rate: HeartRateSettings | null
  autolap: string | null
}

export type AthleteSettingsPatch = Partial<AthleteSettings>

export function fetchAthleteSettings(athleteId: number): Promise<AthleteSettings> {
  return apiFetch<AthleteSettings>(`/api/v1/athlete/${athleteId}/settings`)
}

export function patchAthleteSettings(
  athleteId: number,
  patch: AthleteSettingsPatch,
): Promise<AthleteSettings> {
  return apiFetch<AthleteSettings>(`/api/v1/athlete/${athleteId}/settings`, {
    method: "PATCH",
    body: JSON.stringify(patch),
  })
}
