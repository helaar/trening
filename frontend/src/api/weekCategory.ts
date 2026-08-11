import { apiFetch } from "./client"

export type WeekCategory =
  | "build"
  | "maintain"
  | "taper"
  | "race"
  | "restitution"
  | "recovering"

export interface WeekCategoryEntry {
  athlete_id: number
  week_start: string
  category: WeekCategory
  created_at: string
  updated_at: string
}

export function fetchWeekCategories(
  athleteId: number,
  start: string,
  end: string
): Promise<WeekCategoryEntry[]> {
  return apiFetch<WeekCategoryEntry[]>(
    `/api/v1/athlete/${athleteId}/week-category?start=${start}&end=${end}`
  )
}

export function setWeekCategory(
  athleteId: number,
  weekStart: string,
  category: WeekCategory
): Promise<WeekCategoryEntry> {
  return apiFetch<WeekCategoryEntry>(`/api/v1/athlete/${athleteId}/week-category`, {
    method: "POST",
    body: JSON.stringify({ week_start: weekStart, category }),
  })
}
