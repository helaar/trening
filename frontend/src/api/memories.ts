import { apiFetch } from "./client"

export interface AthleteMemory {
  memory_id: string
  scope: "recent" | "long_term"
  category: "recovery" | "habit" | "performance" | "risk" | "goal"
  content: string
  confidence: number
}

export function fetchAthleteMemories(athleteId: string): Promise<AthleteMemory[]> {
  return apiFetch<AthleteMemory[]>(`/api/v1/athlete/${athleteId}/memories`)
}

export function fetchSuppressedMemories(athleteId: string): Promise<AthleteMemory[]> {
  return apiFetch<AthleteMemory[]>(`/api/v1/athlete/${athleteId}/memories/suppressed`)
}

export function suppressMemory(athleteId: string, memoryId: string): Promise<void> {
  return apiFetch<void>(`/api/v1/athlete/${athleteId}/memories/${memoryId}`, {
    method: "DELETE",
  })
}

export function restoreMemory(athleteId: string, memoryId: string): Promise<void> {
  return apiFetch<void>(`/api/v1/athlete/${athleteId}/memories/${memoryId}/restore`, {
    method: "POST",
  })
}
