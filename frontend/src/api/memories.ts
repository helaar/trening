import { apiFetch } from "./client"

export interface AthleteMemory {
  scope: "recent" | "long_term"
  category: "recovery" | "habit" | "performance" | "risk" | "goal"
  content: string
  confidence: number
}

export function fetchAthleteMemories(athleteId: number): Promise<AthleteMemory[]> {
  return apiFetch<AthleteMemory[]>(`/api/v1/athlete/${athleteId}/memories`)
}
