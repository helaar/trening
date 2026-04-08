import { apiFetch } from "./client"

export interface Athlete {
  athlete_id: number
  username: string | null
  firstname: string | null
  lastname: string | null
  profile_picture: string | null
}

export function fetchCurrentAthlete(): Promise<Athlete> {
  return apiFetch<Athlete>("/auth/me")
}
