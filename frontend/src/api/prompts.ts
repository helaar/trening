import { apiFetch } from "./client"

export interface CrewDefinitionBase {
  name: string
  updated_at?: string
}

export interface AgentDoc extends CrewDefinitionBase {
  type: "agent"
  role: string
  goal: string
  backstory: string
  llm_model: string
}

export interface TaskDoc extends CrewDefinitionBase {
  type: "task"
  description: string
  expected_output: string
}

export interface PhilosophyDoc extends CrewDefinitionBase {
  type: "philosophy"
  display_name: string
  intensity_targets: string
  coach_guidance: string
  analyst_guidance: string
}

export type CrewDefinition = AgentDoc | TaskDoc | PhilosophyDoc

export async function fetchPrompts(): Promise<CrewDefinition[]> {
  return apiFetch<CrewDefinition[]>("/api/v1/admin/prompts")
}

export async function savePrompts(updates: CrewDefinition[]): Promise<CrewDefinition[]> {
  return apiFetch<CrewDefinition[]>("/api/v1/admin/prompts", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  })
}
