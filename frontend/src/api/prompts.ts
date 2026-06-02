import { apiFetch } from "./client"

export interface PromptConfig {
  key: string
  value: string
  updated_at: string
}

export interface PromptConfigUpdate {
  key: string
  value: string
}

export async function fetchPrompts(): Promise<PromptConfig[]> {
  return apiFetch<PromptConfig[]>("/api/v1/admin/prompts")
}

export async function savePrompts(updates: PromptConfigUpdate[]): Promise<PromptConfig[]> {
  return apiFetch<PromptConfig[]>("/api/v1/admin/prompts", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  })
}
