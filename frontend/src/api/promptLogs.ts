import { apiFetch } from "./client"

export interface AgentUsage {
  agent_role: string | null
  model: string | null
  prompt_tokens: number
  cached_prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  successful_requests: number
  cost_usd: number | null
}

export interface RunUsage {
  prompt_tokens: number
  cached_prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  successful_requests: number
  cost_usd: number | null
  per_agent: AgentUsage[]
}

export interface PromptLogRunSummary {
  run_id: string
  athlete_id: number
  crew_name: string
  agent_roles: (string | null)[]
  call_count: number
  started_at: string
  usage?: RunUsage | null
}

export interface PromptLogMessage {
  role: string
  content: string
  [key: string]: unknown
}

export interface PromptLogEntry {
  run_id: string
  athlete_id: number
  crew_name: string
  kind: "llm_call" | "tool_call"
  agent_role: string | null
  task_name: string | null
  model: string | null
  call_type: string | null
  messages: PromptLogMessage[]
  response: string | null
  tool_name: string | null
  tool_args: Record<string, unknown> | string | null
  tool_output: string | null
  tool_error: string | null
  created_at: string
}

export async function fetchPromptLogRuns(athleteId?: number): Promise<PromptLogRunSummary[]> {
  const params = new URLSearchParams()
  if (athleteId !== undefined) params.set("athlete_id", String(athleteId))
  const qs = params.toString()
  return apiFetch<PromptLogRunSummary[]>(`/api/v1/admin/prompt-logs${qs ? `?${qs}` : ""}`)
}

export async function fetchPromptLogRun(runId: string): Promise<PromptLogEntry[]> {
  return apiFetch<PromptLogEntry[]>(`/api/v1/admin/prompt-logs/${runId}`)
}
