import { apiFetch } from "./client"

export type TaskStatus = "pending" | "running" | "completed" | "failed"

export interface TaskResponse {
  task_id: string
  status: TaskStatus
  progress: number
  result?: Record<string, unknown>
  error?: string
  created_at: string
  started_at?: string
  completed_at?: string
  duration_seconds?: number
}

export interface CoachingFeedback {
  todays_recap: string
  key_takeaway: string
  looking_ahead: string
  coach_notes: string
  athlete_message: string
}

export interface WorkoutAnalysis {
  daily_summary: string
  no_data: boolean
  workouts: unknown[]
}

export interface RestitutionAnalysis {
  data_quality_note: string
  trend_analysis: string
  load_recovery_correlation: string
  overall_recovery_quality: "good" | "adequate" | "concerning"
  coach_recommendations: string[]
}

export interface StoredAnalysis {
  athlete_id: number
  date: string
  workout_analysis: WorkoutAnalysis | null
  restitution_analysis: RestitutionAnalysis | null
  coaching_feedback: CoachingFeedback | null
  analyzed_at: string
}

export async function createDailyAnalysisTask(
  athleteId: number,
  date: string,
): Promise<TaskResponse> {
  return apiFetch<TaskResponse>("/api/v1/tasks", {
    method: "POST",
    body: JSON.stringify({
      task_type: "daily_llm_analysis",
      parameters: { date },
    }),
  })
}

export async function getTaskStatus(taskId: string): Promise<TaskResponse> {
  return apiFetch<TaskResponse>(`/api/v1/tasks/${taskId}`)
}

export async function fetchStoredAnalysis(
  athleteId: number,
  date: string,
): Promise<StoredAnalysis | null> {
  return apiFetch<StoredAnalysis | null>(
    `/api/v1/athlete/${athleteId}/daily-analysis?date=${date}`,
  )
}
