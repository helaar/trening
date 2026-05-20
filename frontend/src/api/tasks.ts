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

export interface RiskFlag {
  description: string
  severity: "low" | "moderate" | "high"
}

export interface WorkoutOutput {
  activity_id: number | null
  session_name: string
  is_commute: boolean
  is_erg_mode: boolean
  executive_summary: string
  quantitative_summary: string
  qualitative_assessment: string
  progress_indicators: string
  risk_flags: RiskFlag[]
  coach_recommendations: string[]
  commute_note: string
  data_gaps: string[]
}

export interface WorkoutAnalysis {
  daily_summary: string
  no_data: boolean
  workouts: WorkoutOutput[]
}

export interface RecoveryBaseline {
  hrv_typical: number | null
  resting_hr_typical: number | null
  sleep_hours_typical_weekday: number | null
  sleep_hours_typical_weekend: number | null
  sleep_quality_typical: number | null
  readiness_typical: number | null
}

export interface RestitutionAnalysis {
  data_quality_note: string
  recovery_baseline: RecoveryBaseline
  trend_analysis: string
  load_recovery_correlation: string
  risk_flags: RiskFlag[]
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
