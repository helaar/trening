import { useState, type ReactNode } from "react"
import { Brain, ChevronDown, ChevronUp, AlertCircle } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"
import type {
  TaskStatus,
  CoachingFeedback,
  WorkoutAnalysis,
  WorkoutOutput,
  RestitutionAnalysis,
  RiskFlag,
  WeeklyAssessment,
} from "../api/tasks"
import { WeeklyIntensityBar } from "./WeeklyIntensityBar"

interface Props {
  status: TaskStatus
  progress: number
  result?: Record<string, unknown>
  error?: string
  analyzedAt?: string
}

function ProgressBar({ progress }: { progress: number }) {
  const pct = Math.round(progress * 100)
  return (
    <div className="w-full bg-gray-200 rounded-full h-2">
      <div
        className="bg-blue-500 h-2 rounded-full transition-all duration-500"
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}

function Section({ title, children, defaultOpen }: { title: string; children: ReactNode; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border rounded-md overflow-hidden">
      <button
        className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 hover:bg-gray-100 text-left text-sm font-medium"
        onClick={() => setOpen((o) => !o)}
      >
        {title}
        {open ? <ChevronUp className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
      </button>
      {open && <div className="px-4 py-4 text-sm text-gray-700 space-y-4">{children}</div>}
    </div>
  )
}

function FieldBlock({ label, value }: { label: string; value: string }) {
  if (!value) return null
  return (
    <div>
      <h4 className="font-semibold text-gray-800 mb-1">{label}</h4>
      <p className="leading-relaxed whitespace-pre-wrap">{value}</p>
    </div>
  )
}

function ListBlock({ label, items }: { label: string; items: string[] }) {
  if (!items.length) return null
  return (
    <div>
      <h4 className="font-semibold text-gray-800 mb-1">{label}</h4>
      <ul className="list-disc list-inside space-y-1">
        {items.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
    </div>
  )
}

const severityStyle: Record<string, string> = {
  low: "border-yellow-200 bg-yellow-50 text-yellow-800",
  moderate: "border-orange-200 bg-orange-50 text-orange-800",
  high: "border-red-200 bg-red-50 text-red-800",
}

function RiskFlagList({ label, flags }: { label: string; flags: RiskFlag[] }) {
  if (!flags.length) return null
  return (
    <div>
      <h4 className="font-semibold text-gray-800 mb-1">{label}</h4>
      <div className="space-y-1">
        {flags.map((flag, i) => (
          <div key={i} className={`border rounded px-2 py-1 text-xs ${severityStyle[flag.severity]}`}>
            <span className="font-medium capitalize">{flag.severity}:</span> {flag.description}
          </div>
        ))}
      </div>
    </div>
  )
}

function WorkoutItem({ workout }: { workout: WorkoutOutput }) {
  return (
    <div className="border-l-2 border-blue-200 pl-3 space-y-3">
      <h4 className="font-semibold text-gray-900">
        {workout.session_name}
        {workout.is_erg_mode && <span className="ml-1 text-xs font-normal text-gray-500">(ERG)</span>}
      </h4>
      {workout.is_commute ? (
        workout.commute_note && <p className="text-xs italic text-gray-500">{workout.commute_note}</p>
      ) : (
        <>
          <FieldBlock label="Executive Summary" value={workout.executive_summary} />
          <FieldBlock label="Quantitative Summary" value={workout.quantitative_summary} />
          <FieldBlock label="Qualitative Assessment" value={workout.qualitative_assessment} />
          <FieldBlock label="Progress Indicators" value={workout.progress_indicators} />
          <RiskFlagList label="Risk Flags" flags={workout.risk_flags} />
          <ListBlock label="Coach Recommendations" items={workout.coach_recommendations} />
          <ListBlock label="Data Gaps" items={workout.data_gaps} />
        </>
      )}
    </div>
  )
}

const qualityStyle: Record<string, string> = {
  good: "text-green-700 font-semibold",
  adequate: "text-yellow-700 font-semibold",
  concerning: "text-red-700 font-semibold",
}

function RecoveryContent({ r }: { r: RestitutionAnalysis }) {
  const b = r.recovery_baseline
  const baselineItems = [
    b.hrv_typical != null ? `HRV: ${b.hrv_typical}` : null,
    b.resting_hr_typical != null ? `Resting HR: ${b.resting_hr_typical} bpm` : null,
    b.sleep_hours_typical_weekday != null ? `Sleep (weekday): ${b.sleep_hours_typical_weekday}h` : null,
    b.sleep_hours_typical_weekend != null ? `Sleep (weekend): ${b.sleep_hours_typical_weekend}h` : null,
    b.sleep_quality_typical != null ? `Sleep quality: ${b.sleep_quality_typical}` : null,
    b.readiness_typical != null ? `Readiness: ${b.readiness_typical}` : null,
  ].filter((x): x is string => x !== null)

  return (
    <>
      {r.data_quality_note && (
        <div className="border border-amber-200 bg-amber-50 rounded px-3 py-2 text-xs text-amber-800">
          {r.data_quality_note}
        </div>
      )}
      {baselineItems.length > 0 && <ListBlock label="Recovery Baseline" items={baselineItems} />}
      <FieldBlock label="Trend Analysis" value={r.trend_analysis} />
      <FieldBlock label="Load-Recovery Correlation" value={r.load_recovery_correlation} />
      <RiskFlagList label="Risk Flags" flags={r.risk_flags} />
      <div>
        <h4 className="font-semibold text-gray-800 mb-1">Overall Recovery Quality</h4>
        <span className={qualityStyle[r.overall_recovery_quality] ?? "font-semibold"}>
          {r.overall_recovery_quality.charAt(0).toUpperCase() + r.overall_recovery_quality.slice(1)}
        </span>
      </div>
      <ListBlock label="Coach Recommendations" items={r.coach_recommendations} />
    </>
  )
}

export function AnalysisPanel({ status, progress, result, error, analyzedAt }: Props) {
  const coachingFeedback = result?.coaching_feedback as CoachingFeedback | null | undefined
  const workoutAnalysis = result?.workout_analysis as WorkoutAnalysis | null | undefined
  const restitutionAnalysis = result?.restitution_analysis as RestitutionAnalysis | null | undefined
  const weeklyAssessment = result?.weekly_philosophy_assessment as WeeklyAssessment | null | undefined

  const coachingText = coachingFeedback?.athlete_message
  const hasWorkout = workoutAnalysis && !workoutAnalysis.no_data
  const hasRestitution = !!restitutionAnalysis

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Brain className="w-4 h-4 text-blue-500" />
          AI Coaching Analysis
          {analyzedAt && (
            <span className="ml-auto text-xs font-normal text-gray-400">
              {new Date(analyzedAt).toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" })}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {(status === "pending" || status === "running") && (
          <div className="space-y-2">
            <p className="text-sm text-gray-500">
              {status === "pending" ? "Starting analysis…" : "Analysing your training…"}
            </p>
            <ProgressBar progress={progress} />
          </div>
        )}

        {status === "failed" && (
          <div className="flex items-start gap-2 text-red-600 text-sm">
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
            <span>{error ?? "Analysis failed. Please try again."}</span>
          </div>
        )}

        {status === "completed" && (
          <div className="space-y-2">
            {weeklyAssessment && <WeeklyIntensityBar a={weeklyAssessment} />}
            {coachingText && (
              <Section title="Coaching Feedback" defaultOpen={true}>
                <p className="leading-relaxed whitespace-pre-wrap">{coachingText}</p>
              </Section>
            )}
            {hasWorkout && (
              <Section title="Performance Analysis" defaultOpen={false}>
                <FieldBlock label="Daily Summary" value={workoutAnalysis.daily_summary} />
                {workoutAnalysis.workouts.map((w, i) => (
                  <WorkoutItem key={w.activity_id ?? i} workout={w} />
                ))}
              </Section>
            )}
            {hasRestitution && (
              <Section title="Recovery Analysis" defaultOpen={false}>
                <RecoveryContent r={restitutionAnalysis} />
              </Section>
            )}
            {!coachingText && !hasWorkout && !hasRestitution && !weeklyAssessment && (
              <p className="text-sm text-gray-500">No analysis output returned.</p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
