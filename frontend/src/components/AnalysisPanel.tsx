import { useState } from "react"
import { Brain, ChevronDown, ChevronUp, AlertCircle } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"
import type { TaskStatus } from "../api/tasks"

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

function Section({ title, content, defaultOpen }: { title: string; content: string; defaultOpen: boolean }) {
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
      {open && (
        <div className="px-4 py-3 text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
          {content}
        </div>
      )}
    </div>
  )
}

export function AnalysisPanel({ status, progress, result, error, analyzedAt }: Props) {
  const coachingFeedback = result?.coaching_feedback as string | undefined
  const workoutAnalysis = result?.workout_analysis as string | undefined

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
            {coachingFeedback && (
              <Section title="Coaching Feedback" content={coachingFeedback} defaultOpen={true} />
            )}
            {workoutAnalysis && (
              <Section title="Performance Analysis" content={workoutAnalysis} defaultOpen={false} />
            )}
            {!coachingFeedback && !workoutAnalysis && (
              <p className="text-sm text-gray-500">No analysis output returned.</p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
