import { useEffect, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Loader2,
  CheckCircle,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  RefreshCw,
  Brain,
  StickyNote,
  Trash2,
} from "lucide-react"
import { Button } from "../ui/button"
import { RestitutionForm } from "../RestitutionForm"
import { ActivityCard } from "../ActivityCard"
import { AnalysisPanel } from "../AnalysisPanel"
import {
  fetchDetailedWorkouts,
  deleteWorkout,
  createNote,
  updateNote,
  type WorkoutAnalysis,
} from "../../api/workouts"
import { fetchDailyEntry, saveDailyEntry } from "../../api/dailyEntry"
import type { Restitution, ActivityAssessment } from "../../api/dailyEntry"
import { fetchPlansForDate } from "../../api/plans"
import { createDailyAnalysisTask, fetchStoredAnalysis, getTaskStatus } from "../../api/tasks"
import { PlanCard } from "../PlanCard"

function todayDate(): string {
  return new Date().toISOString().split("T")[0]
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  })
}

// Use activity_id when available, fall back to array index
function workoutKey(workout: { activity_id: number | null }, index: number): number {
  return workout.activity_id ?? -(index + 1)
}

type AssessmentMap = Record<number, { rpe?: number; notes?: string; tags?: string[] }>

interface DayDetailPanelProps {
  athleteId: number
  selectedDate: string
  onDateChange: (date: string) => void
}

export function DayDetailPanel({ athleteId, selectedDate, onDateChange }: DayDetailPanelProps) {
  const queryClient = useQueryClient()
  const isToday = selectedDate === todayDate()

  const [restitution, setRestitution] = useState<Restitution>({})
  const [assessments, setAssessments] = useState<AssessmentMap>({})
  const [saved, setSaved] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<{ activityId: number; name: string } | null>(
    null
  )
  const [analysisTaskId, setAnalysisTaskId] = useState<string | null>(null)
  const [showNoteModal, setShowNoteModal] = useState(false)
  const [noteText, setNoteText] = useState("Day off")

  useEffect(() => {
    setRestitution({})
    setAssessments({})
    setSaved(false)
    setAnalysisTaskId(null)
  }, [selectedDate])

  function goToPrevDay() {
    const d = new Date(selectedDate)
    d.setUTCDate(d.getUTCDate() - 1)
    onDateChange(d.toISOString().split("T")[0])
  }

  function goToNextDay() {
    const d = new Date(selectedDate)
    d.setUTCDate(d.getUTCDate() + 1)
    onDateChange(d.toISOString().split("T")[0])
  }

  const { data: workouts, isLoading: loadingWorkouts, error: workoutsError } = useQuery({
    queryKey: ["workouts", athleteId, selectedDate],
    queryFn: () => fetchDetailedWorkouts(athleteId, selectedDate),
    staleTime: Infinity,
  })

  const syncMutation = useMutation({
    mutationFn: () => fetchDetailedWorkouts(athleteId, selectedDate, true),
    onSuccess: (data) => {
      queryClient.setQueryData(["workouts", athleteId, selectedDate], data)
    },
  })

  const { data: existingEntry } = useQuery({
    queryKey: ["daily-entry", athleteId, selectedDate],
    queryFn: () => fetchDailyEntry(athleteId, selectedDate),
  })

  const { data: plans } = useQuery({
    queryKey: ["plans", athleteId, selectedDate],
    queryFn: () => fetchPlansForDate(athleteId, selectedDate),
  })

  const { data: storedAnalysis } = useQuery({
    queryKey: ["daily-analysis", athleteId, selectedDate],
    queryFn: () => fetchStoredAnalysis(athleteId, selectedDate),
    staleTime: Infinity,
  })

  const { data: analysisTask } = useQuery({
    queryKey: ["task", analysisTaskId],
    queryFn: () => getTaskStatus(analysisTaskId!),
    enabled: !!analysisTaskId,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === "completed" || status === "failed" ? false : 3000
    },
  })

  async function triggerAnalysis() {
    const task = await createDailyAnalysisTask(athleteId, selectedDate)
    setAnalysisTaskId(task.task_id)
  }

  const analysisRunning =
    !!analysisTaskId &&
    analysisTask?.status !== "completed" &&
    analysisTask?.status !== "failed"

  // Invalidate cached result once the running task finishes
  useEffect(() => {
    if (analysisTask?.status === "completed" || analysisTask?.status === "failed") {
      queryClient.invalidateQueries({ queryKey: ["daily-analysis", athleteId, selectedDate] })
    }
  }, [analysisTask?.status])

  // Pre-fill forms from existing entry
  useEffect(() => {
    if (existingEntry) {
      if (existingEntry.restitution) setRestitution(existingEntry.restitution)
      const map: AssessmentMap = {}
      for (const a of existingEntry.activity_assessments) {
        map[a.activity_id] = { rpe: a.rpe, notes: a.notes, tags: a.tags }
      }
      setAssessments(map)
    }
  }, [existingEntry])

  // Initialise RPE and tags for non-commute activities not yet in state
  useEffect(() => {
    if (!workouts) return
    setAssessments((prev) => {
      const next = { ...prev }
      workouts.forEach((w, i) => {
        if (w.session.commute !== "no") return
        const key = workoutKey(w, i)
        if (!(key in next)) {
          next[key] = { tags: w.session.tags?.length ? [...w.session.tags] : undefined }
        }
      })
      return next
    })
  }, [workouts])

  const allWorkouts = workouts ?? []

  const saveMutation = useMutation({
    mutationFn: () => {
      const activityAssessments: ActivityAssessment[] = allWorkouts
        .filter((w) => w.session.commute === "no")
        .flatMap((w, i) => {
          const key = workoutKey(w, i)
          const assessment = assessments[key]
          if (!w.activity_id || !assessment || assessment.rpe === undefined) return []
          const entry: ActivityAssessment = {
            activity_id: w.activity_id,
            activity_name: w.session.name ?? w.session.category,
            rpe: assessment.rpe,
          }
          if (assessment.notes !== undefined) entry.notes = assessment.notes
          if (assessment.tags?.length) entry.tags = assessment.tags
          return [entry]
        })

      return saveDailyEntry(athleteId, {
        date: selectedDate,
        restitution: hasAnyRestitution(restitution) ? restitution : undefined,
        activity_assessments: activityAssessments,
      })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["daily-entry"] })
      setSaved(true)
      setTimeout(() => setSaved(false), 3000)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (activityId: number) => deleteWorkout(athleteId, activityId),
    onSuccess: (_, activityId) => {
      queryClient.setQueryData<WorkoutAnalysis[]>(
        ["workouts", athleteId, selectedDate],
        (old) => old?.filter((w) => w.activity_id !== activityId) ?? []
      )
      setPendingDelete(null)
    },
  })

  const createNoteMutation = useMutation({
    mutationFn: () => createNote(athleteId, selectedDate, noteText),
    onSuccess: (newNote) => {
      queryClient.setQueryData<WorkoutAnalysis[]>(
        ["workouts", athleteId, selectedDate],
        (old) => [...(old ?? []), newNote]
      )
      setShowNoteModal(false)
      setNoteText("Day off")
    },
  })

  const updateNoteMutation = useMutation({
    mutationFn: ({ activityId, text }: { activityId: number; text: string }) =>
      updateNote(athleteId, activityId, text),
    onSuccess: (updated) => {
      queryClient.setQueryData<WorkoutAnalysis[]>(
        ["workouts", athleteId, selectedDate],
        (old) => old?.map((w) => (w.activity_id === updated.activity_id ? updated : w)) ?? []
      )
    },
  })

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-2xl space-y-6 p-4 sm:p-6">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-bold">
              {isToday ? "Today's Training" : "Training"}
            </h1>
            <p className="text-sm text-muted-foreground">{formatDate(selectedDate)}</p>
          </div>
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="icon" onClick={goToPrevDay} aria-label="Previous day">
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={goToNextDay}
              disabled={isToday}
              aria-label="Next day"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => syncMutation.mutate()}
              disabled={syncMutation.isPending || loadingWorkouts}
              aria-label="Sync from Strava"
              title="Sync from Strava"
            >
              <RefreshCw className={`h-4 w-4 ${syncMutation.isPending ? "animate-spin" : ""}`} />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={triggerAnalysis}
              disabled={analysisRunning}
              aria-label="Run AI analysis"
              title="Run AI coaching analysis"
            >
              <Brain className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <RestitutionForm value={restitution} onChange={setRestitution} />

        {plans && plans.length > 0 && (
          <section className="space-y-4">
            <h2 className="font-semibold text-muted-foreground">Plan</h2>
            {plans.map((plan) => (
              <PlanCard key={plan.id} plan={plan} />
            ))}
          </section>
        )}

        <section className="space-y-4">
          <div className="flex items-center gap-2">
            <h2 className="font-semibold text-muted-foreground">
              Workouts
              {loadingWorkouts && <Loader2 className="ml-2 inline h-4 w-4 animate-spin" />}
            </h2>
            <button
              className="ml-auto p-1 rounded text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
              aria-label="Add day note"
              title="Add a note for this day"
              onClick={() => setShowNoteModal(true)}
            >
              <StickyNote className="h-4 w-4" />
            </button>
          </div>

          {workoutsError && (
            <div className="flex items-center gap-2 rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
              <AlertCircle className="h-4 w-4 shrink-0" />
              Failed to load workouts: {(workoutsError as Error).message}
            </div>
          )}

          {!loadingWorkouts && !workoutsError && allWorkouts.length === 0 && (
            <p className="text-sm text-muted-foreground">
              No workouts found for {isToday ? "today" : "this day"}.
            </p>
          )}

          {allWorkouts.map((workout, index) => {
            const key = workoutKey(workout, index)
            return (
              <div key={key} className="relative group">
                <ActivityCard
                  workout={workout}
                  value={assessments[key] ?? {}}
                  onChange={(v) => setAssessments((prev) => ({ ...prev, [key]: v }))}
                  onSaveNote={
                    workout.session.manual && workout.activity_id !== null
                      ? (text) =>
                          updateNoteMutation.mutate({ activityId: workout.activity_id!, text })
                      : undefined
                  }
                />
                {workout.activity_id !== null && (
                  <button
                    className="absolute top-3 right-3 p-1.5 rounded-md text-muted-foreground opacity-0 group-hover:opacity-100 hover:text-destructive hover:bg-destructive/10 transition-all"
                    aria-label="Delete workout"
                    onClick={() =>
                      setPendingDelete({
                        activityId: workout.activity_id!,
                        name: workout.session.name ?? workout.session.category,
                      })
                    }
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                )}
              </div>
            )
          })}
        </section>

        {analysisRunning && analysisTask && (
          <AnalysisPanel
            status={analysisTask.status}
            progress={analysisTask.progress}
            result={analysisTask.result}
            error={analysisTask.error}
          />
        )}
        {!analysisRunning && storedAnalysis && (
          <AnalysisPanel
            status="completed"
            progress={1}
            result={{
              workout_analysis: storedAnalysis.workout_analysis,
              restitution_analysis: storedAnalysis.restitution_analysis,
              coaching_feedback: storedAnalysis.coaching_feedback,
            }}
            analyzedAt={storedAnalysis.analyzed_at}
          />
        )}
      </div>

      {pendingDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-sm rounded-lg border bg-background p-6 shadow-lg space-y-4">
            <h3 className="font-semibold">Delete workout?</h3>
            <p className="text-sm text-muted-foreground">
              "{pendingDelete.name}" will be removed from the local database. This will not delete
              it from Strava.
            </p>
            {deleteMutation.isError && (
              <p className="text-sm text-destructive">Delete failed. Try again.</p>
            )}
            <div className="flex justify-end gap-2">
              <Button
                variant="ghost"
                onClick={() => setPendingDelete(null)}
                disabled={deleteMutation.isPending}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={() => deleteMutation.mutate(pendingDelete.activityId)}
                disabled={deleteMutation.isPending}
              >
                {deleteMutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                Delete
              </Button>
            </div>
          </div>
        </div>
      )}

      {showNoteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-sm rounded-lg border bg-background p-6 shadow-lg space-y-4">
            <h3 className="font-semibold">Add day note</h3>
            <p className="text-sm text-muted-foreground">
              Tell your coach what kind of day this was.
            </p>
            <input
              type="text"
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              value={noteText}
              onChange={(e) => setNoteText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && noteText.trim()) createNoteMutation.mutate()
                if (e.key === "Escape") setShowNoteModal(false)
              }}
              placeholder="Day off, Travel, Rest…"
              autoFocus
            />
            {createNoteMutation.isError && (
              <p className="text-sm text-destructive">Failed to save note. Try again.</p>
            )}
            <div className="flex justify-end gap-2">
              <Button
                variant="ghost"
                onClick={() => setShowNoteModal(false)}
                disabled={createNoteMutation.isPending}
              >
                Cancel
              </Button>
              <Button
                onClick={() => createNoteMutation.mutate()}
                disabled={createNoteMutation.isPending || !noteText.trim()}
              >
                {createNoteMutation.isPending && (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                )}
                Save
              </Button>
            </div>
          </div>
        </div>
      )}

      <div className="sticky bottom-0 border-t bg-background/95 p-4 backdrop-blur">
        <div className="mx-auto max-w-2xl flex items-center justify-between gap-4">
          {saved && (
            <span className="flex items-center gap-1.5 text-sm text-green-600">
              <CheckCircle className="h-4 w-4" />
              Saved
            </span>
          )}
          {saveMutation.isError && (
            <span className="text-sm text-destructive">Save failed. Try again.</span>
          )}
          <div className="ml-auto">
            <Button onClick={() => saveMutation.mutate()} disabled={saveMutation.isPending}>
              {saveMutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Save Entry
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

function hasAnyRestitution(r: Restitution): boolean {
  return (
    r.sleep_hours !== undefined ||
    r.hrv !== undefined ||
    r.resting_hr !== undefined ||
    (r.comment !== undefined && r.comment !== "")
  )
}
