import { useEffect, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Loader2, CheckCircle, AlertCircle, ChevronLeft, ChevronRight, RefreshCw, Settings, CalendarDays, Trash2 } from "lucide-react"
import { Link } from "@tanstack/react-router"
import { Button } from "../components/ui/button"
import { RestitutionForm } from "../components/RestitutionForm"
import { ActivityCard } from "../components/ActivityCard"
import { fetchCurrentAthlete } from "../api/auth"
import { fetchDetailedWorkouts, deleteWorkout } from "../api/workouts"
import { fetchDailyEntry, saveDailyEntry } from "../api/dailyEntry"
import type { Restitution, ActivityAssessment } from "../api/dailyEntry"
import { fetchPlansForDate } from "../api/plans"
import { PlanCard } from "../components/PlanCard"

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

type AssessmentMap = Record<number, { rpe?: number; notes?: string }>

export function TodayTraining() {
  const queryClient = useQueryClient()

  const [selectedDate, setSelectedDate] = useState<string>(todayDate())
  const isToday = selectedDate === todayDate()

  const [restitution, setRestitution] = useState<Restitution>({})
  const [assessments, setAssessments] = useState<AssessmentMap>({})
  const [saved, setSaved] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<{ activityId: number; name: string } | null>(null)

  useEffect(() => {
    setRestitution({})
    setAssessments({})
    setSaved(false)
  }, [selectedDate])

  function goToPrevDay() {
    setSelectedDate((prev) => {
      const d = new Date(prev)
      d.setUTCDate(d.getUTCDate() - 1)
      return d.toISOString().split("T")[0]
    })
  }

  function goToNextDay() {
    setSelectedDate((prev) => {
      const d = new Date(prev)
      d.setUTCDate(d.getUTCDate() + 1)
      return d.toISOString().split("T")[0]
    })
  }

  const { data: athlete, isLoading: loadingAthlete, error: athleteError } = useQuery({
    queryKey: ["athlete"],
    queryFn: fetchCurrentAthlete,
  })

  const { data: workouts, isLoading: loadingWorkouts, error: workoutsError } = useQuery({
    queryKey: ["workouts", athlete?.athlete_id, selectedDate],
    queryFn: () => fetchDetailedWorkouts(athlete!.athlete_id, selectedDate),
    enabled: !!athlete,
  })

  const syncMutation = useMutation({
    mutationFn: () => fetchDetailedWorkouts(athlete!.athlete_id, selectedDate, true),
    onSuccess: (data) => {
      queryClient.setQueryData(["workouts", athlete?.athlete_id, selectedDate], data)
    },
  })

  const { data: existingEntry } = useQuery({
    queryKey: ["daily-entry", athlete?.athlete_id, selectedDate],
    queryFn: () => fetchDailyEntry(athlete!.athlete_id, selectedDate),
    enabled: !!athlete,
  })

  const { data: plans } = useQuery({
    queryKey: ["plans", athlete?.athlete_id, selectedDate],
    queryFn: () => fetchPlansForDate(athlete!.athlete_id, selectedDate),
    enabled: !!athlete,
  })

  // Pre-fill forms from existing entry
  useEffect(() => {
    if (existingEntry) {
      if (existingEntry.restitution) setRestitution(existingEntry.restitution)
      const map: AssessmentMap = {}
      for (const a of existingEntry.activity_assessments) {
        map[a.activity_id] = { rpe: a.rpe, notes: a.notes }
      }
      setAssessments(map)
    }
  }, [existingEntry])

  // Initialise RPE=5 for non-commute activities not yet in state
  useEffect(() => {
    if (!workouts) return
    setAssessments((prev) => {
      const next = { ...prev }
      workouts.forEach((w, i) => {
        if (w.session.commute !== "no") return
        const key = workoutKey(w, i)
        if (!(key in next)) next[key] = {}
      })
      return next
    })
  }, [workouts])

  const allWorkouts = workouts ?? []

  const saveMutation = useMutation({
    mutationFn: () => {
      const activityAssessments: ActivityAssessment[] = allWorkouts
        .filter((w) => w.session.commute === "no")
        .map((w, i) => {
          const key = workoutKey(w, i)
          const assessment = assessments[key]
          if (!w.activity_id || !assessment || assessment.rpe === undefined) return null
          return {
            activity_id: w.activity_id,
            activity_name: w.session.name ?? w.session.category,
            rpe: assessment.rpe,
            notes: assessment.notes,
          }
        })
        .filter((a): a is ActivityAssessment => a !== null)

      return saveDailyEntry(athlete!.athlete_id, {
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
    mutationFn: (activityId: number) => deleteWorkout(athlete!.athlete_id, activityId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["workouts", athlete?.athlete_id, selectedDate] })
      setPendingDelete(null)
    },
  })

  if (loadingAthlete) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (athleteError) {
    return (
      <div className="flex min-h-screen items-center justify-center p-4">
        <p className="text-sm text-destructive">
          Could not load athlete. Is the backend running?
        </p>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-4 pb-24 sm:p-6 sm:pb-24">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold">{isToday ? "Today's Training" : "Training"}</h1>
          <p className="text-sm text-muted-foreground">{formatDate(selectedDate)}</p>
        </div>
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="icon" onClick={goToPrevDay} aria-label="Previous day">
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon" onClick={goToNextDay} disabled={isToday} aria-label="Next day">
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
          <Link to="/plans">
            <Button variant="ghost" size="icon" aria-label="Training plans">
              <CalendarDays className="h-4 w-4" />
            </Button>
          </Link>
          <Link to="/settings">
            <Button variant="ghost" size="icon" aria-label="Athlete settings">
              <Settings className="h-4 w-4" />
            </Button>
          </Link>
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
        <h2 className="font-semibold text-muted-foreground">
          Workouts
          {loadingWorkouts && <Loader2 className="ml-2 inline h-4 w-4 animate-spin" />}
        </h2>

        {workoutsError && (
          <div className="flex items-center gap-2 rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
            <AlertCircle className="h-4 w-4 shrink-0" />
            Failed to load workouts: {(workoutsError as Error).message}
          </div>
        )}

        {!loadingWorkouts && !workoutsError && allWorkouts.length === 0 && (
          <p className="text-sm text-muted-foreground">No workouts found for {isToday ? "today" : "this day"}.</p>
        )}

        {allWorkouts.map((workout, index) => {
          const key = workoutKey(workout, index)
          return (
            <div key={key} className="relative group">
              <ActivityCard
                workout={workout}
                value={assessments[key] ?? {}}
                onChange={(v) => setAssessments((prev) => ({ ...prev, [key]: v }))}
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

      {pendingDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-sm rounded-lg border bg-background p-6 shadow-lg space-y-4">
            <h3 className="font-semibold">Delete workout?</h3>
            <p className="text-sm text-muted-foreground">
              "{pendingDelete.name}" will be removed from the local database. This will not delete it from Strava.
            </p>
            {deleteMutation.isError && (
              <p className="text-sm text-destructive">Delete failed. Try again.</p>
            )}
            <div className="flex justify-end gap-2">
              <Button variant="ghost" onClick={() => setPendingDelete(null)} disabled={deleteMutation.isPending}>
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

      <div className="fixed bottom-0 left-0 right-0 border-t bg-background/95 p-4 backdrop-blur">
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
