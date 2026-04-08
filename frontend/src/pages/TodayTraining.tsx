import { useEffect, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Loader2, CheckCircle } from "lucide-react"
import { Button } from "../components/ui/button"
import { RestitutionForm } from "../components/RestitutionForm"
import { ActivityCard } from "../components/ActivityCard"
import { fetchCurrentAthlete } from "../api/auth"
import { fetchDetailedWorkouts } from "../api/workouts"
import { fetchDailyEntry, saveDailyEntry } from "../api/dailyEntry"
import type { Restitution, ActivityAssessment } from "../api/dailyEntry"

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

type AssessmentMap = Record<number, Pick<ActivityAssessment, "rpe" | "notes">>

export function TodayTraining() {
  const today = todayDate()
  const queryClient = useQueryClient()

  const [restitution, setRestitution] = useState<Restitution>({})
  const [assessments, setAssessments] = useState<AssessmentMap>({})
  const [saved, setSaved] = useState(false)

  const { data: athlete, isLoading: loadingAthlete, error: athleteError } = useQuery({
    queryKey: ["athlete"],
    queryFn: fetchCurrentAthlete,
  })

  const { data: workouts, isLoading: loadingWorkouts } = useQuery({
    queryKey: ["workouts", athlete?.athlete_id, today],
    queryFn: () => fetchDetailedWorkouts(athlete!.athlete_id, today),
    enabled: !!athlete,
  })

  const { data: existingEntry } = useQuery({
    queryKey: ["daily-entry", athlete?.athlete_id, today],
    queryFn: () => fetchDailyEntry(athlete!.athlete_id, today),
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

  // Initialise RPE=5 for any new assessable activity not yet in state
  useEffect(() => {
    if (!workouts) return
    const assessable = workouts.filter((w) => w.session.commute === "no" && w.activity_id)
    setAssessments((prev) => {
      const next = { ...prev }
      for (const w of assessable) {
        if (w.activity_id && !(w.activity_id in next)) {
          next[w.activity_id] = { rpe: 5 }
        }
      }
      return next
    })
  }, [workouts])

  const saveMutation = useMutation({
    mutationFn: () => {
      const assessable = (workouts ?? []).filter(
        (w) => w.session.commute === "no" && w.activity_id
      )
      const activityAssessments: ActivityAssessment[] = assessable
        .filter((w) => w.activity_id && assessments[w.activity_id])
        .map((w) => ({
          activity_id: w.activity_id!,
          activity_name: w.session.name ?? w.session.category,
          rpe: assessments[w.activity_id!].rpe,
          notes: assessments[w.activity_id!].notes,
        }))

      return saveDailyEntry(athlete!.athlete_id, {
        date: today,
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

  const assessableWorkouts = (workouts ?? []).filter(
    (w) => w.session.commute === "no"
  )

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
    <div className="mx-auto max-w-2xl space-y-6 p-4 pb-24 sm:p-6">
      <div>
        <h1 className="text-2xl font-bold">Today's Training</h1>
        <p className="text-sm text-muted-foreground">{formatDate(today)}</p>
      </div>

      <RestitutionForm value={restitution} onChange={setRestitution} />

      <section className="space-y-4">
        <h2 className="font-semibold text-muted-foreground">
          Workouts
          {loadingWorkouts && (
            <Loader2 className="ml-2 inline h-4 w-4 animate-spin" />
          )}
        </h2>

        {!loadingWorkouts && assessableWorkouts.length === 0 && (
          <p className="text-sm text-muted-foreground">No workouts found for today.</p>
        )}

        {assessableWorkouts.map((workout) => {
          const id = workout.activity_id
          if (!id) return null
          return (
            <ActivityCard
              key={id}
              workout={workout}
              value={assessments[id] ?? { rpe: 5 }}
              onChange={(v) => setAssessments((prev) => ({ ...prev, [id]: v }))}
            />
          )
        })}
      </section>

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
            <Button
              onClick={() => saveMutation.mutate()}
              disabled={saveMutation.isPending}
            >
              {saveMutation.isPending && (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              )}
              Save Today's Entry
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
