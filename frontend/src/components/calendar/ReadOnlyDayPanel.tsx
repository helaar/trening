import { useQuery } from "@tanstack/react-query"
import { Loader2 } from "lucide-react"
import { fetchCoachAthleteFeed, fetchCoachAthleteDailyAnalysis } from "../../api/coach"
import { AnalysisPanel } from "../AnalysisPanel"

function formatDate(iso: string): string {
  return new Date(iso + "T00:00:00").toLocaleDateString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  })
}

interface ReadOnlyDayPanelProps {
  athleteId: number
  selectedDate: string
}

export function ReadOnlyDayPanel({ athleteId, selectedDate }: ReadOnlyDayPanelProps) {
  const { data: feed, isLoading: loadingFeed } = useQuery({
    queryKey: ["coach-feed", athleteId, selectedDate, selectedDate],
    queryFn: () => fetchCoachAthleteFeed(athleteId, selectedDate, selectedDate),
  })

  const { data: analysis } = useQuery({
    queryKey: ["coach-daily-analysis", athleteId, selectedDate],
    queryFn: () => fetchCoachAthleteDailyAnalysis(athleteId, selectedDate),
  })

  const day = feed?.[0]

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-2xl space-y-6 p-4 sm:p-6">
        <div>
          <h1 className="text-2xl font-bold">Training</h1>
          <p className="text-sm text-muted-foreground">{formatDate(selectedDate)}</p>
        </div>

        {loadingFeed && (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        )}

        {!loadingFeed && day?.restitution && (
          <section className="space-y-2">
            <h2 className="font-semibold text-muted-foreground">Restitution</h2>
            <div className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-3">
              {day.restitution.sleep_hours != null && (
                <div>Sleep: {day.restitution.sleep_hours}h</div>
              )}
              {day.restitution.sleep_quality != null && (
                <div>Sleep quality: {day.restitution.sleep_quality}/5</div>
              )}
              {day.restitution.hrv != null && <div>HRV: {day.restitution.hrv}</div>}
              {day.restitution.resting_hr != null && (
                <div>Resting HR: {day.restitution.resting_hr}</div>
              )}
              {day.restitution.readiness != null && (
                <div>Readiness: {day.restitution.readiness}/5</div>
              )}
            </div>
            {day.restitution.comment && (
              <p className="text-sm text-muted-foreground italic">{day.restitution.comment}</p>
            )}
          </section>
        )}

        {!loadingFeed && (day?.plans?.length ?? 0) > 0 && (
          <section className="space-y-2">
            <h2 className="font-semibold text-muted-foreground">Plan</h2>
            {day!.plans.map((plan) => (
              <div key={plan.id} className="rounded-md border px-3 py-2 text-sm">
                <span className="font-medium">{plan.name}</span>
                {plan.estimated_duration_min ? ` · ${plan.estimated_duration_min}m` : ""}
              </div>
            ))}
          </section>
        )}

        {!loadingFeed && (day?.workouts?.length ?? 0) === 0 && (
          <p className="text-sm text-muted-foreground">No workouts recorded for this day.</p>
        )}

        {!loadingFeed && (day?.workouts?.length ?? 0) > 0 && (
          <section className="space-y-2">
            <h2 className="font-semibold text-muted-foreground">Workouts</h2>
            {day!.workouts.map((workout, i) => (
              <div key={i} className="rounded-md border px-3 py-2 text-sm">
                <span className="font-medium">
                  {workout.session.name ?? workout.session.sport}
                </span>
                {workout.metrics?.training_stress_score
                  ? ` · ${Math.round(workout.metrics.training_stress_score)} TSS`
                  : ""}
              </div>
            ))}
          </section>
        )}

        {analysis && (
          <AnalysisPanel
            status="completed"
            progress={1}
            result={{
              workout_analysis: analysis.workout_analysis,
              restitution_analysis: analysis.restitution_analysis,
              coaching_feedback: analysis.coaching_feedback,
            }}
            analyzedAt={analysis.analyzed_at}
          />
        )}
      </div>
    </div>
  )
}
