import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"
import type { WorkoutAnalysis } from "@/api/workouts"
import type { ActivityAssessment } from "@/api/dailyEntry"

interface Props {
  workout: WorkoutAnalysis
  value: Pick<ActivityAssessment, "rpe" | "notes">
  onChange: (value: Pick<ActivityAssessment, "rpe" | "notes">) => void
}

function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

function rpeColor(rpe: number): string {
  if (rpe <= 3) return "bg-green-100 text-green-800"
  if (rpe <= 5) return "bg-yellow-100 text-yellow-800"
  if (rpe <= 7) return "bg-orange-100 text-orange-800"
  return "bg-red-100 text-red-800"
}

function sportLabel(category: string): string {
  const labels: Record<string, string> = {
    cycling: "Cycling",
    running: "Running",
    skiing: "Skiing",
    strength: "Strength",
    other: "Other",
  }
  return labels[category] ?? category
}

export function ActivityCard({ workout, value, onChange }: Props) {
  const { session, metrics } = workout

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="text-base">{session.name ?? sportLabel(session.category)}</CardTitle>
          <span className="shrink-0 rounded-full bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground">
            {sportLabel(session.category)}
          </span>
        </div>
        <div className="flex flex-wrap gap-3 text-sm text-muted-foreground">
          <span>{formatDuration(session.duration_sec)}</span>
          {session.distance_km > 0 && <span>{session.distance_km.toFixed(1)} km</span>}
          {metrics.heart_rate?.mean && (
            <span>{Math.round(metrics.heart_rate.mean)} bpm avg</span>
          )}
          {metrics.training_stress_score && (
            <span>TSS {Math.round(metrics.training_stress_score)}</span>
          )}
          {metrics.normalized_power && <span>NP {Math.round(metrics.normalized_power)}W</span>}
        </div>
      </CardHeader>

      <CardContent className="space-y-4 border-t pt-4">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label>RPE</Label>
            <span
              className={cn(
                "rounded-full px-2.5 py-0.5 text-sm font-semibold",
                rpeColor(value.rpe)
              )}
            >
              {value.rpe} / 10
            </span>
          </div>
          <input
            type="range"
            min={1}
            max={10}
            step={1}
            value={value.rpe}
            onChange={(e) => onChange({ ...value, rpe: Number(e.target.value) })}
            className="w-full accent-primary"
          />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Very easy</span>
            <span>Moderate</span>
            <span>Max effort</span>
          </div>
        </div>

        <div className="space-y-1.5">
          <Label>Notes</Label>
          <Textarea
            placeholder="How did it feel? Any observations…"
            value={value.notes ?? ""}
            onChange={(e) =>
              onChange({ ...value, notes: e.target.value || undefined })
            }
          />
        </div>
      </CardContent>
    </Card>
  )
}
