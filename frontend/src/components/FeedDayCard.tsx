import * as AccordionPrimitive from "@radix-ui/react-accordion"
import { ChevronDown, Pencil } from "lucide-react"
import { Link } from "@tanstack/react-router"
import { Accordion, AccordionContent, AccordionItem } from "./ui/accordion"
import type { FeedDay } from "../api/feed"
import type { WorkoutAnalysis } from "../api/workouts"
import type { ActivityAssessment } from "../api/dailyEntry"

const SPORT_EMOJI: Record<string, string> = {
  cycling: "🚴",
  running: "🏃",
  strength: "💪",
  skiing_cross: "⛷️",
  skiing_alpine: "⛷️",
  day_off: "😴",
  other: "🏋️",
}

function sportEmoji(sport: string): string {
  return SPORT_EMOJI[sport] ?? "🏋️"
}

function isCommute(workout: WorkoutAnalysis): boolean {
  return workout.session.commute !== "no"
}

function formatDuration(sec: number): string {
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  return h > 0 ? `${h}h ${m}m` : `${m}m`
}

function formatDate(iso: string): string {
  return new Date(iso + "T00:00:00").toLocaleDateString("en-US", {
    weekday: "short",
    day: "numeric",
    month: "short",
  })
}

interface Severity {
  points: number
  missing: string[]
  emptyDay: boolean
}

function computeSeverity(day: FeedDay, nonCommutes: WorkoutAnalysis[]): Severity {
  const rpeById = new Map(day.activity_assessments.map((a) => [a.activity_id, a.rpe]))
  const missing: string[] = []
  let points = 0

  if (!day.restitution) {
    points += 2
    missing.push("restitution")
  }

  for (const w of nonCommutes) {
    if (w.activity_id !== null && !rpeById.has(w.activity_id)) {
      points += 1
      missing.push(`RPE for "${w.session.name ?? "workout"}"`)
    }
  }

  if (day.plans.length > 0 && nonCommutes.length === 0 && day.workouts.length === 0) {
    points += 2
    missing.push("activity for planned workout")
  }

  const emptyDay = day.workouts.length === 0 && day.plans.length === 0 && !day.restitution

  return { points, missing, emptyDay }
}

function workoutIsRace(workout: WorkoutAnalysis, assessments: ActivityAssessment[]): boolean {
  if (workout.session.tags?.includes("race")) return true
  const a = assessments.find((a) => a.activity_id === workout.activity_id)
  return a?.tags?.includes("race") ?? false
}

function dayHasRace(day: FeedDay): boolean {
  return day.workouts.some((w) => workoutIsRace(w, day.activity_assessments))
}

function RaceBadge() {
  return (
    <span className="text-xs px-2 py-0.5 rounded-full border font-medium bg-amber-100 text-amber-700 border-amber-300">
      🏆 Race
    </span>
  )
}

function GapBadge({ points, count }: { points: number; count: number }) {
  if (count === 0) return null
  const cls =
    points >= 3
      ? "bg-red-100 text-red-700 border-red-300"
      : points === 2
        ? "bg-orange-100 text-orange-700 border-orange-300"
        : "bg-yellow-100 text-yellow-700 border-yellow-300"
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${cls}`}>
      ⚠ {count} gap{count !== 1 ? "s" : ""}
    </span>
  )
}

export function FeedDayCard({ day }: { day: FeedDay }) {
  const nonCommutes = day.workouts.filter((w) => !isCommute(w))

  const rpeById = new Map(day.activity_assessments.map((a) => [a.activity_id, a.rpe]))

  const sportSet = new Set(day.workouts.map((w) => w.session.sport))
  const sportEmojis = [...sportSet].map(sportEmoji).join("")

  const totalTss = day.workouts.reduce(
    (sum, w) => sum + (w.metrics.training_stress_score ?? 0),
    0
  )

  const maxRpe =
    day.activity_assessments.length > 0
      ? Math.max(...day.activity_assessments.map((a) => a.rpe))
      : null

  const { points, missing, emptyDay } = computeSeverity(day, nonCommutes)
  const hasRace = dayHasRace(day)

  return (
    <Accordion type="single" collapsible>
      <AccordionItem value="day" className="group">
        <AccordionPrimitive.Header className="flex items-center gap-2 px-4">
          <AccordionPrimitive.Trigger className="flex flex-1 items-center gap-3 py-3 text-left min-w-0 transition-all">
            <span className="font-medium text-sm w-28 shrink-0">{formatDate(day.date)}</span>
            {sportEmojis && <span className="text-base shrink-0">{sportEmojis}</span>}
            {totalTss > 0 && (
              <span className="text-xs text-muted-foreground shrink-0">
                TSS {Math.round(totalTss)}
              </span>
            )}
            {maxRpe !== null && (
              <span className="text-xs text-muted-foreground shrink-0">RPE {maxRpe}</span>
            )}
            {emptyDay && (
              <span className="text-xs text-muted-foreground italic truncate">
                Rest day — add a plan?
              </span>
            )}
          </AccordionPrimitive.Trigger>
          <Link
            to="/"
            search={{ date: day.date }}
            className="shrink-0 p-1 text-muted-foreground hover:text-foreground rounded"
          >
            <Pencil className="h-3.5 w-3.5" />
          </Link>
          {hasRace && <RaceBadge />}
          <GapBadge points={points} count={missing.length} />
          <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground transition-transform duration-200 group-data-[state=open]:rotate-180" />
        </AccordionPrimitive.Header>
        <AccordionContent>
          <div className="space-y-1.5 pt-1 pb-3">
            {day.workouts.map((w, i) => {
              const rpe = w.activity_id !== null ? rpeById.get(w.activity_id) : undefined
              const missingRpe =
                nonCommutes.includes(w) && w.activity_id !== null && rpe === undefined
              return (
                <div key={w.activity_id ?? i} className="flex items-center gap-2 text-sm">
                  <span className="shrink-0">
                    {workoutIsRace(w, day.activity_assessments) ? "🏆" : sportEmoji(w.session.sport)}
                  </span>
                  <span className="flex-1 truncate">{w.session.name ?? "Workout"}</span>
                  <span className="text-muted-foreground shrink-0 text-xs">
                    {formatDuration(w.session.duration_sec)}
                  </span>
                  {w.metrics.training_stress_score !== null && (
                    <span className="text-muted-foreground shrink-0 text-xs w-16 text-right">
                      TSS {Math.round(w.metrics.training_stress_score)}
                    </span>
                  )}
                  {rpe !== undefined ? (
                    <span className="text-muted-foreground shrink-0 text-xs w-14 text-right">
                      RPE {rpe}
                    </span>
                  ) : missingRpe ? (
                    <span className="text-yellow-600 shrink-0 text-xs w-14 text-right">
                      no RPE
                    </span>
                  ) : (
                    <span className="shrink-0 w-14" />
                  )}
                </div>
              )
            })}

            {day.plans.map((p) => (
              <div key={p.id} className="flex items-center gap-2 text-sm text-muted-foreground">
                <span className="shrink-0">📋</span>
                <span className="flex-1 truncate">{p.name}</span>
                {day.workouts.length === 0 && (
                  <span className="text-orange-600 text-xs shrink-0">no log</span>
                )}
              </div>
            ))}

            {day.restitution && (
              <div className="flex items-center gap-1.5 text-sm text-muted-foreground flex-wrap pt-0.5">
                <span>😴</span>
                {day.restitution.readiness != null && (
                  <span>Readiness {day.restitution.readiness}/5</span>
                )}
                {day.restitution.hrv != null && <span>· HRV {day.restitution.hrv}</span>}
                {day.restitution.sleep_hours != null && (
                  <span>· Sleep {day.restitution.sleep_hours}h</span>
                )}
                {day.restitution.resting_hr != null && (
                  <span>· RHR {day.restitution.resting_hr}</span>
                )}
              </div>
            )}

            {missing.length > 0 && (
              <div className="text-xs text-muted-foreground pt-1 border-t mt-2">
                Missing: {missing.join(", ")}
              </div>
            )}
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  )
}
