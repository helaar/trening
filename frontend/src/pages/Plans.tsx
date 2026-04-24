import { useState, useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import { Plus, Loader2, CalendarDays } from "lucide-react"
import { Link } from "@tanstack/react-router"
import { Button } from "../components/ui/button"
import { PlanForm } from "../components/PlanForm"
import { fetchCurrentAthlete } from "../api/auth"
import { fetchPlansForRange } from "../api/plans"
import type { PlannedActivity, Sport } from "../api/plans"
import { cn } from "../lib/utils"

const SPORT_LABELS: Record<Sport, string> = {
  cycling: "Cycling",
  running: "Running",
  strength: "Strength",
  skiing_cross: "XC Skiing",
  skiing_alpine: "Alpine Skiing",
  other: "Other",
}

function isoDate(d: Date): string {
  return d.toISOString().split("T")[0]
}

function offsetDate(base: Date, days: number): Date {
  const d = new Date(base)
  d.setUTCDate(d.getUTCDate() + days)
  return d
}

function formatDateHeading(iso: string): string {
  const d = new Date(iso + "T00:00:00")
  return d.toLocaleDateString("en-US", {
    weekday: "long",
    month: "short",
    day: "numeric",
  })
}

type Selection =
  | { kind: "existing"; plan: PlannedActivity }
  | { kind: "new"; date: string }
  | null

export function Plans() {
  const today = isoDate(new Date())
  const rangeStart = isoDate(offsetDate(new Date(), -3))
  const rangeEnd = isoDate(offsetDate(new Date(), 90))

  const [selection, setSelection] = useState<Selection>(null)

  const { data: athlete, isLoading: loadingAthlete } = useQuery({
    queryKey: ["athlete"],
    queryFn: fetchCurrentAthlete,
  })

  const { data: plans = [], isLoading: loadingPlans } = useQuery({
    queryKey: ["plans", athlete?.athlete_id, rangeStart, rangeEnd],
    queryFn: () => fetchPlansForRange(athlete!.athlete_id, rangeStart, rangeEnd),
    enabled: !!athlete,
  })

  // Build sorted list of all dates in range that either have plans or are today/future
  const dates = useMemo(() => {
    const dateSet = new Set<string>()
    // Always include today through 90 days out
    for (let i = -3; i <= 90; i++) {
      dateSet.add(isoDate(offsetDate(new Date(), i)))
    }
    return Array.from(dateSet).sort()
  }, [])

  // Group plans by date
  const plansByDate = useMemo(() => {
    const map = new Map<string, PlannedActivity[]>()
    for (const plan of plans) {
      const list = map.get(plan.date) ?? []
      list.push(plan)
      map.set(plan.date, list)
    }
    return map
  }, [plans])

  // Dates that have plans or are today/future (hide empty past days)
  const visibleDates = useMemo(
    () => dates.filter((d) => d >= today || (plansByDate.get(d)?.length ?? 0) > 0),
    [dates, today, plansByDate]
  )

  function handleSaved(saved: PlannedActivity) {
    setSelection({ kind: "existing", plan: saved })
  }

  function handleDeleted() {
    setSelection(null)
  }

  const selectedPlan =
    selection?.kind === "existing" ? selection.plan : undefined
  const selectedDate =
    selection?.kind === "existing"
      ? selection.plan.date
      : selection?.kind === "new"
        ? selection.date
        : undefined

  if (loadingAthlete) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <div className="flex h-screen flex-col">
      {/* Header */}
      <header className="flex items-center justify-between border-b px-4 py-3">
        <div className="flex items-center gap-2">
          <CalendarDays className="h-5 w-5 text-muted-foreground" />
          <h1 className="text-lg font-bold">Training Plans</h1>
        </div>
        <Link to="/">
          <Button variant="ghost" size="sm">
            Back
          </Button>
        </Link>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Left — plan list */}
        <aside className="w-72 shrink-0 overflow-y-auto border-r">
          {loadingPlans && (
            <div className="flex justify-center py-8">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          )}

          {!loadingPlans &&
            visibleDates.map((date) => {
              const dayPlans = plansByDate.get(date) ?? []
              const isToday = date === today
              const isPast = date < today

              return (
                <div key={date} className={cn("border-b", isPast && "opacity-60")}>
                  {/* Date header */}
                  <div className="flex items-center justify-between px-3 py-2">
                    <span
                      className={cn(
                        "text-xs font-semibold uppercase tracking-wide",
                        isToday ? "text-primary" : "text-muted-foreground"
                      )}
                    >
                      {isToday ? "Today — " : ""}
                      {formatDateHeading(date)}
                    </span>
                    <button
                      onClick={() => setSelection({ kind: "new", date })}
                      className="rounded p-0.5 text-muted-foreground hover:bg-accent hover:text-foreground"
                      aria-label={`Add plan for ${date}`}
                    >
                      <Plus className="h-3.5 w-3.5" />
                    </button>
                  </div>

                  {/* Activities for this date */}
                  {dayPlans.length === 0 && (
                    <p className="px-3 pb-2 text-xs text-muted-foreground/60">No plans</p>
                  )}
                  {dayPlans.map((plan) => {
                    const isSelected =
                      selection?.kind === "existing" && selection.plan.id === plan.id
                    return (
                      <button
                        key={plan.id}
                        onClick={() => setSelection({ kind: "existing", plan })}
                        className={cn(
                          "w-full px-3 py-2 text-left text-sm transition-colors hover:bg-accent",
                          isSelected && "bg-accent"
                        )}
                      >
                        <div className="font-medium leading-snug">{plan.name}</div>
                        <div className="mt-0.5 flex flex-wrap gap-2 text-xs text-muted-foreground">
                          <span>{SPORT_LABELS[plan.sport]}</span>
                          {plan.estimated_duration_min && (
                            <span>{plan.estimated_duration_min} min</span>
                          )}
                          {plan.estimated_tss && <span>TSS {plan.estimated_tss}</span>}
                        </div>
                      </button>
                    )
                  })}
                </div>
              )
            })}
        </aside>

        {/* Right — detail / form */}
        <main className="flex-1 overflow-y-auto p-6">
          {!selection && (
            <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-muted-foreground">
              <CalendarDays className="h-10 w-10 opacity-30" />
              <p className="text-sm">Select a planned activity to view or edit,</p>
              <p className="text-sm">or press + next to a date to add one.</p>
            </div>
          )}

          {selection && athlete && (
            <div className="mx-auto max-w-lg">
              <h2 className="mb-5 text-base font-semibold">
                {selection.kind === "new"
                  ? `New plan — ${formatDateHeading(selection.date)}`
                  : formatDateHeading(selection.plan.date)}
              </h2>
              <PlanForm
                key={
                  selection.kind === "existing"
                    ? selection.plan.id
                    : `new-${selection.date}`
                }
                athleteId={athlete.athlete_id}
                date={selectedDate!}
                existing={selectedPlan}
                onSaved={handleSaved}
                onDeleted={handleDeleted}
              />
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
