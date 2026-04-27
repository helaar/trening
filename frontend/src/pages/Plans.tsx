import { useState, useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import { Plus, Loader2, CalendarDays, X } from "lucide-react"
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
  day_off: "Day off",
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

// All dates from start through end, inclusive
function dateRange(start: string, end: string): string[] {
  const result: string[] = []
  const d = new Date(start + "T00:00:00Z")
  const last = new Date(end + "T00:00:00Z")
  while (d <= last) {
    result.push(isoDate(d))
    d.setUTCDate(d.getUTCDate() + 1)
  }
  return result
}

interface DatePickerModalProps {
  onSelect: (date: string) => void
  onClose: () => void
}

function DatePickerModal({ onSelect, onClose }: DatePickerModalProps) {
  const [value, setValue] = useState(isoDate(new Date()))

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="w-80 rounded-lg border bg-background p-5 shadow-lg">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="font-semibold">Pick a date</h2>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X className="h-4 w-4" />
          </button>
        </div>
        <input
          type="date"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          className="mb-4 flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />
        <Button className="w-full" onClick={() => value && onSelect(value)}>
          Add plan for this date
        </Button>
      </div>
    </div>
  )
}

export function Plans() {
  const today = isoDate(new Date())
  // Wide fetch range — 3 days back to 1 year forward
  const rangeStart = isoDate(offsetDate(new Date(), -3))
  const rangeEnd = isoDate(offsetDate(new Date(), 365))

  const [selection, setSelection] = useState<Selection>(null)
  const [showDatePicker, setShowDatePicker] = useState(false)

  const { data: athlete, isLoading: loadingAthlete } = useQuery({
    queryKey: ["athlete"],
    queryFn: fetchCurrentAthlete,
  })

  const { data: plans = [], isLoading: loadingPlans } = useQuery({
    queryKey: ["plans", athlete?.athlete_id, rangeStart, rangeEnd],
    queryFn: () => fetchPlansForRange(athlete!.athlete_id, rangeStart, rangeEnd),
    enabled: !!athlete,
  })

  const plansByDate = useMemo(() => {
    const map = new Map<string, PlannedActivity[]>()
    for (const plan of plans) {
      const list = map.get(plan.date) ?? []
      list.push(plan)
      map.set(plan.date, list)
    }
    return map
  }, [plans])

  // Only show dates that actually have plans
  const visibleDates = useMemo(
    () => [...new Set(plans.map((p) => p.date))].sort(),
    [plans]
  )

  function handleSaved(saved: PlannedActivity) {
    setSelection({ kind: "existing", plan: saved })
  }

  function handleDeleted() {
    setSelection(null)
  }

  function handleDatePicked(date: string) {
    setShowDatePicker(false)
    setSelection({ kind: "new", date })
  }

  const selectedPlan = selection?.kind === "existing" ? selection.plan : undefined
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
      {showDatePicker && (
        <DatePickerModal onSelect={handleDatePicked} onClose={() => setShowDatePicker(false)} />
      )}

      {/* Header */}
      <header className="flex items-center justify-between border-b px-4 py-3">
        <div className="flex items-center gap-2">
          <CalendarDays className="h-5 w-5 text-muted-foreground" />
          <h1 className="text-lg font-bold">Training Plans</h1>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={() => setShowDatePicker(true)}>
            <Plus className="h-4 w-4" />
            Add plan
          </Button>
          <Link to="/">
            <Button variant="ghost" size="sm">
              Back
            </Button>
          </Link>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Left — plan list */}
        <aside className="w-72 shrink-0 overflow-y-auto border-r">
          {loadingPlans && (
            <div className="flex justify-center py-8">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          )}

          {!loadingPlans && visibleDates.length === 0 && (
            <p className="px-4 py-6 text-xs text-muted-foreground">
              No plans yet. Use "Add plan" to get started.
            </p>
          )}

          {!loadingPlans &&
            visibleDates.map((date) => {
              const dayPlans = plansByDate.get(date) ?? []
              const isToday = date === today
              const isPast = date < today

              return (
                <div key={date} className={cn("border-b", isPast && "opacity-60")}>
                  {/* Date heading */}
                  <div
                    className={cn(
                      "px-3 py-2",
                      isToday && "bg-primary/5"
                    )}
                  >
                    <span
                      className={cn(
                        "text-xs font-bold uppercase tracking-wider",
                        isToday ? "text-primary" : "text-foreground"
                      )}
                    >
                      {isToday ? "Today · " : ""}
                      {formatDateHeading(date)}
                    </span>
                  </div>

                  {/* Indented activity rows */}
                  {dayPlans.map((plan) => {
                    const isSelected =
                      selection?.kind === "existing" && selection.plan.id === plan.id
                    return (
                      <button
                        key={plan.id}
                        onClick={() => setSelection({ kind: "existing", plan })}
                        className={cn(
                          "w-full border-l-2 py-2 pl-5 pr-3 text-left text-sm transition-colors hover:bg-accent",
                          isSelected
                            ? "border-primary bg-accent"
                            : "border-transparent"
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
              <p className="text-sm">or use "Add plan" to schedule a new one.</p>
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
