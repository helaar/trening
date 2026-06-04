import { useEffect } from "react"
import { useQuery } from "@tanstack/react-query"
import { Loader2, ChevronLeft, ChevronRight, CalendarDays } from "lucide-react"
import { useSearch, useNavigate } from "@tanstack/react-router"
import { fetchCurrentAthlete } from "../api/auth"
import { DayDetailPanel } from "../components/calendar/DayDetailPanel"
import { MonthView } from "../components/calendar/MonthView"
import { WeekView } from "../components/calendar/WeekView"
import { Button } from "../components/ui/button"
import { cn } from "../lib/utils"

type CalendarView = "month" | "week" | "day"

function todayDate(): string {
  return new Date().toISOString().split("T")[0]
}

function formatPeriodLabel(view: CalendarView, date: string): string {
  const [year, month] = date.split("-").map(Number)
  if (view === "month") {
    return new Date(Date.UTC(year, month - 1, 1)).toLocaleDateString("en-US", {
      month: "long",
      year: "numeric",
    })
  }
  if (view === "week") {
    const d = new Date(date + "T00:00:00Z")
    const dow = (d.getUTCDay() + 6) % 7
    const mon = new Date(d)
    mon.setUTCDate(d.getUTCDate() - dow)
    const sun = new Date(mon)
    sun.setUTCDate(mon.getUTCDate() + 6)
    const fmt = (dt: Date) =>
      dt.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" })
    return `${fmt(mon)} – ${fmt(sun)}`
  }
  return new Date(date + "T00:00:00Z").toLocaleDateString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
    timeZone: "UTC",
  })
}

function shiftDate(view: CalendarView, date: string, delta: number): string {
  const [year, month, day] = date.split("-").map(Number)
  if (view === "month") {
    const d = new Date(Date.UTC(year, month - 1 + delta, 1))
    return d.toISOString().split("T")[0]
  }
  if (view === "week") {
    const d = new Date(date + "T00:00:00Z")
    d.setUTCDate(d.getUTCDate() + delta * 7)
    return d.toISOString().split("T")[0]
  }
  // day
  const d = new Date(Date.UTC(year, month - 1, day + delta))
  return d.toISOString().split("T")[0]
}

const VIEW_LABELS: { value: CalendarView; label: string }[] = [
  { value: "month", label: "Month" },
  { value: "week", label: "Week" },
  { value: "day", label: "Day" },
]

export function CalendarPage() {
  const { date: dateParam, view: viewParam } = useSearch({ from: "/" })
  const navigate = useNavigate({ from: "/" })

  const today = todayDate()
  const selectedDate = dateParam ?? today
  const view: CalendarView = viewParam ?? "month"

  function update(patch: { date?: string; view?: CalendarView }) {
    navigate({ search: (prev) => ({ ...prev, ...patch }) })
  }

  function handleViewChange(v: CalendarView) {
    update({ view: v })
  }

  function handlePrev() {
    update({ date: shiftDate(view, selectedDate, -1) })
  }

  function handleNext() {
    update({ date: shiftDate(view, selectedDate, 1) })
  }

  function handleToday() {
    update({ date: today })
  }

  function handleSelectDate(date: string) {
    update({ date, view: "day" })
  }

  function handleDateChange(date: string) {
    update({ date })
  }

  // Keyboard navigation
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement).tagName
      if (tag === "INPUT" || tag === "TEXTAREA") return
      if (e.key === "ArrowLeft") handlePrev()
      else if (e.key === "ArrowRight") handleNext()
      else if (e.key === "t" || e.key === "T") handleToday()
      else if (e.key === "m" || e.key === "M") handleViewChange("month")
      else if (e.key === "w" || e.key === "W") handleViewChange("week")
      else if (e.key === "d" || e.key === "D") handleViewChange("day")
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  })

  const { data: athlete, isLoading: loadingAthlete, error: athleteError } = useQuery({
    queryKey: ["athlete"],
    queryFn: fetchCurrentAthlete,
  })

  if (loadingAthlete) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (athleteError || !athlete) {
    return (
      <div className="flex h-full items-center justify-center p-4">
        <p className="text-sm text-destructive">
          Could not load athlete. Is the backend running?
        </p>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Header bar */}
      <div className="flex shrink-0 items-center gap-2 border-b px-4 py-3">
        {/* Period navigation */}
        <Button variant="ghost" size="icon" onClick={handlePrev} aria-label="Previous period">
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" onClick={handleNext} aria-label="Next period">
          <ChevronRight className="h-4 w-4" />
        </Button>

        <span className="min-w-[160px] text-sm font-medium">
          {formatPeriodLabel(view, selectedDate)}
        </span>

        <Button
          variant="outline"
          size="sm"
          onClick={handleToday}
          className="ml-1 h-7 px-2 text-xs"
        >
          Today
        </Button>

        {/* Date picker */}
        <input
          type="date"
          value={selectedDate}
          onChange={(e) => e.target.value && update({ date: e.target.value })}
          className="ml-1 h-7 rounded-md border border-input bg-background px-2 text-xs focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          aria-label="Pick date"
        />

        {/* View toggle */}
        <div className="ml-auto flex rounded-md border overflow-hidden">
          {VIEW_LABELS.map(({ value, label }) => (
            <button
              key={value}
              onClick={() => handleViewChange(value)}
              className={cn(
                "px-3 py-1 text-xs font-medium transition-colors",
                view === value
                  ? "bg-primary text-primary-foreground"
                  : "bg-background hover:bg-accent text-foreground"
              )}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* View content */}
      <div className="flex-1 overflow-auto p-4">
        {view === "month" && (
          <MonthView
            athleteId={athlete.athlete_id}
            date={selectedDate}
            selectedDate={selectedDate}
            onSelectDate={handleSelectDate}
          />
        )}
        {view === "week" && (
          <WeekView
            athleteId={athlete.athlete_id}
            date={selectedDate}
            selectedDate={selectedDate}
            onSelectDate={handleSelectDate}
          />
        )}
        {view === "day" && (
          <DayDetailPanel
            athleteId={athlete.athlete_id}
            selectedDate={selectedDate}
            onDateChange={handleDateChange}
          />
        )}
      </div>

      {/* Keyboard hint */}
      {view !== "day" && (
        <div className="shrink-0 border-t px-4 py-1.5 text-xs text-muted-foreground flex items-center gap-2">
          <CalendarDays className="h-3 w-3" />
          <span>← → navigate · T today · M month · W week · D day</span>
        </div>
      )}
    </div>
  )
}
