import { useState } from "react"
import { useParams, Link } from "@tanstack/react-router"
import { ArrowLeft, ChevronLeft, ChevronRight } from "lucide-react"
import { MonthView } from "../components/calendar/MonthView"
import { ReadOnlyDayPanel } from "../components/calendar/ReadOnlyDayPanel"
import { Button } from "../components/ui/button"
import { fetchCoachAthleteFeed } from "../api/coach"

function todayDate(): string {
  return new Date().toISOString().split("T")[0]
}

function shiftMonth(date: string, delta: number): string {
  const [year, month] = date.split("-").map(Number)
  return new Date(Date.UTC(year, month - 1 + delta, 1)).toISOString().split("T")[0]
}

export function CoachAthleteDetailPage() {
  const { athleteId } = useParams({ from: "/coach/athletes/$athleteId" })
  const id = Number(athleteId)
  const today = todayDate()
  const [monthDate, setMonthDate] = useState(today)
  const [selectedDate, setSelectedDate] = useState<string | null>(null)

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex shrink-0 items-center gap-2 border-b px-4 py-3">
        <Link to="/coach">
          <Button variant="ghost" size="sm" className="gap-1.5 text-muted-foreground hover:text-foreground">
            <ArrowLeft className="h-4 w-4" />
            Roster
          </Button>
        </Link>
        <span className="text-sm font-medium ml-2">Read-only view</span>
        {!selectedDate && (
          <div className="ml-auto flex items-center gap-1">
            <Button variant="ghost" size="icon" onClick={() => setMonthDate((d) => shiftMonth(d, -1))}>
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={() => setMonthDate((d) => shiftMonth(d, 1))}>
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        )}
        {selectedDate && (
          <Button
            variant="ghost"
            size="sm"
            className="ml-auto"
            onClick={() => setSelectedDate(null)}
          >
            Back to month
          </Button>
        )}
      </div>

      <div className="flex-1 overflow-auto p-4">
        {selectedDate ? (
          <ReadOnlyDayPanel athleteId={id} selectedDate={selectedDate} />
        ) : (
          <MonthView
            athleteId={id}
            date={monthDate}
            selectedDate={today}
            onSelectDate={setSelectedDate}
            readOnly
            fetchFeedFn={fetchCoachAthleteFeed}
          />
        )}
      </div>
    </div>
  )
}
