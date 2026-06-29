import { useState } from "react"
import { useParams, Link } from "@tanstack/react-router"
import { useQuery } from "@tanstack/react-query"
import { ArrowLeft, ChevronLeft, ChevronRight } from "lucide-react"
import { MonthView } from "../components/calendar/MonthView"
import { ReadOnlyDayPanel } from "../components/calendar/ReadOnlyDayPanel"
import { Button } from "../components/ui/button"
import { fetchCoachAthleteFeed, fetchCoachRoster } from "../api/coach"
import type { RiskCounts } from "../api/coach"
import { recoveryQualityTextStyle, severityBadgeStyle, severityFor } from "../lib/risk"
import { cn, localToday } from "../lib/utils"

function RiskBadge({ label, counts }: { label: string; counts: RiskCounts }) {
  const total = counts.low + counts.moderate + counts.high
  if (total === 0) {
    return <span className="text-xs text-muted-foreground">{label}: none</span>
  }
  const severity = severityFor(counts)
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs", severityBadgeStyle[severity])}>
      {label}: {counts.high > 0 && `${counts.high} high`}
      {counts.high > 0 && (counts.moderate > 0 || counts.low > 0) && ", "}
      {counts.moderate > 0 && `${counts.moderate} moderate`}
      {counts.moderate > 0 && counts.low > 0 && ", "}
      {counts.low > 0 && `${counts.low} low`}
    </span>
  )
}

function shiftMonth(date: string, delta: number): string {
  const [year, month] = date.split("-").map(Number)
  return new Date(Date.UTC(year, month - 1 + delta, 1)).toISOString().split("T")[0]
}

export function CoachAthleteDetailPage() {
  const { athleteId } = useParams({ from: "/coach/athletes/$athleteId" })
  const id = Number(athleteId)
  const today = localToday()
  const [monthDate, setMonthDate] = useState(today)
  const [selectedDate, setSelectedDate] = useState<string | null>(null)

  const { data: roster } = useQuery({
    queryKey: ["coach-roster"],
    queryFn: () => fetchCoachRoster(),
  })
  const athlete = roster?.find((a) => a.athlete_id === id)

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b px-4 py-3">
        <Link to="/coach">
          <Button variant="ghost" size="sm" className="gap-1.5 text-muted-foreground hover:text-foreground">
            <ArrowLeft className="h-4 w-4" />
            Athletes
          </Button>
        </Link>
        <span className="text-sm font-medium ml-2">Read-only view</span>
        {athlete && (
          <div className="flex items-center gap-2">
            <RiskBadge label="Performance" counts={athlete.performance_risk} />
            <RiskBadge label="Restitution" counts={athlete.restitution_risk} />
            {athlete.overall_recovery_quality && (
              <span className={cn("text-xs", recoveryQualityTextStyle[athlete.overall_recovery_quality])}>
                {athlete.overall_recovery_quality}
              </span>
            )}
          </div>
        )}
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
