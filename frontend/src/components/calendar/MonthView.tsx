import { useQuery } from "@tanstack/react-query"
import { Loader2 } from "lucide-react"
import { fetchFeed, type FeedDay } from "../../api/feed"
import { CalendarDayCell } from "./CalendarDayCell"

const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

function getMonthRange(date: string): { start: string; end: string } {
  const [year, month] = date.split("-").map(Number)
  const start = `${year}-${String(month).padStart(2, "0")}-01`
  const lastDay = new Date(year, month, 0).getDate()
  const end = `${year}-${String(month).padStart(2, "0")}-${lastDay}`
  return { start, end }
}

function buildCalendarGrid(
  year: number,
  month: number,
  feedMap: Map<string, FeedDay>
): Array<{ date: string; inMonth: boolean; feedDay: FeedDay | null }> {
  const firstOfMonth = new Date(Date.UTC(year, month - 1, 1))
  // Monday = 0 in our grid; getUTCDay() returns 0=Sun,1=Mon...
  const startDow = (firstOfMonth.getUTCDay() + 6) % 7
  const daysInMonth = new Date(Date.UTC(year, month, 0)).getUTCDate()
  const totalCells = Math.ceil((startDow + daysInMonth) / 7) * 7

  const cells: Array<{ date: string; inMonth: boolean; feedDay: FeedDay | null }> = []
  for (let i = 0; i < totalCells; i++) {
    const d = new Date(Date.UTC(year, month - 1, 1 - startDow + i))
    const dateStr = d.toISOString().split("T")[0]
    const inMonth = d.getUTCMonth() + 1 === month && d.getUTCFullYear() === year
    cells.push({ date: dateStr, inMonth, feedDay: feedMap.get(dateStr) ?? null })
  }
  return cells
}

function maxTssInFeed(feed: FeedDay[]): number {
  let max = 0
  for (const day of feed) {
    const tss = day.workouts.reduce((s, w) => s + (w.metrics?.training_stress_score ?? 0), 0)
    if (tss > max) max = tss
  }
  return Math.max(max, 50)
}

interface MonthViewProps {
  athleteId: number
  date: string
  selectedDate: string
  onSelectDate: (date: string) => void
}

export function MonthView({ athleteId, date, selectedDate, onSelectDate }: MonthViewProps) {
  const [year, month] = date.split("-").map(Number)
  const { start, end } = getMonthRange(date)
  const today = new Date().toISOString().split("T")[0]

  const { data: feed, isLoading } = useQuery({
    queryKey: ["feed", athleteId, start, end],
    queryFn: () => fetchFeed(athleteId, start, end),
  })

  const feedMap = new Map<string, FeedDay>()
  for (const day of feed ?? []) {
    feedMap.set(day.date, day)
  }

  const cells = buildCalendarGrid(year, month, feedMap)
  const maxTss = maxTssInFeed(feed ?? [])

  return (
    <div className="relative">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/60 z-10">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      )}
      <div className="grid grid-cols-7 gap-1">
        {WEEKDAYS.map((d) => (
          <div key={d} className="py-1 text-center text-xs font-medium text-muted-foreground">
            {d}
          </div>
        ))}
        {cells.map(({ date: cellDate, inMonth, feedDay }) => (
          <div key={cellDate} className={inMonth ? "" : "opacity-30"}>
            <CalendarDayCell
              day={feedDay}
              date={cellDate}
              isToday={cellDate === today}
              isSelected={cellDate === selectedDate}
              compact={true}
              maxTss={maxTss}
              onClick={() => onSelectDate(cellDate)}
            />
          </div>
        ))}
      </div>
    </div>
  )
}
