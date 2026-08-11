import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Loader2 } from "lucide-react"
import { fetchFeed, type FeedDay } from "../../api/feed"
import {
  fetchWeekCategories,
  type WeekCategory,
  type WeekCategoryEntry,
} from "../../api/weekCategory"
import { CalendarDayCell } from "./CalendarDayCell"
import { WeekCategorySelect } from "./WeekCategorySelect"
import { cn, localToday } from "../../lib/utils"

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

// Splits the flat cell grid (always a multiple of 7, Monday-first) into per-week rows
// so each row can carry its own leading week-category cell.
function chunkIntoWeeks<T>(cells: T[]): T[][] {
  const weeks: T[][] = []
  for (let i = 0; i < cells.length; i += 7) {
    weeks.push(cells.slice(i, i + 7))
  }
  return weeks
}

interface MonthViewProps {
  athleteId: number
  date: string
  selectedDate: string
  onSelectDate: (date: string) => void
  fetchFeedFn?: (athleteId: number, start: string, end: string) => Promise<FeedDay[]>
  fetchWeekCategoryFn?: (
    athleteId: number,
    start: string,
    end: string
  ) => Promise<WeekCategoryEntry[]>
  onSetWeekCategory?: (
    athleteId: number,
    weekStart: string,
    category: WeekCategory
  ) => Promise<WeekCategoryEntry>
}

export function MonthView({
  athleteId,
  date,
  selectedDate,
  onSelectDate,
  fetchFeedFn = fetchFeed,
  fetchWeekCategoryFn = fetchWeekCategories,
  onSetWeekCategory,
}: MonthViewProps) {
  const [year, month] = date.split("-").map(Number)
  const { start, end } = getMonthRange(date)
  const today = localToday()
  const queryClient = useQueryClient()

  const { data: feed, isLoading } = useQuery({
    queryKey: ["feed", athleteId, start, end],
    queryFn: () => fetchFeedFn(athleteId, start, end),
  })

  const weekCategoryQueryKey = ["week-category", athleteId, start, end] as const
  const { data: weekCategories } = useQuery({
    queryKey: weekCategoryQueryKey,
    queryFn: () => fetchWeekCategoryFn(athleteId, start, end),
  })

  const setCategoryMutation = useMutation({
    mutationFn: ({ weekStart, category }: { weekStart: string; category: WeekCategory }) => {
      if (!onSetWeekCategory) throw new Error("Week category is read-only in this view")
      return onSetWeekCategory(athleteId, weekStart, category)
    },
    onSuccess: (updated) => {
      queryClient.setQueryData<WeekCategoryEntry[]>(weekCategoryQueryKey, (old) => {
        const others = (old ?? []).filter((e) => e.week_start !== updated.week_start)
        return [...others, updated].sort((a, b) => a.week_start.localeCompare(b.week_start))
      })
    },
  })

  const categoryByWeek = new Map<string, WeekCategory>()
  for (const entry of weekCategories ?? []) {
    categoryByWeek.set(entry.week_start, entry.category)
  }

  const feedMap = new Map<string, FeedDay>()
  for (const day of feed ?? []) {
    feedMap.set(day.date, day)
  }

  const cells = buildCalendarGrid(year, month, feedMap)
  const weeks = chunkIntoWeeks(cells)

  return (
    <div className="relative">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/60 z-10">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      )}
      <div className="grid grid-cols-8 gap-1">
        <div />
        {WEEKDAYS.map((d) => (
          <div key={d} className="py-1 text-center text-xs font-medium text-muted-foreground">
            {d}
          </div>
        ))}
        {weeks.map((week) => {
          const weekStart = week[0].date
          return (
            <div key={weekStart} className="contents">
              <WeekCategorySelect
                weekStart={weekStart}
                value={categoryByWeek.get(weekStart) ?? null}
                onChange={
                  onSetWeekCategory
                    ? (category) => setCategoryMutation.mutate({ weekStart, category })
                    : undefined
                }
              />
              {week.map(({ date: cellDate, inMonth, feedDay }) => (
                <div key={cellDate} className={cn("h-full", inMonth ? "" : "opacity-30")}>
                  <CalendarDayCell
                    day={feedDay}
                    date={cellDate}
                    isToday={cellDate === today}
                    isSelected={cellDate === selectedDate}
                    onClick={() => onSelectDate(cellDate)}
                  />
                </div>
              ))}
            </div>
          )
        })}
      </div>
    </div>
  )
}
