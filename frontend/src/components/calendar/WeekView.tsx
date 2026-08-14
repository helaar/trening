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
import { WeekSummary } from "./WeekSummary"
import { computeWeekTotals } from "./weekTotals"
import { localToday } from "../../lib/utils"

function getWeekRange(date: string): { start: string; end: string; dates: string[] } {
  const d = new Date(date + "T00:00:00Z")
  const dow = (d.getUTCDay() + 6) % 7 // Mon=0
  const monday = new Date(d)
  monday.setUTCDate(d.getUTCDate() - dow)
  const dates: string[] = []
  for (let i = 0; i < 7; i++) {
    const day = new Date(monday)
    day.setUTCDate(monday.getUTCDate() + i)
    dates.push(day.toISOString().split("T")[0])
  }
  return { start: dates[0], end: dates[6], dates }
}


const DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

interface WeekViewProps {
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

export function WeekView({
  athleteId,
  date,
  selectedDate,
  onSelectDate,
  fetchFeedFn = fetchFeed,
  fetchWeekCategoryFn = fetchWeekCategories,
  onSetWeekCategory,
}: WeekViewProps) {
  const { start, end, dates } = getWeekRange(date)
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

  const currentCategory = weekCategories?.find((e) => e.week_start === start)?.category ?? null

  const feedMap = new Map<string, FeedDay>()
  for (const day of feed ?? []) {
    feedMap.set(day.date, day)
  }

  return (
    <div className="relative">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/60 z-10">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      )}
      <div className="grid grid-cols-8 gap-2">
        <div />
        {dates.map((d, i) => {
          const dayNum = parseInt(d.split("-")[2], 10)
          return (
            <div key={d} className="text-center text-xs font-medium text-muted-foreground pb-1">
              <span>{DAY_LABELS[i]}</span>
              <span className="ml-1 text-foreground">{dayNum}</span>
            </div>
          )
        })}
        <div
          className="flex h-full min-h-[80px] flex-col gap-1 rounded-md border border-dashed border-border px-1.5 py-1.5"
          title={`Week of ${start}`}
        >
          <WeekCategorySelect
            weekStart={start}
            value={currentCategory}
            onChange={
              onSetWeekCategory
                ? (category) => setCategoryMutation.mutate({ weekStart: start, category })
                : undefined
            }
          />
          <WeekSummary
            athleteId={athleteId}
            weekStart={start}
            weekEnd={end}
            totals={computeWeekTotals(dates.map((d) => feedMap.get(d) ?? null))}
          />
        </div>
        {dates.map((d) => (
          <CalendarDayCell
            key={d}
            day={feedMap.get(d) ?? null}
            date={d}
            isToday={d === today}
            isSelected={d === selectedDate}
            onClick={() => onSelectDate(d)}
          />
        ))}
      </div>
    </div>
  )
}
