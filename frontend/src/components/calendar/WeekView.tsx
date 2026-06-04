import { useQuery } from "@tanstack/react-query"
import { Loader2 } from "lucide-react"
import { fetchFeed, type FeedDay } from "../../api/feed"
import { CalendarDayCell } from "./CalendarDayCell"

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
}

export function WeekView({ athleteId, date, selectedDate, onSelectDate }: WeekViewProps) {
  const { start, end, dates } = getWeekRange(date)
  const today = new Date().toISOString().split("T")[0]

  const { data: feed, isLoading } = useQuery({
    queryKey: ["feed", athleteId, start, end],
    queryFn: () => fetchFeed(athleteId, start, end),
  })

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
      <div className="grid grid-cols-7 gap-2">
        {dates.map((d, i) => {
          const dayNum = parseInt(d.split("-")[2], 10)
          return (
            <div key={d} className="text-center text-xs font-medium text-muted-foreground pb-1">
              <span>{DAY_LABELS[i]}</span>
              <span className="ml-1 text-foreground">{dayNum}</span>
            </div>
          )
        })}
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
