import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Loader2 } from "lucide-react"
import { useSearch, useNavigate } from "@tanstack/react-router"
import { fetchCurrentAthlete } from "../api/auth"
import { DayDetailPanel } from "../components/calendar/DayDetailPanel"

function todayDate(): string {
  return new Date().toISOString().split("T")[0]
}

export function CalendarPage() {
  const { date: dateParam } = useSearch({ from: "/" })
  const navigate = useNavigate({ from: "/" })

  const [selectedDate, setSelectedDate] = useState<string>(dateParam ?? todayDate())

  function handleDateChange(date: string) {
    setSelectedDate(date)
    navigate({ search: (prev) => ({ ...prev, date }) })
  }

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
    <div className="h-full">
      <DayDetailPanel
        athleteId={athlete.athlete_id}
        selectedDate={selectedDate}
        onDateChange={handleDateChange}
      />
    </div>
  )
}
