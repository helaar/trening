import { useInfiniteQuery, useQuery } from "@tanstack/react-query"
import { Link } from "@tanstack/react-router"
import { Loader2 } from "lucide-react"
import { fetchCurrentAthlete } from "../api/auth"
import { fetchFeed } from "../api/feed"
import { FeedDayCard } from "../components/FeedDayCard"
import { Button } from "../components/ui/button"

const WINDOW_DAYS = 14

function todayDate(): string {
  return new Date().toISOString().split("T")[0]
}

function subtractDays(dateStr: string, days: number): string {
  const d = new Date(dateStr + "T00:00:00Z")
  d.setUTCDate(d.getUTCDate() - days)
  return d.toISOString().split("T")[0]
}

export function TrainingFeed() {
  const { data: athlete } = useQuery({
    queryKey: ["athlete"],
    queryFn: fetchCurrentAthlete,
  })

  const athleteId = athlete?.athlete_id

  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
    isError,
  } = useInfiniteQuery({
    queryKey: ["feed", athleteId],
    enabled: !!athleteId,
    staleTime: 0,
    initialPageParam: todayDate(),
    queryFn: ({ pageParam }) => {
      const end = pageParam as string
      const start = subtractDays(end, WINDOW_DAYS - 1)
      return fetchFeed(athleteId!, start, end)
    },
    getNextPageParam: (_lastPage, _allPages, lastPageParam) =>
      subtractDays(lastPageParam as string, WINDOW_DAYS),
  })

  const allDays = data?.pages.flat() ?? []

  return (
    <div className="max-w-2xl mx-auto px-4 py-6 space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Training Feed</h1>
        <Link to="/">
          <Button variant="ghost" size="sm">
            Today
          </Button>
        </Link>
      </div>

      {isLoading && (
        <div className="space-y-2">
          {[1, 2, 3].map((n) => (
            <div key={n} className="h-12 rounded-lg border bg-muted animate-pulse" />
          ))}
        </div>
      )}

      {isError && (
        <p className="text-sm text-destructive">Failed to load feed. Please try again.</p>
      )}

      {allDays.length === 0 && !isLoading && !isError && (
        <p className="text-sm text-muted-foreground">No training data found.</p>
      )}

      <div className="space-y-2">
        {allDays.map((day) => (
          <FeedDayCard key={day.date} day={day} />
        ))}
      </div>

      {hasNextPage && (
        <div className="flex justify-center pt-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => fetchNextPage()}
            disabled={isFetchingNextPage}
          >
            {isFetchingNextPage ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Loading…
              </>
            ) : (
              "Load more"
            )}
          </Button>
        </div>
      )}
    </div>
  )
}
