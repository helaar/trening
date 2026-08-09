import { useQuery } from "@tanstack/react-query"
import { Loader2 } from "lucide-react"
import { fetchCurrentAthlete } from "../api/auth"
import { fetchFitnessSeries } from "../api/fitness"
import { FitnessChart } from "../components/FitnessChart"

export function InsightsPage() {
  const { data: athlete, isLoading: loadingAthlete, error: athleteError } = useQuery({
    queryKey: ["athlete"],
    queryFn: fetchCurrentAthlete,
  })

  const {
    data: fitness,
    isLoading: loadingFitness,
    error: fitnessError,
  } = useQuery({
    queryKey: ["fitness", athlete?.athlete_id],
    queryFn: () => fetchFitnessSeries(athlete!.athlete_id),
    enabled: !!athlete,
  })

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-6">
      <h1 className="text-2xl font-bold">Insights</h1>

      {loadingAthlete || loadingFitness ? (
        <div className="flex items-center justify-center py-10">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : athleteError || fitnessError ? (
        <p className="text-sm text-destructive">Failed to load fitness data.</p>
      ) : (
        <FitnessChart data={fitness ?? []} />
      )}

      <p className="text-sm text-muted-foreground">
        AI coaching feedback will appear here.
      </p>
    </div>
  )
}
