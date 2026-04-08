import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { TodayTraining } from "@/pages/TodayTraining"

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 60_000,
    },
  },
})

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TodayTraining />
    </QueryClientProvider>
  )
}
