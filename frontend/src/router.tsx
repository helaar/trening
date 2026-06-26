import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import {
  createRootRoute,
  createRoute,
  createRouter,
} from "@tanstack/react-router"
import { AppLayout } from "./components/layout/AppLayout"
import { CalendarPage } from "./pages/CalendarPage"
import { InsightsPage } from "./pages/InsightsPage"
import { AthleteConfig } from "./pages/AthleteConfig"
import { SetupPage } from "./pages/SetupPage"
import { InspectPage } from "./pages/InspectPage"
import { CoachOverviewPage } from "./pages/CoachOverviewPage"
import { CoachAthleteDetailPage } from "./pages/CoachAthleteDetailPage"

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 60_000,
    },
  },
})

const rootRoute = createRootRoute({
  component: () => (
    <QueryClientProvider client={queryClient}>
      <AppLayout />
    </QueryClientProvider>
  ),
})

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: CalendarPage,
  validateSearch: (search: Record<string, unknown>) => ({
    date: typeof search.date === "string" ? search.date : undefined,
    view: (search.view === "month" || search.view === "week" || search.view === "day")
      ? (search.view as "month" | "week" | "day")
      : undefined,
    from: (search.from === "month" || search.from === "week")
      ? (search.from as "month" | "week")
      : undefined,
  }),
})

const insightsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/insights",
  component: InsightsPage,
})

const settingsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/settings",
  component: AthleteConfig,
})

const setupRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/setup",
  component: SetupPage,
})

const inspectRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/admin/inspect",
  component: InspectPage,
})

const coachOverviewRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/coach",
  component: CoachOverviewPage,
})

const coachAthleteRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/coach/athletes/$athleteId",
  component: CoachAthleteDetailPage,
})

const routeTree = rootRoute.addChildren([
  indexRoute,
  insightsRoute,
  settingsRoute,
  setupRoute,
  inspectRoute,
  coachOverviewRoute,
  coachAthleteRoute,
])

export const router = createRouter({ routeTree })

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router
  }
}
