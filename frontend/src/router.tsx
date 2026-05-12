import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import {
  createRootRoute,
  createRoute,
  createRouter,
  Outlet,
} from "@tanstack/react-router"
import { TodayTraining } from "./pages/TodayTraining"
import { AthleteConfig } from "./pages/AthleteConfig"
import { Plans } from "./pages/Plans"
import { TrainingFeed } from "./pages/TrainingFeed"

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
      <Outlet />
    </QueryClientProvider>
  ),
})

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: TodayTraining,
})

const settingsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/settings",
  component: AthleteConfig,
})

const plansRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/plans",
  component: Plans,
})

const feedRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/feed",
  component: TrainingFeed,
})

const routeTree = rootRoute.addChildren([indexRoute, settingsRoute, plansRoute, feedRoute])

export const router = createRouter({ routeTree })

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router
  }
}
