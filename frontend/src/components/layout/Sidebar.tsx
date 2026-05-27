import { Link, useRouterState } from "@tanstack/react-router"
import { CalendarDays, TrendingUp, User } from "lucide-react"
import { useQuery } from "@tanstack/react-query"
import { fetchCurrentAthlete } from "../../api/auth"
import { cn } from "../../lib/utils"

const NAV_ITEMS = [
  { to: "/", label: "Calendar", icon: CalendarDays, exact: true },
  { to: "/insights", label: "Insights", icon: TrendingUp, exact: false },
] as const

export function Sidebar() {
  const { location } = useRouterState()
  const { data: athlete } = useQuery({
    queryKey: ["athlete"],
    queryFn: fetchCurrentAthlete,
  })

  const profileActive = location.pathname.startsWith("/settings")

  return (
    <aside className="hidden sm:flex h-full w-52 shrink-0 flex-col border-r bg-background">
      <div className="px-4 py-5">
        <span className="text-base font-semibold tracking-tight">Sidekick</span>
      </div>

      <nav className="flex flex-col gap-1 px-3 flex-1">
        {NAV_ITEMS.map(({ to, label, icon: Icon, exact }) => {
          const active = exact
            ? location.pathname === to
            : location.pathname.startsWith(to)
          return (
            <Link
              key={to}
              to={to}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2.5 text-sm font-medium transition-colors",
                "hover:bg-accent hover:text-accent-foreground",
                active && "bg-accent text-accent-foreground"
              )}
            >
              <Icon className="h-4 w-4 shrink-0" />
              {label}
            </Link>
          )
        })}
      </nav>

      <div className="px-3 pb-4">
        <Link
          to="/settings"
          className={cn(
            "flex items-center gap-3 rounded-md px-3 py-2.5 text-sm font-medium transition-colors",
            "hover:bg-accent hover:text-accent-foreground",
            profileActive && "bg-accent text-accent-foreground"
          )}
        >
          {athlete?.profile_picture ? (
            <img
              src={athlete.profile_picture}
              alt={`${athlete.firstname} ${athlete.lastname}`}
              className="h-5 w-5 shrink-0 rounded-full object-cover"
            />
          ) : (
            <User className="h-4 w-4 shrink-0" />
          )}
          Profile
        </Link>
      </div>
    </aside>
  )
}
