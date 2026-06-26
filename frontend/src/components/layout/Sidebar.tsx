import { Link, useRouterState } from "@tanstack/react-router"
import { CalendarDays, Search, Settings, TrendingUp, User, Users } from "lucide-react"
import { useQuery } from "@tanstack/react-query"
import { useEffect, useState } from "react"
import { fetchCurrentAthlete } from "../../api/auth"
import { cn } from "../../lib/utils"

const NAV_ITEMS = [
  { to: "/", label: "Calendar", icon: CalendarDays, exact: true },
  { to: "/insights", label: "Insights", icon: TrendingUp, exact: false },
] as const

const COACH_NAV_ITEMS = [
  { to: "/coach", label: "Roster", icon: Users, exact: true },
] as const

type Perspective = "athlete" | "coach"
const PERSPECTIVE_KEY = "sidekick:perspective"

export function Sidebar() {
  const { location } = useRouterState()
  const { data: athlete } = useQuery({
    queryKey: ["athlete"],
    queryFn: fetchCurrentAthlete,
  })

  const [perspective, setPerspective] = useState<Perspective>(
    () => (localStorage.getItem(PERSPECTIVE_KEY) as Perspective | null) ?? "athlete"
  )

  useEffect(() => {
    localStorage.setItem(PERSPECTIVE_KEY, perspective)
  }, [perspective])

  const profileActive = location.pathname.startsWith("/settings")
  const setupActive = location.pathname.startsWith("/setup")
  const inspectActive = location.pathname.startsWith("/admin/inspect")

  const navItems = perspective === "coach" ? COACH_NAV_ITEMS : NAV_ITEMS

  return (
    <aside className="hidden sm:flex h-full w-52 shrink-0 flex-col border-r bg-background">
      <div className="px-4 py-5">
        <span className="text-base font-semibold tracking-tight">Sidekick</span>
      </div>

      {athlete?.is_coach && (
        <div className="mx-3 mb-2 flex rounded-md border overflow-hidden">
          {(["athlete", "coach"] as const).map((p) => (
            <button
              key={p}
              onClick={() => setPerspective(p)}
              className={cn(
                "flex-1 px-2 py-1.5 text-xs font-medium capitalize transition-colors",
                perspective === p
                  ? "bg-primary text-primary-foreground"
                  : "bg-background hover:bg-accent text-foreground"
              )}
            >
              {p}
            </button>
          ))}
        </div>
      )}

      <nav className="flex flex-col gap-1 px-3 flex-1">
        {navItems.map(({ to, label, icon: Icon, exact }) => {
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

      <div className="px-3 pb-4 space-y-1">
        <Link
          to="/setup"
          className={cn(
            "flex items-center gap-3 rounded-md px-3 py-2.5 text-sm font-medium transition-colors",
            "hover:bg-accent hover:text-accent-foreground",
            setupActive && "bg-accent text-accent-foreground"
          )}
        >
          <Settings className="h-4 w-4 shrink-0" />
          Setup
        </Link>
        <Link
          to="/admin/inspect"
          className={cn(
            "flex items-center gap-3 rounded-md px-3 py-2.5 text-sm font-medium transition-colors",
            "hover:bg-accent hover:text-accent-foreground",
            inspectActive && "bg-accent text-accent-foreground"
          )}
        >
          <Search className="h-4 w-4 shrink-0" />
          Inspect
        </Link>
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
