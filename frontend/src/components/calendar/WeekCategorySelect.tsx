import { cn } from "../../lib/utils"
import type { WeekCategory } from "../../api/weekCategory"

export const WEEK_CATEGORY_LABELS: Record<WeekCategory, string> = {
  build: "Build",
  maintain: "Maintain",
  taper: "Taper",
  race: "Race",
  restitution: "Restitution",
  recovering: "Recovering",
}

export const WEEK_CATEGORY_STYLES: Record<WeekCategory, string> = {
  build: "border-blue-200 bg-blue-50 text-blue-800",
  maintain: "border-slate-200 bg-slate-50 text-slate-700",
  taper: "border-teal-200 bg-teal-50 text-teal-800",
  race: "border-red-200 bg-red-50 text-red-800",
  restitution: "border-emerald-200 bg-emerald-50 text-emerald-800",
  recovering: "border-amber-200 bg-amber-50 text-amber-800",
}

const WEEK_CATEGORY_ORDER: WeekCategory[] = [
  "build",
  "maintain",
  "taper",
  "race",
  "restitution",
  "recovering",
]

interface WeekCategorySelectProps {
  value: WeekCategory | null
  weekStart: string
  onChange?: (category: WeekCategory) => void
}

export function WeekCategorySelect({ value, weekStart, onChange }: WeekCategorySelectProps) {
  if (!onChange) {
    return (
      <div
        className="flex h-full min-h-[80px] items-center justify-center rounded-md border border-dashed border-border px-1 py-1.5"
        title={`Week of ${weekStart}`}
      >
        {value ? (
          <span
            className={cn(
              "rounded-full border px-2 py-0.5 text-center text-[11px] font-medium leading-none",
              WEEK_CATEGORY_STYLES[value]
            )}
          >
            {WEEK_CATEGORY_LABELS[value]}
          </span>
        ) : (
          <span className="text-[11px] text-muted-foreground">—</span>
        )}
      </div>
    )
  }

  return (
    <div
      className="flex h-full min-h-[80px] items-center justify-center rounded-md border border-dashed border-border px-1 py-1.5"
      title={`Week of ${weekStart}`}
    >
      <select
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value as WeekCategory)}
        className={cn(
          "w-full rounded-full border bg-background px-1.5 py-0.5 text-center text-[11px] font-medium leading-none",
          "focus:outline-none focus:ring-1 focus:ring-primary",
          value ? WEEK_CATEGORY_STYLES[value] : "border-border text-muted-foreground"
        )}
      >
        <option value="" disabled>
          Set…
        </option>
        {WEEK_CATEGORY_ORDER.map((c) => (
          <option key={c} value={c}>
            {WEEK_CATEGORY_LABELS[c]}
          </option>
        ))}
      </select>
    </div>
  )
}
