import { cn } from "../../lib/utils"
import type { WeekCategory } from "../../api/weekCategory"
import {
  WEEK_CATEGORY_BADGE_STYLES,
  WEEK_CATEGORY_EMOJI,
  WEEK_CATEGORY_LABELS,
} from "./WeekCategorySelect"

interface WeekCategoryBadgeProps {
  category: WeekCategory
}

export function WeekCategoryBadge({ category }: WeekCategoryBadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-3 py-1",
        WEEK_CATEGORY_BADGE_STYLES[category]
      )}
    >
      {WEEK_CATEGORY_EMOJI[category]} Week:{" "}
      <span className="font-medium">{WEEK_CATEGORY_LABELS[category]}</span>
    </span>
  )
}
