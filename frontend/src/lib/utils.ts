import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// "Today" in the browser's local timezone (YYYY-MM-DD), not UTC — toISOString()
// would shift the date near local midnight for athletes ahead of UTC.
export function localToday(): string {
  const d = new Date()
  const year = d.getFullYear()
  const month = String(d.getMonth() + 1).padStart(2, "0")
  const day = String(d.getDate()).padStart(2, "0")
  return `${year}-${month}-${day}`
}

// Monday of the ISO week containing the given YYYY-MM-DD date — matches the
// `w=` param Intervals.icu's calendar view expects.
export function mondayOf(dateStr: string): string {
  const d = new Date(dateStr + "T00:00:00Z")
  const day = d.getUTCDay()
  const diff = day === 0 ? -6 : 1 - day
  d.setUTCDate(d.getUTCDate() + diff)
  return d.toISOString().split("T")[0]
}
