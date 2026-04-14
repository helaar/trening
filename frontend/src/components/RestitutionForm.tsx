import { useEffect, useRef, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"
import { Input } from "./ui/input"
import { Label } from "./ui/label"
import { Textarea } from "./ui/textarea"
import type { Restitution } from "../api/dailyEntry"

const SLEEP_OPTIONS = [
  { value: 1, emoji: "🌑", label: "Very Poor" },
  { value: 2, emoji: "🌘", label: "Poor" },
  { value: 3, emoji: "🌗", label: "Fair" },
  { value: 4, emoji: "🌔", label: "Good" },
  { value: 5, emoji: "🌕", label: "Excellent" },
]

const READINESS_OPTIONS = [
  { value: 1, emoji: "😴", label: "Exhausted" },
  { value: 2, emoji: "🥱", label: "Tired" },
  { value: 3, emoji: "😐", label: "OK" },
  { value: 4, emoji: "🏃", label: "Good" },
  { value: 5, emoji: "⚡", label: "Excellent" },
]

interface RatingOption {
  value: number
  emoji: string
  label: string
}

interface EmojiRatingProps {
  id: string
  label: string
  options: RatingOption[]
  value: number | undefined
  onChange: (v: number | undefined) => void
}

function EmojiRating({ id, label, options, value, onChange }: EmojiRatingProps) {
  return (
    <div className="col-span-2 space-y-1.5 sm:col-span-3">
      <Label htmlFor={id}>{label}</Label>
      <div id={id} className="grid grid-cols-5 gap-1.5">
        {options.map((opt) => (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(value === opt.value ? undefined : opt.value)}
            className={`flex flex-col items-center rounded-md py-2 transition-colors ${
              value === opt.value
                ? "bg-primary/20 ring-2 ring-primary"
                : "hover:bg-muted"
            }`}
          >
            <span className="text-2xl">{opt.emoji}</span>
            <span className="mt-1 text-xs text-muted-foreground">{opt.label}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

interface Props {
  value: Restitution
  onChange: (value: Restitution) => void
}

function parseSleep(raw: string): number | undefined {
  const s = raw.trim()
  if (!s) return undefined

  // hr:mm format — e.g. "7:30"
  const colonMatch = s.match(/^(\d+):(\d{1,2})$/)
  if (colonMatch) {
    const h = parseInt(colonMatch[1], 10)
    const m = parseInt(colonMatch[2], 10)
    if (m < 60) return Math.round((h + m / 60) * 100) / 100
    return undefined
  }

  // decimal format — e.g. "7.5" or "7,5" (comma as decimal separator)
  const n = parseFloat(s.replace(",", "."))
  return !isNaN(n) && n >= 0 ? n : undefined
}

function formatSleep(hours: number): string {
  const h = Math.floor(hours)
  const m = Math.round((hours - h) * 60)
  return m === 0 ? `${h}` : `${h}:${String(m).padStart(2, "0")}`
}

// Detect browser locale decimal separator ("." or ",")
const decimalSep = Intl.NumberFormat(navigator.language).format(1.1).charAt(1)

export function RestitutionForm({ value, onChange }: Props) {
  // Keep raw text while user is typing; normalise on blur
  const [sleepRaw, setSleepRaw] = useState(
    value.sleep_hours !== undefined ? formatSleep(value.sleep_hours) : ""
  )
  const [sleepInvalid, setSleepInvalid] = useState(false)
  const sleepFocused = useRef(false)

  // Sync raw input when the prop value changes externally (e.g. after DB fetch or date navigation).
  // Never interfere while the user is focused — normalisation on blur handles that.
  useEffect(() => {
    if (sleepFocused.current) return
    if (value.sleep_hours !== undefined) {
      setSleepRaw(formatSleep(value.sleep_hours))
      setSleepInvalid(false)
    } else {
      setSleepRaw("")
      setSleepInvalid(false)
    }
  }, [value.sleep_hours])

  const onSleepChange = (raw: string) => {
    setSleepRaw(raw)
    const parsed = parseSleep(raw)
    setSleepInvalid(raw !== "" && parsed === undefined)
    onChange({ ...value, sleep_hours: parsed })
  }

  const onSleepBlur = () => {
    sleepFocused.current = false
    if (value.sleep_hours !== undefined) {
      setSleepRaw(formatSleep(value.sleep_hours))
      setSleepInvalid(false)
    }
  }

  const set = (field: keyof Restitution, raw: string) => {
    if (field === "comment") {
      onChange({ ...value, comment: raw || undefined })
      return
    }
    const num = raw === "" ? undefined : Number(raw)
    onChange({ ...value, [field]: num })
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Morning Check-in</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <div className="space-y-1.5">
          <Label htmlFor="sleep">Sleep</Label>
          <Input
            id="sleep"
            type="text"
            inputMode="decimal"
            placeholder={`7:30 or 7${decimalSep}5`}
            value={sleepRaw}
            onFocus={() => { sleepFocused.current = true }}
            onChange={(e) => onSleepChange(e.target.value)}
            onBlur={onSleepBlur}
            className={sleepInvalid ? "border-destructive focus-visible:ring-destructive" : undefined}
          />
          {sleepInvalid && (
            <p className="text-xs text-destructive">Use 7:30 or 7{decimalSep}5 format</p>
          )}
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="hrv">HRV</Label>
          <Input
            id="hrv"
            type="number"
            min={0}
            max={300}
            placeholder="38"
            value={value.hrv ?? ""}
            onChange={(e) => set("hrv", e.target.value)}
          />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="rhr">Resting HR</Label>
          <Input
            id="rhr"
            type="number"
            min={20}
            max={200}
            placeholder="50"
            value={value.resting_hr ?? ""}
            onChange={(e) => set("resting_hr", e.target.value)}
          />
        </div>
        <EmojiRating
          id="sleep-quality"
          label="Sleep Quality"
          options={SLEEP_OPTIONS}
          value={value.sleep_quality}
          onChange={(v) => onChange({ ...value, sleep_quality: v })}
        />
        <EmojiRating
          id="readiness"
          label="Readiness"
          options={READINESS_OPTIONS}
          value={value.readiness}
          onChange={(v) => onChange({ ...value, readiness: v })}
        />
        <div className="col-span-2 space-y-1.5 sm:col-span-3">
          <Label htmlFor="comment">Comment</Label>
          <Textarea
            id="comment"
            placeholder="How did you sleep? Any notes on recovery…"
            value={value.comment ?? ""}
            onChange={(e) => set("comment", e.target.value)}
          />
        </div>
      </CardContent>
    </Card>
  )
}
