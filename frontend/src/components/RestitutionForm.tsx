import { useEffect, useRef, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"
import { Input } from "./ui/input"
import { Label } from "./ui/label"
import { Textarea } from "./ui/textarea"
import { cn } from "../lib/utils"
import type { Restitution } from "../api/dailyEntry"

function qualityLabel(v: number): string {
  if (v < 1.5) return "Very Poor"
  if (v < 2.5) return "Poor"
  if (v < 3.5) return "Fair"
  if (v < 4.5) return "Good"
  return "Excellent"
}

function qualityColor(v: number): string {
  if (v < 1.5) return "bg-red-100 text-red-800"
  if (v < 2.5) return "bg-orange-100 text-orange-800"
  if (v < 3.5) return "bg-yellow-100 text-yellow-800"
  if (v < 4.5) return "bg-green-100 text-green-800"
  return "bg-emerald-100 text-emerald-800"
}

interface SubjectiveSliderProps {
  id: string
  label: string
  lowLabel: string
  highLabel: string
  value: number | undefined
  onChange: (v: number | undefined) => void
}

function SubjectiveSlider({ id, label, lowLabel, highLabel, value, onChange }: SubjectiveSliderProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label htmlFor={id}>{label}</Label>
        {value != null ? (
          <span className={cn("rounded-full px-2.5 py-0.5 text-sm font-semibold", qualityColor(value))}>
            {qualityLabel(value)}
          </span>
        ) : (
          <span className="rounded-full bg-muted px-2.5 py-0.5 text-sm font-semibold text-muted-foreground">
            Not set
          </span>
        )}
      </div>
      <input
        id={id}
        type="range"
        min={1}
        max={5}
        step={0.01}
        value={value ?? 3}
        onChange={(e) => onChange(Number(e.target.value))}
        className={cn("w-full accent-primary", value == null && "opacity-40")}
      />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{lowLabel}</span>
        <span>{highLabel}</span>
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
        <div className="col-span-2 grid grid-cols-2 gap-4 sm:col-span-3">
          <SubjectiveSlider
            id="sleep-quality"
            label="Sleep Quality"
            lowLabel="Poor"
            highLabel="Excellent"
            value={value.sleep_quality}
            onChange={(v) => onChange({ ...value, sleep_quality: v })}
          />
          <SubjectiveSlider
            id="readiness"
            label="Readiness"
            lowLabel="Exhausted"
            highLabel="Excellent"
            value={value.readiness}
            onChange={(v) => onChange({ ...value, readiness: v })}
          />
        </div>
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
