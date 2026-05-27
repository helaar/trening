import { useState, useEffect } from "react"
import { Card, CardHeader, CardTitle, CardContent } from "./ui/card"
import { Label } from "./ui/label"
import { Textarea } from "./ui/textarea"
import { X } from "lucide-react"
import { cn } from "../lib/utils"
import type { WorkoutAnalysis } from "../api/workouts"

interface Props {
  workout: WorkoutAnalysis
  value: { rpe?: number; notes?: string; tags?: string[] }
  onChange: (value: { rpe?: number; notes?: string; tags?: string[] }) => void
  onSaveNote?: (text: string) => void
}

function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

function rpeColor(rpe: number): string {
  if (rpe <= 3) return "bg-green-100 text-green-800"
  if (rpe <= 5) return "bg-yellow-100 text-yellow-800"
  if (rpe <= 7) return "bg-orange-100 text-orange-800"
  return "bg-red-100 text-red-800"
}

function sportLabel(category: string): string {
  const labels: Record<string, string> = {
    cycling: "Cycling",
    running: "Running",
    skiing: "Skiing",
    strength: "Strength",
    other: "Other",
  }
  return labels[category] ?? category
}

export function ActivityCard({ workout, value, onChange, onSaveNote }: Props) {
  const { session, metrics } = workout
  const isNote = session.manual === true
  const isCommute = session.commute !== "no"

  // Note editing state — declared before any early return to satisfy hook rules
  const [editingNote, setEditingNote] = useState(false)
  const [noteDraft, setNoteDraft] = useState(session.name ?? "")

  // Keep draft in sync when parent updates the note text after a successful save
  useEffect(() => {
    setNoteDraft(session.name ?? "")
  }, [session.name])

  if (isNote) {
    function commitEdit() {
      const trimmed = noteDraft.trim()
      if (trimmed && trimmed !== session.name) onSaveNote?.(trimmed)
      setEditingNote(false)
    }

    return (
      <Card>
        <CardHeader className="pb-3">
          <Label>Note</Label>
          {editingNote ? (
            <textarea
              autoFocus
              rows={3}
              value={noteDraft}
              onChange={(e) => setNoteDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Escape") { setNoteDraft(session.name ?? ""); setEditingNote(false) }
                if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) commitEdit()
              }}
              onBlur={commitEdit}
              className="w-full text-sm bg-transparent outline-none border border-input rounded-md px-2 py-1 resize-none focus:border-ring"
            />
          ) : (
            <p
              className="text-sm cursor-pointer hover:text-foreground/70 transition-colors whitespace-pre-wrap"
              onClick={() => setEditingNote(true)}
              title="Click to edit"
            >
              {session.name}
            </p>
          )}
        </CardHeader>
      </Card>
    )
  }

  const tags = value.tags ?? []

  function addTag(raw: string) {
    const trimmed = raw.trim().replace(/,+$/, "").trim()
    if (trimmed && !tags.includes(trimmed)) {
      onChange({ ...value, tags: [...tags, trimmed] })
    }
  }

  function removeTag(tag: string) {
    onChange({ ...value, tags: tags.filter((t) => t !== tag) })
  }

  function handleTagKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    const input = e.currentTarget
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault()
      addTag(input.value)
      input.value = ""
    } else if (e.key === "Backspace" && input.value === "" && tags.length > 0) {
      onChange({ ...value, tags: tags.slice(0, -1) })
    }
  }

  return (
    <Card className={isCommute ? "opacity-60" : undefined}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="text-base">{session.name ?? sportLabel(session.category)}</CardTitle>
          <div className="flex shrink-0 gap-1.5">
            {isCommute && (
              <span className="rounded-full bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
                Commute
              </span>
            )}
            <span className="rounded-full bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground">
              {sportLabel(session.category)}
            </span>
          </div>
        </div>
        <div className="flex flex-wrap gap-3 text-sm text-muted-foreground">
          <span>{formatDuration(session.duration_sec)}</span>
          {session.distance_km > 0 && <span>{session.distance_km.toFixed(1)} km</span>}
          {metrics.heart_rate?.mean && (
            <span>{Math.round(metrics.heart_rate.mean)} bpm avg</span>
          )}
          {metrics.training_stress_score && (
            <span>TSS {Math.round(metrics.training_stress_score)}</span>
          )}
          {metrics.normalized_power && <span>NP {Math.round(metrics.normalized_power)}W</span>}
        </div>
      </CardHeader>

      {!isCommute && (
        <CardContent className="space-y-4 border-t pt-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>RPE</Label>
              {value.rpe !== undefined ? (
                <span
                  className={cn(
                    "rounded-full px-2.5 py-0.5 text-sm font-semibold",
                    rpeColor(value.rpe)
                  )}
                >
                  {value.rpe} / 10
                </span>
              ) : (
                <span className="rounded-full bg-muted px-2.5 py-0.5 text-sm font-semibold text-muted-foreground">
                  Not set
                </span>
              )}
            </div>
            <input
              type="range"
              min={1}
              max={10}
              step={1}
              value={value.rpe ?? 5}
              onChange={(e) => onChange({ ...value, rpe: Number(e.target.value) })}
              className={cn("w-full accent-primary", value.rpe === undefined && "opacity-40")}
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Very easy</span>
              <span>Moderate</span>
              <span>Max effort</span>
            </div>
          </div>

          <div className="space-y-1.5">
            <Label>Notes</Label>
            <Textarea
              placeholder="How did it feel? Any observations…"
              value={value.notes ?? ""}
              onChange={(e) =>
                onChange({ ...value, notes: e.target.value || undefined })
              }
            />
          </div>

          <div className="space-y-1.5">
            <Label>Tags</Label>
            <div className="flex min-h-9 flex-wrap gap-1.5 rounded-md border border-input bg-background px-3 py-2 text-sm focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2">
              {tags.map((tag) => (
                <span
                  key={tag}
                  className="flex items-center gap-1 rounded-full bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground"
                >
                  {tag}
                  <button
                    type="button"
                    onClick={() => removeTag(tag)}
                    className="text-muted-foreground hover:text-foreground"
                    aria-label={`Remove ${tag}`}
                  >
                    <X className="h-3 w-3" />
                  </button>
                </span>
              ))}
              <input
                type="text"
                onKeyDown={handleTagKeyDown}
                onBlur={(e) => {
                  if (e.target.value) {
                    addTag(e.target.value)
                    e.target.value = ""
                  }
                }}
                placeholder={tags.length === 0 ? "race, indoor, long…" : ""}
                className="min-w-24 flex-1 bg-transparent outline-none placeholder:text-muted-foreground"
              />
            </div>
            <p className="text-xs text-muted-foreground">Press Enter or comma to add a tag</p>
          </div>
        </CardContent>
      )}
    </Card>
  )
}
