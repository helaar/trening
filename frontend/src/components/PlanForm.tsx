import { useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { X, Loader2, Trash2 } from "lucide-react"
import { Button } from "./ui/button"
import { Input } from "./ui/input"
import { Label } from "./ui/label"
import { Textarea } from "./ui/textarea"
import { cn } from "../lib/utils"
import { createPlan, updatePlan, deletePlan } from "../api/plans"
import type { PlannedActivity, PlannedActivityRequest, Sport } from "../api/plans"

const SPORT_OPTIONS: { value: Sport; label: string }[] = [
  { value: "cycling", label: "Cycling" },
  { value: "running", label: "Running" },
  { value: "strength", label: "Strength" },
  { value: "skiing_cross", label: "XC Skiing" },
  { value: "skiing_alpine", label: "Alpine Skiing" },
  { value: "day_off", label: "Day off" },
  { value: "other", label: "Other" },
]

interface Props {
  athleteId: number
  date: string
  existing?: PlannedActivity
  onSaved: (plan: PlannedActivity) => void
  onDeleted: () => void
}

export function PlanForm({ athleteId, date, existing, onSaved, onDeleted }: Props) {
  const queryClient = useQueryClient()
  const isEditing = !!existing

  const [sport, setSport] = useState<Sport>(existing?.sport ?? "cycling")
  const [name, setName] = useState(existing?.name ?? "")
  const [description, setDescription] = useState(existing?.description ?? "")
  const [purpose, setPurpose] = useState(existing?.purpose ?? "")
  const [durationMin, setDurationMin] = useState(
    existing?.estimated_duration_min?.toString() ?? ""
  )
  const [tss, setTss] = useState(existing?.estimated_tss?.toString() ?? "")
  const [labels, setLabels] = useState<string[]>(existing?.labels ?? [])
  const [labelInput, setLabelInput] = useState("")

  function addLabel(raw: string) {
    const trimmed = raw.trim().replace(/,+$/, "").trim()
    if (trimmed && !labels.includes(trimmed)) {
      setLabels((prev) => [...prev, trimmed])
    }
    setLabelInput("")
  }

  function handleLabelKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault()
      addLabel(labelInput)
    } else if (e.key === "Backspace" && labelInput === "" && labels.length > 0) {
      setLabels((prev) => prev.slice(0, -1))
    }
  }

  function buildRequest(): PlannedActivityRequest {
    return {
      date,
      sport,
      name: name.trim(),
      description: description.trim() || undefined,
      purpose: purpose.trim() || undefined,
      labels,
      estimated_duration_min: durationMin ? parseInt(durationMin, 10) : undefined,
      estimated_tss: tss ? parseInt(tss, 10) : undefined,
    }
  }

  const saveMutation = useMutation({
    mutationFn: () =>
      isEditing
        ? updatePlan(athleteId, existing.id, buildRequest())
        : createPlan(athleteId, buildRequest()),
    onSuccess: (saved) => {
      queryClient.invalidateQueries({ queryKey: ["plans", athleteId] })
      onSaved(saved)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: () => deletePlan(athleteId, existing!.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["plans", athleteId] })
      onDeleted()
    },
  })

  const isPending = saveMutation.isPending || deleteMutation.isPending

  return (
    <div className="space-y-5">
      <div className="space-y-1.5">
        <Label htmlFor="sport">Sport</Label>
        <select
          id="sport"
          value={sport}
          onChange={(e) => setSport(e.target.value as Sport)}
          className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
        >
          {SPORT_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </div>

      <div className="space-y-1.5">
        <Label htmlFor="name">Name</Label>
        <Input
          id="name"
          placeholder="e.g. Zwift SubLT 3x10"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1.5">
          <Label htmlFor="duration">Duration (min)</Label>
          <Input
            id="duration"
            type="number"
            min={1}
            placeholder="90"
            value={durationMin}
            onChange={(e) => setDurationMin(e.target.value)}
          />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="tss">Est. TSS</Label>
          <Input
            id="tss"
            type="number"
            min={1}
            placeholder="85"
            value={tss}
            onChange={(e) => setTss(e.target.value)}
          />
        </div>
      </div>

      <div className="space-y-1.5">
        <Label>Labels</Label>
        <div className="flex min-h-10 flex-wrap gap-1.5 rounded-md border border-input bg-background px-3 py-2 text-sm focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2">
          {labels.map((label) => (
            <span
              key={label}
              className="flex items-center gap-1 rounded-full bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground"
            >
              {label}
              <button
                type="button"
                onClick={() => setLabels((prev) => prev.filter((l) => l !== label))}
                className="text-muted-foreground hover:text-foreground"
                aria-label={`Remove ${label}`}
              >
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
          <input
            type="text"
            value={labelInput}
            onChange={(e) => setLabelInput(e.target.value)}
            onKeyDown={handleLabelKeyDown}
            onBlur={() => labelInput && addLabel(labelInput)}
            placeholder={labels.length === 0 ? "LT1, optional, outdoor…" : ""}
            className="min-w-24 flex-1 bg-transparent outline-none placeholder:text-muted-foreground"
          />
        </div>
        <p className="text-xs text-muted-foreground">Press Enter or comma to add a label</p>
      </div>

      <div className="space-y-1.5">
        <Label htmlFor="description">Description</Label>
        <Textarea
          id="description"
          placeholder="3x10 min @82%, cadence 70-80rpm, 5 min z1 recovery between sets"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={3}
        />
      </div>

      <div className="space-y-1.5">
        <Label htmlFor="purpose">Purpose</Label>
        <Textarea
          id="purpose"
          placeholder="Why this session — training goal or coach rationale"
          value={purpose}
          onChange={(e) => setPurpose(e.target.value)}
          rows={2}
        />
      </div>

      {saveMutation.isError && (
        <p className="text-sm text-destructive">
          Save failed: {(saveMutation.error as Error).message}
        </p>
      )}
      {deleteMutation.isError && (
        <p className="text-sm text-destructive">
          Delete failed: {(deleteMutation.error as Error).message}
        </p>
      )}

      <div className={cn("flex gap-3", isEditing && "justify-between")}>
        {isEditing && (
          <Button
            variant="outline"
            onClick={() => deleteMutation.mutate()}
            disabled={isPending}
            className="text-destructive hover:bg-destructive/10 hover:text-destructive"
          >
            {deleteMutation.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Trash2 className="h-4 w-4" />
            )}
            Delete
          </Button>
        )}
        <Button
          onClick={() => saveMutation.mutate()}
          disabled={isPending || !name.trim()}
          className="ml-auto"
        >
          {saveMutation.isPending && <Loader2 className="h-4 w-4 animate-spin" />}
          {isEditing ? "Save changes" : "Add plan"}
        </Button>
      </div>
    </div>
  )
}
