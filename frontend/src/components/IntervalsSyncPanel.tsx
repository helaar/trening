import { useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { CheckCircle, Loader2, RefreshCw } from "lucide-react"
import { Button } from "./ui/button"
import {
  fetchIntervalsPreview,
  patchAthleteSettings,
  type IntervalsPreview,
  type IntervalsSyncSection,
  type ZoneDefinition,
} from "../api/athleteSettings"

export interface DiffField<T> {
  label: string
  get: (value: T) => string | number | null
}

interface IntervalsSyncPanelProps<S extends IntervalsSyncSection> {
  athleteId: number
  section: S
  saved: IntervalsPreview[S] | null
  intervalsConfigured: boolean
  fields: DiffField<IntervalsPreview[S]>[]
  zonesLabel: string
  getZones: (value: IntervalsPreview[S]) => ZoneDefinition[]
}

function formatZone(zone: ZoneDefinition | undefined): string {
  if (!zone) return "—"
  const range = zone.max === null ? `${zone.min}+` : `${zone.min}–${zone.max}`
  return `${zone.name}: ${range}`
}

function sameZone(a: ZoneDefinition | undefined, b: ZoneDefinition | undefined): boolean {
  return a?.name === b?.name && a?.min === b?.min && a?.max === b?.max
}

// apiFetch errors read "API <status>: <body>"; FastAPI bodies carry a readable `detail`.
function errorDetail(error: Error): string {
  const body = error.message.replace(/^API \d+: /, "")
  try {
    const detail = JSON.parse(body).detail
    if (typeof detail === "string") return detail
  } catch {
    // Not JSON, fall through to the raw message.
  }
  return error.message
}

export function IntervalsSyncPanel<S extends IntervalsSyncSection>({
  athleteId,
  section,
  saved,
  intervalsConfigured,
  fields,
  zonesLabel,
  getZones,
}: IntervalsSyncPanelProps<S>) {
  const queryClient = useQueryClient()
  const [proposed, setProposed] = useState<IntervalsPreview[S] | null>(null)
  const [justSaved, setJustSaved] = useState(false)

  const preview = useMutation({
    mutationFn: () => fetchIntervalsPreview(athleteId, section),
    onSuccess: (data) => {
      setJustSaved(false)
      setProposed(data)
    },
  })

  const apply = useMutation({
    mutationFn: (value: IntervalsPreview[S]) =>
      patchAthleteSettings(athleteId, { [section]: value }),
    onSuccess: (data) => {
      queryClient.setQueryData(["athlete-settings", athleteId], data)
      setProposed(null)
      setJustSaved(true)
      setTimeout(() => setJustSaved(false), 3000)
    },
  })

  const savedZones = saved ? getZones(saved) : []
  const proposedZones = proposed ? getZones(proposed) : []
  const fieldRows = proposed
    ? fields.map((f) => ({
        label: f.label,
        current: saved ? f.get(saved) : null,
        next: f.get(proposed),
      }))
    : []
  const zoneRows = Array.from(
    { length: Math.max(savedZones.length, proposedZones.length) },
    (_, i) => ({ current: savedZones[i], next: proposedZones[i] }),
  )
  const inSync =
    proposed !== null &&
    fieldRows.every((r) => r.current === r.next) &&
    zoneRows.every((r) => sameZone(r.current, r.next))

  return (
    <div className="space-y-3 rounded-md border p-3">
      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="outline"
          size="sm"
          onClick={() => preview.mutate()}
          disabled={!intervalsConfigured || preview.isPending || apply.isPending}
        >
          {preview.isPending ? (
            <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
          ) : (
            <RefreshCw className="mr-2 h-3.5 w-3.5" />
          )}
          Sync from Intervals.icu
        </Button>
        {!intervalsConfigured && (
          <span className="text-xs text-muted-foreground">
            Configure Intervals.icu below to enable syncing.
          </span>
        )}
        {justSaved && (
          <span className="flex items-center gap-1 text-sm text-green-600">
            <CheckCircle className="h-3.5 w-3.5" />
            Saved from Intervals.icu
          </span>
        )}
        {inSync && (
          <span className="flex items-center gap-1 text-sm text-muted-foreground">
            <CheckCircle className="h-3.5 w-3.5" />
            Already in sync with Intervals.icu
          </span>
        )}
      </div>

      {preview.isError && (
        <p className="text-sm text-destructive">{errorDetail(preview.error)}</p>
      )}

      {proposed && !inSync && (
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">
            Review the changes from Intervals.icu. Saving replaces this section, including
            any unsaved edits in the form below.
          </p>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-muted-foreground">
                <th className="py-1 pr-2 font-medium" />
                <th className="py-1 pr-2 font-medium">Current</th>
                <th className="py-1 font-medium">Intervals.icu</th>
              </tr>
            </thead>
            <tbody>
              {fieldRows.map((r) => (
                <tr
                  key={r.label}
                  className={r.current !== r.next ? "bg-amber-500/10" : undefined}
                >
                  <td className="py-1 pr-2 text-muted-foreground">{r.label}</td>
                  <td className="py-1 pr-2">{r.current ?? "—"}</td>
                  <td className="py-1 font-medium">{r.next ?? "—"}</td>
                </tr>
              ))}
              <tr>
                <td colSpan={3} className="pt-3 pb-1 text-xs font-medium text-muted-foreground">
                  {zonesLabel}
                </td>
              </tr>
              {zoneRows.map((r, i) => (
                <tr
                  key={i}
                  className={!sameZone(r.current, r.next) ? "bg-amber-500/10" : undefined}
                >
                  <td className="py-1 pr-2 text-muted-foreground">Z{i + 1}</td>
                  <td className="py-1 pr-2">{formatZone(r.current)}</td>
                  <td className="py-1 font-medium">{formatZone(r.next)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="flex items-center gap-3">
            <Button size="sm" onClick={() => apply.mutate(proposed)} disabled={apply.isPending}>
              {apply.isPending && <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />}
              Save
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setProposed(null)}
              disabled={apply.isPending}
            >
              Cancel
            </Button>
            {apply.isError && (
              <span className="text-sm text-destructive">Save failed. Try again.</span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
