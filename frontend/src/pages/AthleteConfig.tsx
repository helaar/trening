import { useState, useEffect } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Loader2, CheckCircle, Plus, Trash2, ChevronLeft } from "lucide-react"
import { Link } from "@tanstack/react-router"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { Label } from "../components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card"
import {
  fetchAthleteSettings,
  patchAthleteSettings,
  type ZoneDefinition,
  type SportSettings,
  type HeartRateSettings,
  type AthleteSettings,
} from "../api/athleteSettings"
import { fetchCurrentAthlete } from "../api/auth"

// ── helpers ──────────────────────────────────────────────────────────────────

function autolapToMinutes(iso: string | null): string {
  if (!iso) return ""
  const m = iso.match(/PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?/)
  if (!m) return ""
  const hours = parseInt(m[1] ?? "0")
  const mins = parseInt(m[2] ?? "0")
  const secs = parseInt(m[3] ?? "0")
  const total = hours * 60 + mins + secs / 60
  return total > 0 ? String(total) : ""
}

function minutesToIso(raw: string): string | null {
  const n = parseFloat(raw)
  if (isNaN(n) || n <= 0) return null
  const totalSecs = Math.round(n * 60)
  const h = Math.floor(totalSecs / 3600)
  const m = Math.floor((totalSecs % 3600) / 60)
  const s = totalSecs % 60
  let iso = "PT"
  if (h) iso += `${h}H`
  if (m) iso += `${m}M`
  if (s) iso += `${s}S`
  return iso === "PT" ? null : iso
}

function emptyZone(): ZoneDefinition {
  return { name: "", min: 0, max: null }
}

function emptySport(): SportSettings {
  return { ftp: 0, measured_activity: null, measured_date: null, power_zones: [] }
}

function emptyHR(): HeartRateSettings {
  return { max: 0, lt: 0, measured_activity: null, measured_date: null, hr_zones: [] }
}

// ── zone table ────────────────────────────────────────────────────────────────

interface ZoneTableProps {
  zones: ZoneDefinition[]
  onChange: (zones: ZoneDefinition[]) => void
}

function ZoneTable({ zones, onChange }: ZoneTableProps) {
  function update(index: number, patch: Partial<ZoneDefinition>) {
    onChange(zones.map((z, i) => (i === index ? { ...z, ...patch } : z)))
  }

  function remove(index: number) {
    onChange(zones.filter((_, i) => i !== index))
  }

  return (
    <div className="space-y-2">
      {zones.length > 0 && (
        <div className="grid grid-cols-[1fr_80px_80px_32px] gap-2 text-xs font-medium text-muted-foreground px-1">
          <span>Name</span>
          <span>Min</span>
          <span>Max</span>
          <span />
        </div>
      )}
      {zones.map((zone, i) => (
        <div key={i} className="grid grid-cols-[1fr_80px_80px_32px] gap-2 items-center">
          <Input
            value={zone.name}
            placeholder="Zone name"
            onChange={(e) => update(i, { name: e.target.value })}
          />
          <Input
            type="number"
            min={0}
            value={zone.min}
            onChange={(e) => update(i, { min: parseInt(e.target.value) || 0 })}
          />
          <Input
            type="number"
            min={0}
            placeholder="∞"
            value={zone.max ?? ""}
            onChange={(e) =>
              update(i, { max: e.target.value === "" ? null : parseInt(e.target.value) || 0 })
            }
          />
          <Button
            variant="ghost"
            size="icon"
            onClick={() => remove(i)}
            aria-label="Remove zone"
            className="h-8 w-8 text-muted-foreground hover:text-destructive"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      ))}
      <Button
        variant="outline"
        size="sm"
        onClick={() => onChange([...zones, emptyZone()])}
        className="mt-1"
      >
        <Plus className="mr-1.5 h-3.5 w-3.5" />
        Add zone
      </Button>
    </div>
  )
}

// ── sport section ─────────────────────────────────────────────────────────────

interface SportSectionProps {
  title: string
  field: "cycling" | "running"
  initial: SportSettings | null
  athleteId: number
}

function SportSection({ title, field, initial, athleteId }: SportSectionProps) {
  const queryClient = useQueryClient()
  const [value, setValue] = useState<SportSettings>(initial ?? emptySport())
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    setValue(initial ?? emptySport())
  }, [initial])

  const mutation = useMutation({
    mutationFn: () => patchAthleteSettings(athleteId, { [field]: value }),
    onSuccess: (data) => {
      queryClient.setQueryData(["athlete-settings", athleteId], data)
      setSaved(true)
      setTimeout(() => setSaved(false), 3000)
    },
  })

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
          <div className="space-y-1.5">
            <Label htmlFor={`${field}-ftp`}>FTP (W)</Label>
            <Input
              id={`${field}-ftp`}
              type="number"
              min={0}
              placeholder="280"
              value={value.ftp || ""}
              onChange={(e) => setValue({ ...value, ftp: parseInt(e.target.value) || 0 })}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor={`${field}-date`}>Measured date</Label>
            <Input
              id={`${field}-date`}
              type="date"
              value={value.measured_date ?? ""}
              onChange={(e) =>
                setValue({ ...value, measured_date: e.target.value || null })
              }
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor={`${field}-activity`}>Measured in activity</Label>
            <Input
              id={`${field}-activity`}
              placeholder="Activity name or ID"
              value={value.measured_activity ?? ""}
              onChange={(e) =>
                setValue({ ...value, measured_activity: e.target.value || null })
              }
            />
          </div>
        </div>

        <div className="space-y-2">
          <Label>Power zones</Label>
          <ZoneTable
            zones={value.power_zones}
            onChange={(zones) => setValue({ ...value, power_zones: zones })}
          />
        </div>

        <SaveRow saved={saved} isPending={mutation.isPending} isError={mutation.isError} onSave={() => mutation.mutate()} />
      </CardContent>
    </Card>
  )
}

// ── heart rate section ────────────────────────────────────────────────────────

interface HRSectionProps {
  initial: HeartRateSettings | null
  athleteId: number
}

function HRSection({ initial, athleteId }: HRSectionProps) {
  const queryClient = useQueryClient()
  const [value, setValue] = useState<HeartRateSettings>(initial ?? emptyHR())
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    setValue(initial ?? emptyHR())
  }, [initial])

  const mutation = useMutation({
    mutationFn: () => patchAthleteSettings(athleteId, { heart_rate: value }),
    onSuccess: (data) => {
      queryClient.setQueryData(["athlete-settings", athleteId], data)
      setSaved(true)
      setTimeout(() => setSaved(false), 3000)
    },
  })

  return (
    <Card>
      <CardHeader>
        <CardTitle>Heart Rate</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div className="space-y-1.5">
            <Label htmlFor="hr-max">Max HR (bpm)</Label>
            <Input
              id="hr-max"
              type="number"
              min={0}
              max={250}
              placeholder="185"
              value={value.max || ""}
              onChange={(e) => setValue({ ...value, max: parseInt(e.target.value) || 0 })}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="hr-lt">Lactate threshold (bpm)</Label>
            <Input
              id="hr-lt"
              type="number"
              min={0}
              max={250}
              placeholder="162"
              value={value.lt || ""}
              onChange={(e) => setValue({ ...value, lt: parseInt(e.target.value) || 0 })}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="hr-date">Measured date</Label>
            <Input
              id="hr-date"
              type="date"
              value={value.measured_date ?? ""}
              onChange={(e) =>
                setValue({ ...value, measured_date: e.target.value || null })
              }
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="hr-activity">Measured in activity</Label>
            <Input
              id="hr-activity"
              placeholder="Activity name or ID"
              value={value.measured_activity ?? ""}
              onChange={(e) =>
                setValue({ ...value, measured_activity: e.target.value || null })
              }
            />
          </div>
        </div>

        <div className="space-y-2">
          <Label>HR zones</Label>
          <ZoneTable
            zones={value.hr_zones}
            onChange={(zones) => setValue({ ...value, hr_zones: zones })}
          />
        </div>

        <SaveRow saved={saved} isPending={mutation.isPending} isError={mutation.isError} onSave={() => mutation.mutate()} />
      </CardContent>
    </Card>
  )
}

// ── autolap section ───────────────────────────────────────────────────────────

interface AutolapSectionProps {
  initial: string | null
  athleteId: number
}

function AutolapSection({ initial, athleteId }: AutolapSectionProps) {
  const queryClient = useQueryClient()
  const [raw, setRaw] = useState(autolapToMinutes(initial))
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    setRaw(autolapToMinutes(initial))
  }, [initial])

  const mutation = useMutation({
    mutationFn: () => patchAthleteSettings(athleteId, { autolap: minutesToIso(raw) }),
    onSuccess: (data) => {
      queryClient.setQueryData(["athlete-settings", athleteId], data)
      setSaved(true)
      setTimeout(() => setSaved(false), 3000)
    },
  })

  return (
    <Card>
      <CardHeader>
        <CardTitle>Autolap</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="max-w-[160px] space-y-1.5">
          <Label htmlFor="autolap">Interval (minutes)</Label>
          <Input
            id="autolap"
            type="number"
            min={0}
            step={1}
            placeholder="10"
            value={raw}
            onChange={(e) => setRaw(e.target.value)}
          />
          <p className="text-xs text-muted-foreground">Leave empty to disable autolap</p>
        </div>
        <SaveRow saved={saved} isPending={mutation.isPending} isError={mutation.isError} onSave={() => mutation.mutate()} />
      </CardContent>
    </Card>
  )
}

// ── shared save row ───────────────────────────────────────────────────────────

interface SaveRowProps {
  saved: boolean
  isPending: boolean
  isError: boolean
  onSave: () => void
}

function SaveRow({ saved, isPending, isError, onSave }: SaveRowProps) {
  return (
    <div className="flex items-center gap-3 pt-2">
      <Button onClick={onSave} disabled={isPending} size="sm">
        {isPending && <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />}
        Save
      </Button>
      {saved && (
        <span className="flex items-center gap-1 text-sm text-green-600">
          <CheckCircle className="h-3.5 w-3.5" />
          Saved
        </span>
      )}
      {isError && (
        <span className="text-sm text-destructive">Save failed. Try again.</span>
      )}
    </div>
  )
}

// ── page ──────────────────────────────────────────────────────────────────────

export function AthleteConfig() {
  const { data: athlete, isLoading: loadingAthlete, error: athleteError } = useQuery({
    queryKey: ["athlete"],
    queryFn: fetchCurrentAthlete,
  })

  const { data: settings, isLoading: loadingSettings } = useQuery<AthleteSettings>({
    queryKey: ["athlete-settings", athlete?.athlete_id],
    queryFn: () => fetchAthleteSettings(athlete!.athlete_id),
    enabled: !!athlete,
  })

  if (loadingAthlete || loadingSettings) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (athleteError) {
    return (
      <div className="flex min-h-screen items-center justify-center p-4">
        <p className="text-sm text-destructive">Could not load athlete. Is the backend running?</p>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-4 sm:p-6">
      <div className="flex items-center gap-3">
        <Link to="/">
          <Button variant="ghost" size="icon" aria-label="Back">
            <ChevronLeft className="h-4 w-4" />
          </Button>
        </Link>
        <div>
          <h1 className="text-2xl font-bold">Athlete Settings</h1>
          <p className="text-sm text-muted-foreground">
            {athlete?.firstname} {athlete?.lastname}
          </p>
        </div>
      </div>

      <SportSection
        title="Cycling"
        field="cycling"
        initial={settings?.cycling ?? null}
        athleteId={athlete!.athlete_id}
      />

      <SportSection
        title="Running"
        field="running"
        initial={settings?.running ?? null}
        athleteId={athlete!.athlete_id}
      />

      <HRSection
        initial={settings?.heart_rate ?? null}
        athleteId={athlete!.athlete_id}
      />

      <AutolapSection
        initial={settings?.autolap ?? null}
        athleteId={athlete!.athlete_id}
      />
    </div>
  )
}
