import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"
import { Input } from "./ui/input"
import { Label } from "./ui/label"
import { Textarea } from "./ui/textarea"
import type { Restitution } from "../api/dailyEntry"

interface Props {
  value: Restitution
  onChange: (value: Restitution) => void
}

export function RestitutionForm({ value, onChange }: Props) {
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
          <Label htmlFor="sleep">Sleep (hours)</Label>
          <Input
            id="sleep"
            type="number"
            min={0}
            max={15}
            step={0.5}
            placeholder="7.5"
            value={value.sleep_hours ?? ""}
            onChange={(e) => set("sleep_hours", e.target.value)}
          />
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
