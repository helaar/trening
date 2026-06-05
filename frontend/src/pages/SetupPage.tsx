import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Loader2, CheckCircle, Clipboard, ClipboardCheck, Upload } from "lucide-react"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { Label } from "../components/ui/label"
import { Textarea } from "../components/ui/textarea"
import {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from "../components/ui/accordion"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogClose,
} from "../components/ui/dialog"
import { fetchPrompts, savePrompts, type PromptConfig } from "../api/prompts"

function groupPrompts(prompts: PromptConfig[]): Map<string, PromptConfig[]> {
  const groups = new Map<string, PromptConfig[]>()
  for (const p of prompts) {
    const parts = p.key.split(".")
    const groupKey = parts.length >= 2 ? `${parts[0]}.${parts[1]}` : parts[0]
    const list = groups.get(groupKey) ?? []
    list.push(p)
    groups.set(groupKey, list)
  }
  return groups
}

function groupLabel(groupKey: string): string {
  const parts = groupKey.split(".")
  if (parts.length < 2) return groupKey
  const section = parts[0] === "agents" ? "Agent" : "Task"
  const name = parts[1].replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
  return `${section}: ${name}`
}

function fieldLabel(key: string): string {
  const parts = key.split(".")
  return parts[parts.length - 1].replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

function GroupCopyButton({
  items,
  currentValues,
}: {
  items: PromptConfig[]
  currentValues: (p: PromptConfig) => string
}) {
  const [copied, setCopied] = useState(false)
  function handleCopy(e: React.MouseEvent) {
    e.stopPropagation()
    const payload = items.map((p) => ({ key: p.key, value: currentValues(p) }))
    navigator.clipboard.writeText(JSON.stringify(payload, null, 2))
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <button
      type="button"
      onClick={handleCopy}
      className="rounded p-1 text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
      title="Copy as JSON"
    >
      {copied
        ? <ClipboardCheck className="h-4 w-4 text-green-600" />
        : <Clipboard className="h-4 w-4" />}
    </button>
  )
}

interface ImportModalProps {
  groupKey: string
  onImport: (updates: Record<string, string>) => void
  onClose: () => void
}

function ImportModal({ groupKey, onImport, onClose }: ImportModalProps) {
  const [text, setText] = useState("")
  const [error, setError] = useState<string | null>(null)

  function handleApply() {
    const trimmed = text.trim()
    if (!trimmed) { setError("Paste the JSON above."); return }
    try {
      const parsed = JSON.parse(trimmed)
      if (!Array.isArray(parsed)) throw new Error()
      const updates: Record<string, string> = {}
      for (const item of parsed) {
        if (typeof item.key === "string" && typeof item.value === "string") {
          updates[item.key] = item.value
        }
      }
      if (Object.keys(updates).length === 0) throw new Error()
      onImport(updates)
      onClose()
    } catch {
      setError("Invalid JSON — expected [{key, value}, …].")
    }
  }

  return (
    <DialogContent className="max-w-2xl">
      <DialogHeader>
        <DialogTitle>Import — {groupLabel(groupKey)}</DialogTitle>
        <DialogDescription>
          Paste the JSON array back from the LLM.
        </DialogDescription>
      </DialogHeader>
      <Textarea
        autoFocus
        placeholder={'[\n  { "key": "…", "value": "…" }\n]'}
        value={text}
        onChange={(e) => { setText(e.target.value); setError(null) }}
        rows={14}
        className="font-mono text-xs resize-y"
      />
      {error && <p className="text-xs text-destructive mt-1">{error}</p>}
      <div className="flex justify-end gap-2 mt-4">
        <DialogClose asChild>
          <Button variant="outline" size="sm" onClick={onClose}>Cancel</Button>
        </DialogClose>
        <Button size="sm" onClick={handleApply}>Apply</Button>
      </div>
    </DialogContent>
  )
}

const GLOBAL_PHILOSOPHY_KEYS = [
  { key: "philosophy.name", label: "Philosophy name", placeholder: "e.g. Polarized (80/20)", multiline: false },
  { key: "philosophy.intensity_targets", label: "Intensity targets", placeholder: "e.g. low ≥80%, moderate <5%, high ~20%", multiline: false },
  { key: "philosophy.coach_guidance", label: "Coach guidance", placeholder: "What the daily coach should do with the intensity data…", multiline: true },
  { key: "philosophy.analyst_guidance", label: "Analyst guidance", placeholder: "What the performance analyst should flag…", multiline: true },
]

function GlobalPhilosophySection({
  prompts,
  edits,
  onChange,
}: {
  prompts: PromptConfig[] | undefined
  edits: Record<string, string>
  onChange: (key: string, value: string) => void
}) {
  const byKey = new Map((prompts ?? []).map((p) => [p.key, p.value]))

  function currentValue(key: string): string {
    return edits[key] ?? byKey.get(key) ?? ""
  }

  return (
    <AccordionItem value="philosophy.global" className="border rounded-lg px-4">
      <AccordionTrigger className="text-sm font-medium py-3">
        Default Training Philosophy
      </AccordionTrigger>
      <AccordionContent className="space-y-4 pb-4">
        <p className="text-xs text-muted-foreground">
          Global default applied to all athletes. Per-athlete overrides are set in Athlete Settings.
        </p>
        {GLOBAL_PHILOSOPHY_KEYS.map(({ key, label, placeholder, multiline }) => (
          <div key={key} className="space-y-1.5">
            <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              {label}
            </Label>
            {multiline ? (
              <Textarea
                value={currentValue(key)}
                placeholder={placeholder}
                onChange={(e) => onChange(key, e.target.value)}
                rows={4}
                className="font-mono text-xs resize-y"
              />
            ) : (
              <Input
                value={currentValue(key)}
                placeholder={placeholder}
                onChange={(e) => onChange(key, e.target.value)}
                className="font-mono text-xs"
              />
            )}
          </div>
        ))}
      </AccordionContent>
    </AccordionItem>
  )
}

export function SetupPage() {
  const queryClient = useQueryClient()
  const { data: prompts, isLoading } = useQuery({
    queryKey: ["admin-prompts"],
    queryFn: fetchPrompts,
  })

  const [edits, setEdits] = useState<Record<string, string>>({})
  const [saved, setSaved] = useState(false)
  const [importGroup, setImportGroup] = useState<string | null>(null)

  const mutation = useMutation({
    mutationFn: savePrompts,
    onSuccess: (updated) => {
      queryClient.setQueryData<PromptConfig[]>(["admin-prompts"], (prev) => {
        if (!prev) return updated
        const byKey = new Map(updated.map((p) => [p.key, p]))
        const merged = prev.map((p) => byKey.get(p.key) ?? p)
        const added = updated.filter((p) => !prev.some((e) => e.key === p.key))
        return [...merged, ...added]
      })
      setEdits({})
      setSaved(true)
      setTimeout(() => setSaved(false), 3000)
    },
  })

  function currentValue(p: PromptConfig): string {
    return edits[p.key] ?? p.value
  }

  function handleChange(key: string, value: string) {
    setEdits((prev) => ({ ...prev, [key]: value }))
  }

  function handleImport(updates: Record<string, string>) {
    setEdits((prev) => ({ ...prev, ...updates }))
  }

  function handleSave() {
    const updates = Object.entries(edits).map(([key, value]) => ({ key, value }))
    if (updates.length > 0) mutation.mutate(updates)
  }

  const hasChanges = Object.keys(edits).length > 0

  if (isLoading) {
    return (
      <div className="h-full overflow-y-auto flex items-center justify-center p-12 text-muted-foreground">
        <Loader2 className="h-5 w-5 animate-spin mr-2" />
        Loading prompts…
      </div>
    )
  }

  const groups = groupPrompts(prompts ?? [])

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-3xl mx-auto px-4 py-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold">LLM Prompt Configuration</h1>
            <p className="text-sm text-muted-foreground mt-1">
              Overrides are stored in the database. Clearing a field and saving restores the YAML default.
            </p>
          </div>
          <div className="flex items-center gap-3">
            {saved && (
              <span className="flex items-center gap-1.5 text-sm text-green-600">
                <CheckCircle className="h-4 w-4" /> Saved
              </span>
            )}
            <Button size="sm" onClick={handleSave} disabled={!hasChanges || mutation.isPending}>
              {mutation.isPending && <Loader2 className="h-4 w-4 animate-spin mr-2" />}
              Save changes
            </Button>
          </div>
        </div>

        <Accordion type="multiple" className="space-y-2">
          <GlobalPhilosophySection prompts={prompts} edits={edits} onChange={handleChange} />
          {Array.from(groups.entries()).map(([groupKey, items]) => (
            <AccordionItem key={groupKey} value={groupKey} className="border rounded-lg px-4">
              <AccordionTrigger className="text-sm font-medium py-3">
                <span className="flex-1 text-left">{groupLabel(groupKey)}</span>
                <div className="flex items-center gap-1 mr-2" onClick={(e) => e.stopPropagation()}>
                  <GroupCopyButton items={items} currentValues={currentValue} />
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); setImportGroup(groupKey) }}
                    className="rounded p-1 text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
                    title="Import from JSON"
                  >
                    <Upload className="h-4 w-4" />
                  </button>
                </div>
              </AccordionTrigger>
              <AccordionContent className="space-y-4 pb-4">
                {items.map((p) => (
                  <div key={p.key} className="space-y-1.5">
                    <label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                      {fieldLabel(p.key)}
                    </label>
                    <Textarea
                      value={currentValue(p)}
                      onChange={(e) => handleChange(p.key, e.target.value)}
                      rows={Math.min(Math.max(currentValue(p).split("\n").length + 1, 3), 12)}
                      className="font-mono text-xs resize-y"
                    />
                  </div>
                ))}
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </div>

      <Dialog open={importGroup !== null} onOpenChange={(open) => { if (!open) setImportGroup(null) }}>
        {importGroup !== null && (
          <ImportModal
            groupKey={importGroup}
            onImport={handleImport}
            onClose={() => setImportGroup(null)}
          />
        )}
      </Dialog>
    </div>
  )
}
