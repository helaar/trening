import { useEffect, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Loader2,
  CheckCircle,
  Clipboard,
  ClipboardCheck,
  Upload,
  ChevronDown,
  ChevronRight,
  Plus,
} from "lucide-react"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { Label } from "../components/ui/label"
import { Textarea } from "../components/ui/textarea"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogClose,
} from "../components/ui/dialog"
import {
  fetchPrompts,
  savePrompts,
  type CrewDefinition,
  type PhilosophyDoc,
} from "../api/prompts"
import { createMemoryConsolidationTask, getTaskStatus } from "../api/tasks"

// ── document field model ───────────────────────────────────────────────────────

type DefType = CrewDefinition["type"]

// Editable fields per document type, in display order.
const FIELD_DEFS: Record<DefType, string[]> = {
  agent: ["role", "goal", "backstory", "llm_model"],
  task: ["description", "expected_output"],
  philosophy: ["display_name", "intensity_targets", "coach_guidance", "analyst_guidance"],
}

const FIELD_LABELS: Record<string, string> = {
  llm_model: "LLM Model",
  display_name: "Display Name",
  expected_output: "Expected Output",
  intensity_targets: "Intensity Targets",
  coach_guidance: "Coach Guidance",
  analyst_guidance: "Analyst Guidance",
}

function humanize(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

function fieldLabel(field: string): string {
  return FIELD_LABELS[field] ?? humanize(field)
}

function docLabel(doc: CrewDefinition): string {
  if (doc.type === "philosophy") return doc.display_name?.trim() || humanize(doc.name)
  return humanize(doc.name)
}

function slugify(name: string): string {
  return name.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "")
}

function editKey(type: DefType, name: string, field: string): string {
  return `${type}/${name}/${field}`
}

// ── tree node types ────────────────────────────────────────────────────────────

interface TreeNode {
  type: DefType
  name: string
}

// ── copy button ────────────────────────────────────────────────────────────────

function DocCopyButton({ doc }: { doc: Record<string, string> }) {
  const [copied, setCopied] = useState(false)
  function handleCopy() {
    navigator.clipboard.writeText(JSON.stringify(doc, null, 2))
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <button
      type="button"
      onClick={handleCopy}
      className="rounded p-1.5 text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
      title="Copy as JSON"
    >
      {copied
        ? <ClipboardCheck className="h-4 w-4 text-green-600" />
        : <Clipboard className="h-4 w-4" />}
    </button>
  )
}

// ── import modal ────────────────────────────────────────────────────────────────

interface ImportModalProps {
  label: string
  fields: string[]
  onImport: (values: Record<string, string>) => void
  onClose: () => void
}

function ImportModal({ label, fields, onImport, onClose }: ImportModalProps) {
  const [text, setText] = useState("")
  const [error, setError] = useState<string | null>(null)

  function handleApply() {
    const trimmed = text.trim()
    if (!trimmed) { setError("Paste the JSON above."); return }
    try {
      const parsed = JSON.parse(trimmed)
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) throw new Error()
      const values: Record<string, string> = {}
      for (const field of fields) {
        if (typeof parsed[field] === "string") values[field] = parsed[field]
      }
      if (Object.keys(values).length === 0) throw new Error()
      onImport(values)
      onClose()
    } catch {
      setError("Invalid JSON — expected an object with this document's fields.")
    }
  }

  return (
    <DialogContent className="max-w-2xl">
      <DialogHeader>
        <DialogTitle>Import — {label}</DialogTitle>
        <DialogDescription>Paste the JSON document back from the LLM.</DialogDescription>
      </DialogHeader>
      <Textarea
        autoFocus
        placeholder={'{\n  "role": "…",\n  "goal": "…"\n}'}
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

// ── tree sidebar ────────────────────────────────────────────────────────────────

interface TreeBranchProps {
  label: string
  type: DefType
  items: { name: string; label: string }[]
  selected: TreeNode | null
  onSelect: (node: TreeNode) => void
  extra?: React.ReactNode
}

function TreeBranch({ label, type, items, selected, onSelect, extra }: TreeBranchProps) {
  const [open, setOpen] = useState(true)
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 w-full px-3 py-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground hover:text-foreground transition-colors"
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        {label}
      </button>
      {open && (
        <div className="ml-3 pl-3 border-l border-border">
          {items.map(({ name, label: itemLabel }) => {
            const isSelected = selected?.type === type && selected.name === name
            return (
              <button
                key={name}
                type="button"
                onClick={() => onSelect({ type, name })}
                className={`w-full text-left px-2 py-1.5 text-sm rounded-md transition-colors truncate ${
                  isSelected
                    ? "bg-accent text-accent-foreground font-medium"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                }`}
              >
                {itemLabel}
              </button>
            )
          })}
          {extra}
        </div>
      )}
    </div>
  )
}

// ── detail panel ────────────────────────────────────────────────────────────────

interface DetailPanelProps {
  doc: CrewDefinition
  fields: string[]
  value: (field: string) => string
  onChange: (field: string, value: string) => void
  onImport: (values: Record<string, string>) => void
}

function DetailPanel({ doc, fields, value, onChange, onImport }: DetailPanelProps) {
  const [showImport, setShowImport] = useState(false)
  const label = docLabel(doc)
  const snapshot = Object.fromEntries(fields.map((f) => [f, value(f)]))

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-6 py-4 border-b">
        <h2 className="text-sm font-semibold">{label}</h2>
        <div className="flex items-center gap-1">
          <DocCopyButton doc={snapshot} />
          <button
            type="button"
            onClick={() => setShowImport(true)}
            className="rounded p-1.5 text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
            title="Import from JSON"
          >
            <Upload className="h-4 w-4" />
          </button>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-5">
        {fields.map((field) => (
          <div key={field} className="space-y-1.5">
            <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              {fieldLabel(field)}
            </Label>
            <Textarea
              value={value(field)}
              onChange={(e) => onChange(field, e.target.value)}
              rows={Math.min(Math.max(value(field).split("\n").length + 1, field === "llm_model" ? 1 : 3), 12)}
              className="font-mono text-xs resize-y"
            />
          </div>
        ))}
      </div>
      <Dialog open={showImport} onOpenChange={(open) => { if (!open) setShowImport(false) }}>
        {showImport && (
          <ImportModal
            label={label}
            fields={fields}
            onImport={onImport}
            onClose={() => setShowImport(false)}
          />
        )}
      </Dialog>
    </div>
  )
}

// ── add philosophy dialog ──────────────────────────────────────────────────────

interface AddPhilosophyDialogProps {
  onAdd: (slug: string, name: string) => void
  onClose: () => void
  existingSlugs: Set<string>
}

function AddPhilosophyDialog({ onAdd, onClose, existingSlugs }: AddPhilosophyDialogProps) {
  const [name, setName] = useState("")
  const [error, setError] = useState<string | null>(null)

  function handleAdd() {
    const trimmed = name.trim()
    if (!trimmed) { setError("Enter a name."); return }
    const slug = slugify(trimmed)
    if (!slug) { setError("Name must contain at least one letter or number."); return }
    if (existingSlugs.has(slug)) { setError("A philosophy with this name already exists."); return }
    onAdd(slug, trimmed)
    onClose()
  }

  return (
    <DialogContent className="max-w-sm">
      <DialogHeader>
        <DialogTitle>New Philosophy</DialogTitle>
        <DialogDescription>Enter a name for the new training philosophy.</DialogDescription>
      </DialogHeader>
      <Input
        autoFocus
        placeholder="e.g. Polarized"
        value={name}
        onChange={(e) => { setName(e.target.value); setError(null) }}
        onKeyDown={(e) => { if (e.key === "Enter") handleAdd() }}
      />
      {error && <p className="text-xs text-destructive mt-1">{error}</p>}
      <div className="flex justify-end gap-2 mt-4">
        <DialogClose asChild>
          <Button variant="outline" size="sm" onClick={onClose}>Cancel</Button>
        </DialogClose>
        <Button size="sm" onClick={handleAdd}>Add</Button>
      </div>
    </DialogContent>
  )
}

// ── page ────────────────────────────────────────────────────────────────────────

function newPhilosophy(slug: string, name: string): PhilosophyDoc {
  return {
    type: "philosophy",
    name: slug,
    display_name: name,
    intensity_targets: "",
    coach_guidance: "",
    analyst_guidance: "",
  }
}

export function SetupPage() {
  const queryClient = useQueryClient()
  const { data: prompts, isLoading } = useQuery({
    queryKey: ["admin-prompts"],
    queryFn: fetchPrompts,
  })

  const [edits, setEdits] = useState<Record<string, string>>({})
  const [added, setAdded] = useState<PhilosophyDoc[]>([])
  const [saved, setSaved] = useState(false)
  const [selected, setSelected] = useState<TreeNode | null>(null)
  const [showAddPhilosophy, setShowAddPhilosophy] = useState(false)

  const [consolidationTaskId, setConsolidationTaskId] = useState<string | null>(null)
  const [consolidationStatus, setConsolidationStatus] = useState<"idle" | "success" | "skipped" | "failed">("idle")
  const [consolidationSummary, setConsolidationSummary] = useState<string>("")

  const { data: consolidationTask } = useQuery({
    queryKey: ["task", consolidationTaskId],
    queryFn: () => getTaskStatus(consolidationTaskId!),
    enabled: !!consolidationTaskId,
    refetchInterval: (query) => {
      const s = query.state.data?.status
      return s === "completed" || s === "failed" ? false : 3000
    },
  })

  useEffect(() => {
    if (!consolidationTask) return
    if (consolidationTask.status === "completed") {
      const r = consolidationTask.result ?? {}
      if (r.skipped) {
        setConsolidationStatus("skipped")
        setConsolidationSummary("Skipped — last run was too recent")
      } else {
        setConsolidationStatus("success")
        setConsolidationSummary(
          `Updated ${r.updates ?? 0}, promoted ${r.promotions ?? 0}, deactivated ${r.deactivations ?? 0}, new long-term ${r.new_long_term ?? 0}`
        )
      }
      setTimeout(() => { setConsolidationStatus("idle"); setConsolidationTaskId(null) }, 5000)
    } else if (consolidationTask.status === "failed") {
      setConsolidationStatus("failed")
      setConsolidationSummary(consolidationTask.error ?? "Unknown error")
      setTimeout(() => { setConsolidationStatus("idle"); setConsolidationTaskId(null) }, 5000)
    }
  }, [consolidationTask?.status])

  const consolidationRunning =
    !!consolidationTaskId &&
    consolidationTask?.status !== "completed" &&
    consolidationTask?.status !== "failed"

  async function triggerConsolidation() {
    const task = await createMemoryConsolidationTask()
    setConsolidationTaskId(task.task_id)
    setConsolidationStatus("idle")
  }

  const mutation = useMutation({
    mutationFn: savePrompts,
    onSuccess: (updated) => {
      queryClient.setQueryData<CrewDefinition[]>(["admin-prompts"], (prev) => {
        const base = prev ?? []
        const idx = new Map(base.map((d, i) => [`${d.type}/${d.name}`, i]))
        const next = [...base]
        for (const d of updated) {
          const k = `${d.type}/${d.name}`
          if (idx.has(k)) next[idx.get(k)!] = d
          else next.push(d)
        }
        return next
      })
      setEdits({})
      setAdded([])
      setSaved(true)
      setTimeout(() => setSaved(false), 3000)
    },
  })

  const allDocs: CrewDefinition[] = [...(prompts ?? []), ...added]

  function findDoc(node: TreeNode): CrewDefinition | undefined {
    return allDocs.find((d) => d.type === node.type && d.name === node.name)
  }

  function fieldValue(doc: CrewDefinition, field: string): string {
    const ek = editKey(doc.type, doc.name, field)
    if (ek in edits) return edits[ek]
    return (doc as unknown as Record<string, unknown>)[field] as string ?? ""
  }

  function handleChange(doc: CrewDefinition, field: string, value: string) {
    setEdits((prev) => ({ ...prev, [editKey(doc.type, doc.name, field)]: value }))
  }

  function handleImport(doc: CrewDefinition, values: Record<string, string>) {
    setEdits((prev) => {
      const next = { ...prev }
      for (const [field, value] of Object.entries(values)) {
        next[editKey(doc.type, doc.name, field)] = value
      }
      return next
    })
  }

  function handleAddPhilosophy(slug: string, name: string) {
    setAdded((prev) => [...prev, newPhilosophy(slug, name)])
    setSelected({ type: "philosophy", name: slug })
  }

  function handleSave() {
    const changed = new Set<string>()
    for (const k of Object.keys(edits)) {
      const [type, name] = k.split("/")
      changed.add(`${type}/${name}`)
    }
    for (const d of added) changed.add(`${d.type}/${d.name}`)

    const payload: CrewDefinition[] = []
    for (const key of changed) {
      const sep = key.indexOf("/")
      const node: TreeNode = { type: key.slice(0, sep) as DefType, name: key.slice(sep + 1) }
      const base = findDoc(node)
      if (!base) continue
      const merged: Record<string, unknown> = { ...base }
      for (const field of FIELD_DEFS[node.type]) {
        const ek = editKey(node.type, node.name, field)
        if (ek in edits) merged[field] = edits[ek]
      }
      payload.push(merged as unknown as CrewDefinition)
    }
    if (payload.length > 0) mutation.mutate(payload)
  }

  const hasChanges = Object.keys(edits).length > 0 || added.length > 0

  if (isLoading) {
    return (
      <div className="h-full overflow-y-auto flex items-center justify-center p-12 text-muted-foreground">
        <Loader2 className="h-5 w-5 animate-spin mr-2" />
        Loading prompts…
      </div>
    )
  }

  const agentItems = allDocs
    .filter((d) => d.type === "agent")
    .map((d) => ({ name: d.name, label: docLabel(d) }))
  const taskItems = allDocs
    .filter((d) => d.type === "task")
    .map((d) => ({ name: d.name, label: docLabel(d) }))
  const philosophyItems = allDocs
    .filter((d) => d.type === "philosophy")
    .map((d) => ({ name: d.name, label: docLabel(d) }))
  const existingPhilosophySlugs = new Set(philosophyItems.map((p) => p.name))

  const selectedDoc = selected ? findDoc(selected) : undefined

  return (
    <div className="h-full flex flex-col">
      {/* header */}
      <div className="flex items-center justify-between px-6 py-4 border-b shrink-0">
        <div>
          <h1 className="text-lg font-semibold">LLM Crew Configuration</h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            Agents, tasks, and training philosophies stored in the database — the single source of truth.
          </p>
        </div>
        <div className="flex items-center gap-3">
          {consolidationStatus === "success" && (
            <span className="flex items-center gap-1.5 text-sm text-green-600">
              <CheckCircle className="h-4 w-4" /> {consolidationSummary}
            </span>
          )}
          {consolidationStatus === "skipped" && (
            <span className="text-sm text-muted-foreground">{consolidationSummary}</span>
          )}
          {consolidationStatus === "failed" && (
            <span className="text-sm text-destructive">{consolidationSummary}</span>
          )}
          <Button size="sm" variant="outline" onClick={triggerConsolidation} disabled={consolidationRunning}>
            {consolidationRunning && <Loader2 className="h-4 w-4 animate-spin mr-2" />}
            Run memory consolidation
          </Button>
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

      {/* body: sidebar + detail */}
      <div className="flex flex-1 min-h-0">
        {/* tree sidebar */}
        <div className="w-56 shrink-0 border-r overflow-y-auto py-3 space-y-1">
          <TreeBranch
            label="Philosophies"
            type="philosophy"
            items={philosophyItems}
            selected={selected}
            onSelect={setSelected}
            extra={
              <button
                type="button"
                onClick={() => setShowAddPhilosophy(true)}
                className="flex items-center gap-1.5 w-full px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground hover:bg-accent/50 rounded-md transition-colors"
              >
                <Plus className="h-3 w-3" />
                Add philosophy
              </button>
            }
          />
          <TreeBranch
            label="Agents"
            type="agent"
            items={agentItems}
            selected={selected}
            onSelect={setSelected}
          />
          <TreeBranch
            label="Tasks"
            type="task"
            items={taskItems}
            selected={selected}
            onSelect={setSelected}
          />
        </div>

        {/* detail panel */}
        <div className="flex-1 min-w-0">
          {selectedDoc ? (
            <DetailPanel
              doc={selectedDoc}
              fields={FIELD_DEFS[selectedDoc.type]}
              value={(field) => fieldValue(selectedDoc, field)}
              onChange={(field, value) => handleChange(selectedDoc, field, value)}
              onImport={(values) => handleImport(selectedDoc, values)}
            />
          ) : (
            <div className="h-full flex items-center justify-center text-sm text-muted-foreground">
              {selected ? "Not found." : "Select an item from the tree to edit."}
            </div>
          )}
        </div>
      </div>

      <Dialog open={showAddPhilosophy} onOpenChange={(open) => { if (!open) setShowAddPhilosophy(false) }}>
        {showAddPhilosophy && (
          <AddPhilosophyDialog
            onAdd={handleAddPhilosophy}
            onClose={() => setShowAddPhilosophy(false)}
            existingSlugs={existingPhilosophySlugs}
          />
        )}
      </Dialog>
    </div>
  )
}
