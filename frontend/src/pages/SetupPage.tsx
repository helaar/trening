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
import { fetchPrompts, savePrompts, type PromptConfig } from "../api/prompts"
import { createMemoryConsolidationTask, getTaskStatus } from "../api/tasks"

// ── data helpers ──────────────────────────────────────────────────────────────

const PHILOSOPHY_SUB_KEYS = ["name", "intensity_targets", "coach_guidance", "analyst_guidance"]

function isNumeric(s: string): boolean {
  return /^\d+$/.test(s)
}

function parsePhilosophyGroups(prompts: PromptConfig[]): Map<string, PromptConfig[]> {
  const groups = new Map<string, PromptConfig[]>()
  for (const p of prompts) {
    if (!p.key.startsWith("philosophy.")) continue
    const parts = p.key.split(".")
    if (parts.length < 3) continue
    const slug = parts[1]
    // skip numeric segments (athlete selections) and leaf sub-keys used as group names
    if (isNumeric(slug) || PHILOSOPHY_SUB_KEYS.includes(slug)) continue
    const list = groups.get(slug) ?? []
    list.push(p)
    groups.set(slug, list)
  }
  return groups
}

function groupPrompts(prompts: PromptConfig[], prefix: "agents" | "tasks"): Map<string, PromptConfig[]> {
  const groups = new Map<string, PromptConfig[]>()
  for (const p of prompts) {
    if (!p.key.startsWith(`${prefix}.`)) continue
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
  const name = (parts[1] ?? parts[0]).replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
  return name
}

function philosophyLabel(slug: string): string {
  return slug.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

function fieldLabel(key: string): string {
  const parts = key.split(".")
  return parts[parts.length - 1].replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

function slugify(name: string): string {
  return name.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "")
}

// ── tree node types ──────────────────────────────────────────────────────────

type TreeNode =
  | { branch: "philosophies"; key: string }
  | { branch: "agents"; key: string }
  | { branch: "tasks"; key: string }

// ── copy button ───────────────────────────────────────────────────────────────

function GroupCopyButton({
  items,
  currentValues,
}: {
  items: PromptConfig[]
  currentValues: (p: PromptConfig) => string
}) {
  const [copied, setCopied] = useState(false)
  function handleCopy() {
    const payload = items.map((p) => ({ key: p.key, value: currentValues(p) }))
    navigator.clipboard.writeText(JSON.stringify(payload, null, 2))
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

// ── import modal ──────────────────────────────────────────────────────────────

interface ImportModalProps {
  label: string
  onImport: (updates: Record<string, string>) => void
  onClose: () => void
}

function ImportModal({ label, onImport, onClose }: ImportModalProps) {
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
        <DialogTitle>Import — {label}</DialogTitle>
        <DialogDescription>Paste the JSON array back from the LLM.</DialogDescription>
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

// ── tree sidebar ──────────────────────────────────────────────────────────────

interface TreeBranchProps {
  label: string
  items: { key: string; label: string }[]
  selected: TreeNode | null
  branch: TreeNode["branch"]
  onSelect: (node: TreeNode) => void
  extra?: React.ReactNode
}

function TreeBranch({ label, items, selected, branch, onSelect, extra }: TreeBranchProps) {
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
          {items.map(({ key, label: itemLabel }) => {
            const isSelected = selected?.branch === branch && selected.key === key
            return (
              <button
                key={key}
                type="button"
                onClick={() => onSelect({ branch, key })}
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

// ── detail panel ──────────────────────────────────────────────────────────────

interface DetailPanelProps {
  items: PromptConfig[]
  label: string
  currentValue: (p: PromptConfig) => string
  onChange: (key: string, value: string) => void
  onImport: (updates: Record<string, string>) => void
}

function DetailPanel({ items, label, currentValue, onChange, onImport }: DetailPanelProps) {
  const [showImport, setShowImport] = useState(false)

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-6 py-4 border-b">
        <h2 className="text-sm font-semibold">{label}</h2>
        <div className="flex items-center gap-1">
          <GroupCopyButton items={items} currentValues={currentValue} />
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
        {items.map((p) => (
          <div key={p.key} className="space-y-1.5">
            <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              {fieldLabel(p.key)}
            </Label>
            <Textarea
              value={currentValue(p)}
              onChange={(e) => onChange(p.key, e.target.value)}
              rows={Math.min(Math.max(currentValue(p).split("\n").length + 1, 3), 12)}
              className="font-mono text-xs resize-y"
            />
          </div>
        ))}
      </div>
      <Dialog open={showImport} onOpenChange={(open) => { if (!open) setShowImport(false) }}>
        {showImport && (
          <ImportModal
            label={label}
            onImport={onImport}
            onClose={() => setShowImport(false)}
          />
        )}
      </Dialog>
    </div>
  )
}

// ── add philosophy dialog ────────────────────────────────────────────────────

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

// ── page ──────────────────────────────────────────────────────────────────────

export function SetupPage() {
  const queryClient = useQueryClient()
  const { data: prompts, isLoading } = useQuery({
    queryKey: ["admin-prompts"],
    queryFn: fetchPrompts,
  })

  const [edits, setEdits] = useState<Record<string, string>>({})
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

  function currentValueByKey(key: string): string {
    const p = (prompts ?? []).find((x) => x.key === key)
    return edits[key] ?? p?.value ?? ""
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

  function handleAddPhilosophy(slug: string, name: string) {
    const newEdits: Record<string, string> = {}
    for (const sub of PHILOSOPHY_SUB_KEYS) {
      const key = `philosophy.${slug}.${sub}`
      newEdits[key] = sub === "name" ? name : ""
    }
    setEdits((prev) => ({ ...prev, ...newEdits }))
    setSelected({ branch: "philosophies", key: slug })
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

  const philosophyGroups = parsePhilosophyGroups(prompts ?? [])
  const agentGroups = groupPrompts(prompts ?? [], "agents")
  const taskGroups = groupPrompts(prompts ?? [], "tasks")

  // include any newly added (edits-only) philosophy slugs
  const editPhilosophySlugs = Object.keys(edits)
    .filter((k) => k.startsWith("philosophy."))
    .map((k) => k.split(".")[1])
    .filter((s) => s && !isNumeric(s) && !PHILOSOPHY_SUB_KEYS.includes(s))
  const allPhilosophySlugs = new Set([...philosophyGroups.keys(), ...editPhilosophySlugs])

  // build items for the selected detail panel
  let detailItems: PromptConfig[] = []
  let detailLabel = ""

  if (selected) {
    if (selected.branch === "philosophies") {
      const slug = selected.key
      // merge persisted entries with any edit-only keys
      const persistedItems = philosophyGroups.get(slug) ?? []
      const persistedKeys = new Set(persistedItems.map((p) => p.key))
      const editOnlyItems: PromptConfig[] = Object.keys(edits)
        .filter((k) => k.startsWith(`philosophy.${slug}.`) && !persistedKeys.has(k))
        .map((k) => ({ key: k, value: "", updated_at: "" }))
      detailItems = [...persistedItems, ...editOnlyItems]
      // ensure all sub-keys are present
      const presentSubs = new Set(detailItems.map((p) => p.key.split(".")[2]))
      for (const sub of PHILOSOPHY_SUB_KEYS) {
        if (!presentSubs.has(sub)) {
          detailItems.push({ key: `philosophy.${slug}.${sub}`, value: "", updated_at: "" })
        }
      }
      detailLabel = philosophyLabel(slug)
    } else if (selected.branch === "agents") {
      detailItems = agentGroups.get(selected.key) ?? []
      detailLabel = groupLabel(selected.key)
    } else {
      detailItems = taskGroups.get(selected.key) ?? []
      detailLabel = groupLabel(selected.key)
    }
  }

  const philosophyTreeItems = Array.from(allPhilosophySlugs).map((slug) => ({
    key: slug,
    label: philosophyLabel(slug),
  }))
  const agentTreeItems = Array.from(agentGroups.keys()).map((key) => ({
    key,
    label: groupLabel(key),
  }))
  const taskTreeItems = Array.from(taskGroups.keys()).map((key) => ({
    key,
    label: groupLabel(key),
  }))

  return (
    <div className="h-full flex flex-col">
      {/* header */}
      <div className="flex items-center justify-between px-6 py-4 border-b shrink-0">
        <div>
          <h1 className="text-lg font-semibold">LLM Prompt Configuration</h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            Overrides stored in the database. Clearing a field and saving restores the YAML default.
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
            branch="philosophies"
            items={philosophyTreeItems}
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
            branch="agents"
            items={agentTreeItems}
            selected={selected}
            onSelect={setSelected}
          />
          <TreeBranch
            label="Tasks"
            branch="tasks"
            items={taskTreeItems}
            selected={selected}
            onSelect={setSelected}
          />
        </div>

        {/* detail panel */}
        <div className="flex-1 min-w-0">
          {selected && detailItems.length > 0 ? (
            <DetailPanel
              items={detailItems}
              label={detailLabel}
              currentValue={currentValue}
              onChange={handleChange}
              onImport={handleImport}
            />
          ) : (
            <div className="h-full flex items-center justify-center text-sm text-muted-foreground">
              {selected ? "No fields found." : "Select an item from the tree to edit."}
            </div>
          )}
        </div>
      </div>

      <Dialog open={showAddPhilosophy} onOpenChange={(open) => { if (!open) setShowAddPhilosophy(false) }}>
        {showAddPhilosophy && (
          <AddPhilosophyDialog
            onAdd={handleAddPhilosophy}
            onClose={() => setShowAddPhilosophy(false)}
            existingSlugs={allPhilosophySlugs}
          />
        )}
      </Dialog>
    </div>
  )
}
