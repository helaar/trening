import { useRef, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Loader2, CheckCircle, Clipboard, ClipboardCheck, Upload } from "lucide-react"
import { Button } from "../components/ui/button"
import { Textarea } from "../components/ui/textarea"
import {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from "../components/ui/accordion"
import { fetchPrompts, savePrompts, type PromptConfig, type PromptConfigUpdate } from "../api/prompts"

// Group keys by their top-level prefix (e.g. "agents.daily_coach.backstory" -> "agents.daily_coach")
function groupPrompts(prompts: PromptConfig[]): Map<string, PromptConfig[]> {
  const groups = new Map<string, PromptConfig[]>()
  for (const p of prompts) {
    const parts = p.key.split(".")
    // Group by first two segments (e.g. "agents.daily_coach") when available
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

export function SetupPage() {
  const queryClient = useQueryClient()
  const { data: prompts, isLoading } = useQuery({
    queryKey: ["admin-prompts"],
    queryFn: fetchPrompts,
  })

  const [edits, setEdits] = useState<Record<string, string>>({})
  const [saved, setSaved] = useState(false)
  const [copied, setCopied] = useState(false)
  const importRef = useRef<HTMLInputElement>(null)

  function handleExport() {
    const current = (prompts ?? []).map((p) => ({
      key: p.key,
      value: edits[p.key] ?? p.value,
    }))
    navigator.clipboard.writeText(JSON.stringify(current, null, 2))
    setCopied(true)
    setTimeout(() => setCopied(false), 3000)
  }

  function handleImportFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      try {
        applyImport(ev.target?.result as string)
      } catch {
        alert("Invalid JSON — could not import prompts.")
      }
    }
    reader.readAsText(file)
    e.target.value = ""
  }

  async function handleImportClipboard() {
    try {
      const text = await navigator.clipboard.readText()
      applyImport(text)
    } catch {
      alert("Could not read clipboard. Paste the JSON into a .json file and use the file import instead.")
    }
  }

  function applyImport(json: string) {
    const parsed: unknown = JSON.parse(json)
    if (!Array.isArray(parsed)) throw new Error("Expected array")
    const incoming = parsed as PromptConfigUpdate[]
    const next: Record<string, string> = { ...edits }
    for (const item of incoming) {
      if (typeof item.key === "string" && typeof item.value === "string") {
        next[item.key] = item.value
      }
    }
    setEdits(next)
  }

  const mutation = useMutation({
    mutationFn: savePrompts,
    onSuccess: (updated) => {
      queryClient.setQueryData<PromptConfig[]>(["admin-prompts"], (prev) => {
        if (!prev) return updated
        const byKey = new Map(updated.map((p) => [p.key, p]))
        return prev.map((p) => byKey.get(p.key) ?? p)
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
        <div className="flex items-center gap-2">
          {saved && (
            <span className="flex items-center gap-1.5 text-sm text-green-600">
              <CheckCircle className="h-4 w-4" /> Saved
            </span>
          )}
          <Button variant="outline" size="sm" onClick={handleExport} title="Copy all prompts as JSON">
            {copied ? <ClipboardCheck className="h-4 w-4 mr-1.5 text-green-600" /> : <Clipboard className="h-4 w-4 mr-1.5" />}
            {copied ? "Copied!" : "Export"}
          </Button>
          <Button variant="outline" size="sm" onClick={handleImportClipboard} title="Paste JSON from clipboard">
            <Upload className="h-4 w-4 mr-1.5" />
            Import
          </Button>
          <input ref={importRef} type="file" accept=".json" className="hidden" onChange={handleImportFile} />
          <Button size="sm" onClick={handleSave} disabled={!hasChanges || mutation.isPending}>
            {mutation.isPending && <Loader2 className="h-4 w-4 animate-spin mr-2" />}
            Save changes
          </Button>
        </div>
      </div>

      <Accordion type="multiple" className="space-y-2">
        {Array.from(groups.entries()).map(([groupKey, items]) => (
          <AccordionItem key={groupKey} value={groupKey} className="border rounded-lg px-4">
            <AccordionTrigger className="text-sm font-medium py-3">
              {groupLabel(groupKey)}
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
    </div>
  )
}
