import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Loader2 } from "lucide-react"
import {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from "../components/ui/accordion"
import { cn } from "../lib/utils"
import {
  fetchPromptLogRun,
  fetchPromptLogRuns,
  type PromptLogEntry,
  type PromptLogRunSummary,
} from "../api/promptLogs"

function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleString()
}

function RoleBadge({ role }: { role: string }) {
  const colors: Record<string, string> = {
    system: "bg-slate-200 text-slate-800",
    user: "bg-blue-100 text-blue-800",
    assistant: "bg-emerald-100 text-emerald-800",
    tool: "bg-amber-100 text-amber-800",
  }
  return (
    <span
      className={cn(
        "inline-block rounded px-2 py-0.5 text-xs font-medium uppercase tracking-wide",
        colors[role] ?? "bg-muted text-muted-foreground"
      )}
    >
      {role}
    </span>
  )
}

function PromptCallDetail({ entry, index }: { entry: PromptLogEntry; index: number }) {
  return (
    <AccordionItem value={`call-${index}`}>
      <AccordionTrigger>
        <div className="flex flex-wrap items-center gap-2 text-left">
          <span className="font-mono text-sm text-muted-foreground">#{index + 1}</span>
          <span className="font-medium">{entry.agent_role ?? "Unknown agent"}</span>
          {entry.task_name && (
            <span className="text-sm text-muted-foreground">— {entry.task_name}</span>
          )}
          {entry.model && (
            <span className="text-xs rounded bg-muted px-2 py-0.5 font-mono">{entry.model}</span>
          )}
        </div>
      </AccordionTrigger>
      <AccordionContent className="px-6 pb-4 space-y-3">
        <div className="space-y-2">
          {entry.messages.map((msg, i) => (
            <div key={i} className="rounded-md border bg-background p-3">
              <div className="mb-1.5 flex items-center justify-between">
                <RoleBadge role={msg.role} />
              </div>
              <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed">
                {typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content, null, 2)}
              </pre>
            </div>
          ))}
        </div>
        {entry.response && (
          <div className="rounded-md border-2 border-emerald-200 bg-emerald-50 p-3">
            <div className="mb-1.5">
              <span className="inline-block rounded px-2 py-0.5 text-xs font-medium uppercase tracking-wide bg-emerald-200 text-emerald-900">
                response
              </span>
            </div>
            <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed">
              {entry.response}
            </pre>
          </div>
        )}
      </AccordionContent>
    </AccordionItem>
  )
}

function RunDetail({ runId }: { runId: string }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["prompt-log-run", runId],
    queryFn: () => fetchPromptLogRun(runId),
  })

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-muted-foreground p-6">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading run…
      </div>
    )
  }
  if (error) {
    return <p className="p-6 text-destructive">Failed to load run: {(error as Error).message}</p>
  }
  if (!data || data.length === 0) {
    return <p className="p-6 text-muted-foreground">No calls captured for this run.</p>
  }

  return (
    <Accordion type="multiple" className="space-y-2">
      {data.map((entry, i) => (
        <PromptCallDetail key={i} entry={entry} index={i} />
      ))}
    </Accordion>
  )
}

function RunListItem({
  run,
  selected,
  onSelect,
}: {
  run: PromptLogRunSummary
  selected: boolean
  onSelect: () => void
}) {
  return (
    <button
      onClick={onSelect}
      className={cn(
        "w-full rounded-md border px-3 py-2.5 text-left text-sm transition-colors",
        "hover:bg-accent hover:text-accent-foreground",
        selected && "bg-accent text-accent-foreground border-accent-foreground/20"
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="font-medium">{run.crew_name}</span>
        <span className="text-xs text-muted-foreground">{run.call_count} calls</span>
      </div>
      <div className="text-xs text-muted-foreground">
        Athlete {run.athlete_id} · {formatTimestamp(run.started_at)}
      </div>
      <div className="mt-1 flex flex-wrap gap-1">
        {run.agent_roles.filter(Boolean).map((role) => (
          <span key={role} className="rounded bg-muted px-1.5 py-0.5 text-[11px]">
            {role}
          </span>
        ))}
      </div>
    </button>
  )
}

export function InspectPage() {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const { data: runs, isLoading, error } = useQuery({
    queryKey: ["prompt-log-runs"],
    queryFn: () => fetchPromptLogRuns(),
  })

  return (
    <div className="mx-auto max-w-6xl p-6">
      <div className="mb-4">
        <h1 className="text-2xl font-bold">Inspect</h1>
        <p className="text-sm text-muted-foreground">
          Browse the literal prompts and responses sent to the LLM during recent crew runs —
          a debugging view for agents, tasks, and training philosophy composition.
          Admin-only once user permissions land.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[320px_1fr]">
        <div className="space-y-2">
          {isLoading && (
            <div className="flex items-center gap-2 text-muted-foreground p-3">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading runs…
            </div>
          )}
          {error && <p className="text-destructive p-3">Failed to load runs: {(error as Error).message}</p>}
          {runs && runs.length === 0 && (
            <p className="text-muted-foreground p-3 text-sm">
              No crew runs captured yet. Trigger a daily analysis to see prompts here.
            </p>
          )}
          {runs?.map((run) => (
            <RunListItem
              key={run.run_id}
              run={run}
              selected={run.run_id === selectedRunId}
              onSelect={() => setSelectedRunId(run.run_id)}
            />
          ))}
        </div>

        <div className="rounded-lg border min-h-[200px]">
          {selectedRunId ? (
            <RunDetail runId={selectedRunId} />
          ) : (
            <p className="p-6 text-muted-foreground text-sm">
              Select a run on the left to inspect its prompts and responses.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
