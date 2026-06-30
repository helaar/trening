import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Loader2, Wrench, Brain } from "lucide-react"
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
  type RunUsage,
} from "../api/promptLogs"

function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleString()
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}

function formatCost(usd: number | null): string {
  if (usd == null) return "n/a"
  if (usd > 0 && usd < 0.01) return "<$0.01"
  return `$${usd.toFixed(usd < 1 ? 4 : 2)}`
}

function formatToolArgs(args: PromptLogEntry["tool_args"]): string {
  if (args == null) return ""
  return typeof args === "string" ? args : JSON.stringify(args, null, 2)
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

// ── grouping: consecutive entries by (agent_role, task_name) form a "step" in the call hierarchy ──

interface EntryGroup {
  agentRole: string | null
  taskName: string | null
  entries: { entry: PromptLogEntry; step: number }[]
}

function groupEntries(entries: PromptLogEntry[]): EntryGroup[] {
  const groups: EntryGroup[] = []
  entries.forEach((entry, i) => {
    const last = groups[groups.length - 1]
    if (last && last.agentRole === entry.agent_role && last.taskName === entry.task_name) {
      last.entries.push({ entry, step: i + 1 })
    } else {
      groups.push({ agentRole: entry.agent_role, taskName: entry.task_name, entries: [{ entry, step: i + 1 }] })
    }
  })
  return groups
}

function LlmCallContent({ entry }: { entry: PromptLogEntry }) {
  return (
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
    </div>
  )
}

function ToolCallContent({ entry }: { entry: PromptLogEntry }) {
  const args = formatToolArgs(entry.tool_args)
  return (
    <div className="space-y-2">
      {args && (
        <div className="rounded-md border bg-background p-3">
          <div className="mb-1.5">
            <span className="inline-block rounded px-2 py-0.5 text-xs font-medium uppercase tracking-wide bg-amber-100 text-amber-800">
              args
            </span>
          </div>
          <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed">{args}</pre>
        </div>
      )}
      {entry.tool_output && (
        <div className="rounded-md border-2 border-amber-200 bg-amber-50 p-3">
          <div className="mb-1.5">
            <span className="inline-block rounded px-2 py-0.5 text-xs font-medium uppercase tracking-wide bg-amber-200 text-amber-900">
              output
            </span>
          </div>
          <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed">
            {entry.tool_output}
          </pre>
        </div>
      )}
      {entry.tool_error && (
        <div className="rounded-md border-2 border-destructive/30 bg-destructive/5 p-3">
          <div className="mb-1.5">
            <span className="inline-block rounded px-2 py-0.5 text-xs font-medium uppercase tracking-wide bg-destructive/20 text-destructive">
              error
            </span>
          </div>
          <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-destructive">
            {entry.tool_error}
          </pre>
        </div>
      )}
    </div>
  )
}

function PromptCallDetail({ entry, step }: { entry: PromptLogEntry; step: number }) {
  const isTool = entry.kind === "tool_call"
  return (
    <AccordionItem value={`step-${step}`}>
      <AccordionTrigger>
        <div className="flex flex-wrap items-center gap-2 text-left">
          <span className="font-mono text-sm text-muted-foreground">#{step}</span>
          {isTool ? (
            <Wrench className="h-3.5 w-3.5 text-amber-600 shrink-0" />
          ) : (
            <Brain className="h-3.5 w-3.5 text-emerald-600 shrink-0" />
          )}
          {isTool ? (
            <span className="font-medium font-mono text-sm">{entry.tool_name ?? "tool"}</span>
          ) : (
            <span className="font-medium">LLM call</span>
          )}
          {entry.model && (
            <span className="text-xs rounded bg-muted px-2 py-0.5 font-mono">{entry.model}</span>
          )}
          {entry.call_type && entry.call_type !== "llm_call" && (
            <span className="text-xs rounded bg-muted px-2 py-0.5">{entry.call_type}</span>
          )}
        </div>
      </AccordionTrigger>
      <AccordionContent className="px-6 pb-4">
        {isTool ? <ToolCallContent entry={entry} /> : <LlmCallContent entry={entry} />}
      </AccordionContent>
    </AccordionItem>
  )
}

function GroupSection({ group }: { group: EntryGroup }) {
  const first = group.entries[0].step
  const last = group.entries[group.entries.length - 1].step
  const stepRange = first === last ? `#${first}` : `#${first}–${last}`
  return (
    <div className="rounded-lg border">
      <div className="flex flex-wrap items-center gap-2 border-b bg-muted/40 px-4 py-2.5">
        <span className="font-semibold">{group.agentRole ?? "Unknown agent"}</span>
        {group.taskName && <span className="text-sm text-muted-foreground">— {group.taskName}</span>}
        <span className="ml-auto font-mono text-xs text-muted-foreground">{stepRange}</span>
      </div>
      <Accordion type="multiple" className="space-y-1.5 p-2">
        {group.entries.map(({ entry, step }) => (
          <PromptCallDetail key={step} entry={entry} step={step} />
        ))}
      </Accordion>
    </div>
  )
}

function UsageSummary({ usage }: { usage: RunUsage }) {
  return (
    <div className="rounded-lg border bg-muted/30">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-4 py-2.5 text-sm">
        <span className="font-semibold">Token usage &amp; cost</span>
        <span className="text-muted-foreground">
          {formatTokens(usage.total_tokens)} tokens
        </span>
        <span className="text-muted-foreground">
          {formatTokens(usage.prompt_tokens)} in · {formatTokens(usage.completion_tokens)} out
          {usage.cached_prompt_tokens > 0 && (
            <> · {formatTokens(usage.cached_prompt_tokens)} cached</>
          )}
        </span>
        <span className="text-muted-foreground">{usage.successful_requests} LLM requests</span>
        <span className="ml-auto font-mono font-semibold">{formatCost(usage.cost_usd)}</span>
      </div>
      {usage.per_agent.length > 0 && (
        <div className="border-t">
          <table className="w-full text-xs">
            <thead className="text-muted-foreground">
              <tr className="border-b">
                <th className="px-4 py-1.5 text-left font-medium">Agent</th>
                <th className="px-4 py-1.5 text-left font-medium">Model</th>
                <th className="px-2 py-1.5 text-right font-medium">In</th>
                <th className="px-2 py-1.5 text-right font-medium">Out</th>
                <th className="px-2 py-1.5 text-right font-medium">Total</th>
                <th className="px-4 py-1.5 text-right font-medium">Cost</th>
              </tr>
            </thead>
            <tbody>
              {usage.per_agent.map((a, i) => (
                <tr key={i} className="border-b last:border-0">
                  <td className="px-4 py-1.5">{a.agent_role ?? "—"}</td>
                  <td className="px-4 py-1.5 font-mono text-muted-foreground">{a.model ?? "—"}</td>
                  <td className="px-2 py-1.5 text-right font-mono">{formatTokens(a.prompt_tokens)}</td>
                  <td className="px-2 py-1.5 text-right font-mono">{formatTokens(a.completion_tokens)}</td>
                  <td className="px-2 py-1.5 text-right font-mono">{formatTokens(a.total_tokens)}</td>
                  <td className="px-4 py-1.5 text-right font-mono">{formatCost(a.cost_usd)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function RunDetail({ run }: { run: PromptLogRunSummary }) {
  const runId = run.run_id
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

  const groups = groupEntries(data)

  return (
    <div className="space-y-3 p-3">
      {run.usage ? (
        <UsageSummary usage={run.usage} />
      ) : (
        <div className="rounded-lg border bg-muted/30 px-4 py-2.5 text-sm text-muted-foreground">
          Token usage &amp; cost not recorded for this run (only runs from after this
          feature shipped have usage data).
        </div>
      )}
      <p className="text-xs text-muted-foreground">
        {data.length} calls across {groups.length} agent/task steps, in chronological order —
        each section is one agent working on one task; expand a step to see its messages, response,
        or tool input/output.
      </p>
      {groups.map((group, i) => (
        <GroupSection key={i} group={group} />
      ))}
    </div>
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
      {run.usage && (
        <div className="mt-1 flex items-center justify-between gap-2 text-xs text-muted-foreground">
          <span>{formatTokens(run.usage.total_tokens)} tokens</span>
          <span className="font-mono">{formatCost(run.usage.cost_usd)}</span>
        </div>
      )}
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

  const selectedRun = runs?.find((r) => r.run_id === selectedRunId) ?? null

  return (
    <div className="mx-auto max-w-6xl p-6">
      <div className="mb-4">
        <h1 className="text-2xl font-bold">Inspect</h1>
        <p className="text-sm text-muted-foreground">
          Trace the literal call hierarchy of recent crew runs — which agent ran which task,
          when it called the LLM vs. a tool, and exactly what was sent and returned.
          A debugging view for agents, tasks, and training philosophy composition.
          Admin-only once user permissions land.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[320px_1fr] lg:items-start">
        <div className="space-y-2 overflow-y-auto max-h-[calc(100vh-12rem)] pr-1">
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

        <div className="rounded-lg border min-h-[200px] max-h-[calc(100vh-12rem)] overflow-y-auto">
          {selectedRun ? (
            <RunDetail run={selectedRun} />
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
