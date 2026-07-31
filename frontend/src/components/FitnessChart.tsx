import type { FitnessPoint } from "../api/fitness"

const WIDTH = 600
const LOAD_HEIGHT = 160
const TSB_HEIGHT = 60
const PADDING = 8

function scaleX(index: number, count: number): number {
  if (count <= 1) return PADDING
  return PADDING + (index / (count - 1)) * (WIDTH - PADDING * 2)
}

function scaleY(value: number, height: number, min: number, max: number): number {
  const range = max - min || 1
  return PADDING + (1 - (value - min) / range) * (height - PADDING * 2)
}

function buildPath(values: (number | null)[], height: number, min: number, max: number): string {
  let path = ""
  let started = false
  values.forEach((v, i) => {
    if (v == null) {
      started = false
      return
    }
    const x = scaleX(i, values.length)
    const y = scaleY(v, height, min, max)
    path += `${started ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)} `
    started = true
  })
  return path.trim()
}

function minMax(values: (number | null)[]): [number, number] {
  const nums = values.filter((v): v is number => v != null)
  if (nums.length === 0) return [0, 1]
  return [Math.min(...nums), Math.max(...nums)]
}

export function FitnessChart({ data }: { data: FitnessPoint[] }) {
  const hasData = data.some((d) => d.ctl != null || d.atl != null)

  if (!hasData) {
    return (
      <div className="rounded-md border bg-gray-50 px-4 py-6 text-center text-sm text-muted-foreground">
        No Intervals.icu wellness data available.
      </div>
    )
  }

  const ctlValues = data.map((d) => d.ctl)
  const atlValues = data.map((d) => d.atl)
  const tsbValues = data.map((d) => d.tsb)

  const [loadMinRaw, loadMax] = minMax([...ctlValues, ...atlValues])
  const loadMin = Math.min(0, loadMinRaw)

  const [tsbMinRaw, tsbMaxRaw] = minMax(tsbValues)
  const tsbAbsMax = Math.max(Math.abs(tsbMinRaw), Math.abs(tsbMaxRaw), 5)

  const ctlPath = buildPath(ctlValues, LOAD_HEIGHT, loadMin, loadMax)
  const atlPath = buildPath(atlValues, LOAD_HEIGHT, loadMin, loadMax)
  const tsbZeroY = scaleY(0, TSB_HEIGHT, -tsbAbsMax, tsbAbsMax)
  const barWidth = Math.max((WIDTH - PADDING * 2) / data.length - 1, 1)

  const latest = [...data].reverse().find((d) => d.ctl != null || d.atl != null)

  return (
    <div className="space-y-3 rounded-md border px-4 py-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold text-gray-800">Fitness (CTL/ATL/TSB)</h4>
        {latest && (
          <span className="text-xs text-gray-400">
            {latest.date}
            {latest.ctl != null ? ` · CTL ${latest.ctl.toFixed(0)}` : ""}
            {latest.atl != null ? ` · ATL ${latest.atl.toFixed(0)}` : ""}
            {latest.tsb != null
              ? ` · TSB ${latest.tsb > 0 ? "+" : ""}${latest.tsb.toFixed(0)}`
              : ""}
          </span>
        )}
      </div>

      <svg viewBox={`0 0 ${WIDTH} ${LOAD_HEIGHT}`} className="w-full" preserveAspectRatio="none">
        <path d={ctlPath} fill="none" stroke="#2563eb" strokeWidth={2} />
        <path d={atlPath} fill="none" stroke="#f59e0b" strokeWidth={2} />
      </svg>
      <div className="flex items-center gap-4 text-xs text-gray-600">
        <span className="inline-flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-blue-600" /> CTL (fitness)
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-amber-500" /> ATL (fatigue)
        </span>
      </div>

      <svg viewBox={`0 0 ${WIDTH} ${TSB_HEIGHT}`} className="w-full" preserveAspectRatio="none">
        <line x1={0} y1={tsbZeroY} x2={WIDTH} y2={tsbZeroY} stroke="#d1d5db" strokeWidth={1} />
        {data.map((d, i) => {
          if (d.tsb == null) return null
          const y = scaleY(d.tsb, TSB_HEIGHT, -tsbAbsMax, tsbAbsMax)
          const top = Math.min(y, tsbZeroY)
          const barHeight = Math.max(Math.abs(y - tsbZeroY), 1)
          return (
            <rect
              key={d.date}
              x={scaleX(i, data.length) - barWidth / 2}
              y={top}
              width={barWidth}
              height={barHeight}
              fill={d.tsb >= 0 ? "#16a34a" : "#dc2626"}
            />
          )
        })}
      </svg>
      <p className="text-[10px] text-gray-400">
        TSB (form) — green above the line is fresher, red below is more fatigued.
      </p>
    </div>
  )
}
