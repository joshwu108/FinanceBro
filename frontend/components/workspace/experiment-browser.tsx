"use client"

import { useState, useMemo } from "react"
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table"
import { RefreshCw, ArrowUpDown, ArrowUp, ArrowDown, Check, X, FlaskConical } from "lucide-react"
import type { ExperimentSummary } from "@/lib/types"

// ── Types ────────────────────────────────────────────────────────────────────

interface ExperimentBrowserProps {
  experiments: ExperimentSummary[]
  onRefresh: () => void
  loading?: boolean
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatPct(v: number): string {
  return `${(v * 100).toFixed(2)}%`
}

function formatNum(v: number, decimals = 2): string {
  return v.toFixed(decimals)
}

function valueColor(v: number): string {
  if (v > 0) return "#00FF9C"
  if (v < 0) return "#FF4D4D"
  return "#E6EDF3"
}

function overfittingColor(score: number): {
  bg: string
  text: string
  border: string
} {
  if (score < 0.3)
    return {
      bg: "bg-[#00FF9C]/10",
      text: "text-[#00FF9C]",
      border: "border-[#00FF9C]/30",
    }
  if (score <= 0.6)
    return {
      bg: "bg-[#FFD60A]/10",
      text: "text-[#FFD60A]",
      border: "border-[#FFD60A]/30",
    }
  return {
    bg: "bg-[#FF4D4D]/10",
    text: "text-[#FF4D4D]",
    border: "border-[#FF4D4D]/30",
  }
}

// ── Sort Header ──────────────────────────────────────────────────────────────

function SortHeader({
  label,
  sorted,
}: {
  label: string
  sorted: false | "asc" | "desc"
}) {
  return (
    <span className="flex items-center gap-1 select-none">
      {label}
      {sorted === false && (
        <ArrowUpDown className="w-2.5 h-2.5 opacity-40" />
      )}
      {sorted === "asc" && <ArrowUp className="w-2.5 h-2.5" />}
      {sorted === "desc" && <ArrowDown className="w-2.5 h-2.5" />}
    </span>
  )
}

// ── Column Definitions ───────────────────────────────────────────────────────

function buildColumns(): ColumnDef<ExperimentSummary, unknown>[] {
  return [
    {
      accessorKey: "date",
      header: ({ column }) => (
        <SortHeader label="Date" sorted={column.getIsSorted()} />
      ),
      cell: ({ getValue }) => (
        <span className="text-[#E6EDF3]">{getValue<string>()}</span>
      ),
      sortingFn: "alphanumeric",
    },
    {
      accessorKey: "experiment_id",
      header: "Exp ID",
      cell: ({ getValue }) => (
        <span className="text-[#8B949E] font-mono text-[10px]">
          {getValue<string>()}
        </span>
      ),
      enableSorting: false,
    },
    {
      accessorKey: "symbols",
      header: "Symbols",
      cell: ({ getValue }) => {
        const symbols = getValue<string[]>()
        return (
          <div className="flex flex-wrap gap-1">
            {symbols.map((s) => (
              <span
                key={s}
                className="bg-[#1E2A38] text-[#4CC9F0] text-[9px] px-1.5 py-0.5 rounded-sm font-mono"
              >
                {s}
              </span>
            ))}
          </div>
        )
      },
      enableSorting: false,
    },
    {
      accessorKey: "model",
      header: "Model",
      cell: ({ getValue }) => (
        <span className="font-mono text-[#E6EDF3]">{getValue<string>()}</span>
      ),
      enableSorting: false,
    },
    {
      accessorFn: (row) => row.metrics.sharpe,
      id: "sharpe",
      header: ({ column }) => (
        <SortHeader label="Sharpe" sorted={column.getIsSorted()} />
      ),
      cell: ({ getValue }) => {
        const v = getValue<number>()
        return (
          <span className="tabular-nums" style={{ color: valueColor(v) }}>
            {formatNum(v)}
          </span>
        )
      },
    },
    {
      accessorFn: (row) => row.metrics.max_drawdown,
      id: "max_dd",
      header: ({ column }) => (
        <SortHeader label="Max DD" sorted={column.getIsSorted()} />
      ),
      cell: ({ getValue }) => (
        <span className="tabular-nums text-[#FF4D4D]">
          {formatPct(getValue<number>())}
        </span>
      ),
    },
    {
      accessorFn: (row) => row.metrics.total_return,
      id: "total_return",
      header: ({ column }) => (
        <SortHeader label="Return" sorted={column.getIsSorted()} />
      ),
      cell: ({ getValue }) => {
        const v = getValue<number>()
        return (
          <span className="tabular-nums" style={{ color: valueColor(v) }}>
            {formatPct(v)}
          </span>
        )
      },
    },
    {
      accessorFn: (row) => row.metrics.win_rate,
      id: "win_rate",
      header: ({ column }) => (
        <SortHeader label="Win Rate" sorted={column.getIsSorted()} />
      ),
      cell: ({ getValue }) => (
        <span className="tabular-nums text-[#4CC9F0]">
          {formatPct(getValue<number>())}
        </span>
      ),
    },
    {
      accessorKey: "overfitting_score",
      header: ({ column }) => (
        <SortHeader label="Overfit" sorted={column.getIsSorted()} />
      ),
      cell: ({ getValue }) => {
        const score = getValue<number>()
        const colors = overfittingColor(score)
        return (
          <span
            className={`${colors.bg} ${colors.text} border ${colors.border} text-[9px] px-1.5 py-0.5 rounded-sm`}
          >
            {formatNum(score)}
          </span>
        )
      },
    },
    {
      accessorFn: (row) => row.statistical_significance.is_significant,
      id: "significant",
      header: "Sig?",
      cell: ({ getValue }) => {
        const sig = getValue<boolean>()
        return sig ? (
          <Check className="w-3.5 h-3.5 text-[#00FF9C]" />
        ) : (
          <X className="w-3.5 h-3.5 text-[#FF4D4D]" />
        )
      },
      enableSorting: false,
    },
  ]
}

// ── Main Component ───────────────────────────────────────────────────────────

export function ExperimentBrowser({
  experiments,
  onRefresh,
  loading = false,
}: ExperimentBrowserProps) {
  const [sorting, setSorting] = useState<SortingState>([
    { id: "date", desc: true },
  ])

  const columns = useMemo(() => buildColumns(), [])

  const table = useReactTable({
    data: experiments,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-[#1E2A38] bg-[#0B0F14] shrink-0">
        <div className="flex items-center gap-1.5">
          <FlaskConical className="w-3.5 h-3.5 text-[#4CC9F0]" />
          <span className="text-[10px] font-bold uppercase tracking-widest text-[#4CC9F0]">
            Experiments
          </span>
          <span className="text-[9px] text-[#8B949E] ml-1">
            ({experiments.length})
          </span>
        </div>
        <button
          onClick={onRefresh}
          disabled={loading}
          className="flex items-center gap-1 px-2 py-1 text-[9px] uppercase tracking-wider border border-[#1E2A38] rounded-sm text-[#8B949E] hover:border-[#4CC9F0] hover:text-[#4CC9F0] transition-colors disabled:opacity-40"
        >
          <RefreshCw
            className={`w-3 h-3 ${loading ? "animate-spin" : ""}`}
          />
          Refresh
        </button>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        {experiments.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-[#8B949E]">
            <FlaskConical className="w-6 h-6 opacity-40" />
            <span className="text-[11px]">
              No experiments found. Run the pipeline to generate results.
            </span>
          </div>
        ) : (
          <table className="w-full text-left">
            <thead className="sticky top-0 bg-[#0B0F14] z-10">
              {table.getHeaderGroups().map((headerGroup) => (
                <tr key={headerGroup.id} className="border-b border-[#1E2A38]">
                  {headerGroup.headers.map((header) => (
                    <th
                      key={header.id}
                      className={`text-[9px] uppercase tracking-wider text-[#8B949E] font-medium py-2 px-3 ${
                        header.column.getCanSort()
                          ? "cursor-pointer hover:text-[#E6EDF3] select-none"
                          : ""
                      }`}
                      onClick={header.column.getToggleSortingHandler()}
                    >
                      {header.isPlaceholder
                        ? null
                        : flexRender(
                            header.column.columnDef.header,
                            header.getContext()
                          )}
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody>
              {table.getRowModel().rows.map((row) => (
                <tr
                  key={row.id}
                  className="border-b border-[#1E2A38] last:border-0 hover:bg-[#1A2130] transition-colors"
                >
                  {row.getVisibleCells().map((cell) => (
                    <td
                      key={cell.id}
                      className="text-[11px] font-mono text-[#E6EDF3] py-2 px-3"
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
