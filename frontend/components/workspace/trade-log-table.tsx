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
import { ArrowUpDown, ArrowUp, ArrowDown, ScrollText } from "lucide-react"
import type { Trade } from "@/lib/types"

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatCurrency(v: number): string {
  return `$${v.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`
}

function formatNum(v: number, decimals = 2): string {
  return v.toFixed(decimals)
}

function actionBadge(action: string): { bg: string; text: string; border: string } {
  const lower = action.toLowerCase()
  if (lower === "buy")
    return {
      bg: "bg-[#00FF9C]/10",
      text: "text-[#00FF9C]",
      border: "border-[#00FF9C]/30",
    }
  if (lower === "sell")
    return {
      bg: "bg-[#FF4D4D]/10",
      text: "text-[#FF4D4D]",
      border: "border-[#FF4D4D]/30",
    }
  // liquidate
  return {
    bg: "bg-[#FFD60A]/10",
    text: "text-[#FFD60A]",
    border: "border-[#FFD60A]/30",
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

function buildColumns(): ColumnDef<Trade, unknown>[] {
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
      accessorKey: "action",
      header: ({ column }) => (
        <SortHeader label="Action" sorted={column.getIsSorted()} />
      ),
      cell: ({ getValue }) => {
        const action = getValue<string>()
        const colors = actionBadge(action)
        return (
          <span
            className={`${colors.bg} ${colors.text} border ${colors.border} text-[9px] px-1.5 py-0.5 rounded-sm uppercase font-bold tracking-wider`}
          >
            {action}
          </span>
        )
      },
    },
    {
      accessorKey: "price",
      header: ({ column }) => (
        <SortHeader label="Price" sorted={column.getIsSorted()} />
      ),
      cell: ({ getValue }) => (
        <span className="tabular-nums text-[#E6EDF3]">
          {formatCurrency(getValue<number>())}
        </span>
      ),
    },
    {
      accessorKey: "shares",
      header: "Shares",
      cell: ({ getValue }) => (
        <span className="tabular-nums text-[#E6EDF3]">
          {getValue<number>()}
        </span>
      ),
      enableSorting: false,
    },
    {
      accessorKey: "cost",
      header: "Cost",
      cell: ({ getValue }) => (
        <span className="tabular-nums text-[#E6EDF3]">
          {formatCurrency(getValue<number>())}
        </span>
      ),
      enableSorting: false,
    },
    {
      accessorKey: "slippage",
      header: "Slippage",
      cell: ({ getValue }) => (
        <span className="tabular-nums text-[#FFD60A]">
          {formatNum(getValue<number>(), 4)}
        </span>
      ),
      enableSorting: false,
    },
    {
      accessorKey: "portfolio_value",
      header: "Portfolio Value",
      cell: ({ getValue }) => (
        <span className="tabular-nums text-[#4CC9F0]">
          {formatCurrency(getValue<number>())}
        </span>
      ),
      enableSorting: false,
    },
  ]
}

// ── Types ────────────────────────────────────────────────────────────────────

interface TradeLogTableProps {
  trades: Trade[]
}

// ── Main Component ───────────────────────────────────────────────────────────

export function TradeLogTable({ trades }: TradeLogTableProps) {
  const [sorting, setSorting] = useState<SortingState>([
    { id: "date", desc: false },
  ])

  const columns = useMemo(() => buildColumns(), [])

  const table = useReactTable({
    data: trades,
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
          <ScrollText className="w-3.5 h-3.5 text-[#4CC9F0]" />
          <span className="text-[10px] font-bold uppercase tracking-widest text-[#4CC9F0]">
            Trade Log
          </span>
          <span className="text-[9px] text-[#8B949E] ml-1">
            ({trades.length} trades)
          </span>
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        {trades.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-[#8B949E]">
            <ScrollText className="w-6 h-6 opacity-40" />
            <span className="text-[11px]">
              No trades recorded. Run a backtest to generate trade logs.
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
