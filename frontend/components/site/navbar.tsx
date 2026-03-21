"use client"

import { useState } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Activity, Terminal, Menu, X } from "lucide-react"

const NAV_LINKS = [
  { href: "/", label: "Home" },
  { href: "/pipeline", label: "Pipeline" },
  { href: "/terminal", label: "Terminal" },
  { href: "/about", label: "About" },
  { href: "/contact", label: "Contact" },
]

export function Navbar() {
  const pathname = usePathname()
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <nav className="sticky top-0 z-50 bg-[#0B0F14]/95 backdrop-blur-md border-b border-[#1E2A38]">
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
        {/* Brand */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <Activity className="w-5 h-5 text-[#00FF9C] group-hover:drop-shadow-[0_0_6px_#00FF9C] transition-all" />
          <span className="text-sm font-bold tracking-widest uppercase text-[#E6EDF3]">
            Finance<span className="text-[#00FF9C]">Bro</span>
          </span>
        </Link>

        {/* Desktop Links */}
        <div className="hidden md:flex items-center gap-1">
          {NAV_LINKS.map(({ href, label }) => {
            const isActive = pathname === href
            const isTerminal = href === "/terminal"

            if (isTerminal) {
              return (
                <Link
                  key={href}
                  href={href}
                  className="flex items-center gap-1.5 ml-2 px-4 py-1.5 text-xs font-semibold uppercase tracking-wider rounded-sm border border-[#00FF9C]/30 text-[#00FF9C] hover:bg-[#00FF9C]/10 hover:border-[#00FF9C]/60 transition-all"
                >
                  <Terminal className="w-3.5 h-3.5" />
                  {label}
                </Link>
              )
            }

            return (
              <Link
                key={href}
                href={href}
                className={`px-4 py-1.5 text-xs font-medium uppercase tracking-wider rounded-sm transition-colors ${
                  isActive
                    ? "text-[#4CC9F0] bg-[#4CC9F0]/10"
                    : "text-[#8B949E] hover:text-[#E6EDF3] hover:bg-[#1A2130]"
                }`}
              >
                {label}
              </Link>
            )
          })}
        </div>

        {/* Mobile toggle */}
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="md:hidden text-[#8B949E] hover:text-[#E6EDF3] transition-colors"
        >
          {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <div className="md:hidden border-t border-[#1E2A38] bg-[#0B0F14] px-6 py-3 flex flex-col gap-1">
          {NAV_LINKS.map(({ href, label }) => {
            const isActive = pathname === href
            return (
              <Link
                key={href}
                href={href}
                onClick={() => setMobileOpen(false)}
                className={`px-3 py-2 text-xs font-medium uppercase tracking-wider rounded-sm transition-colors ${
                  isActive
                    ? "text-[#4CC9F0] bg-[#4CC9F0]/10"
                    : "text-[#8B949E] hover:text-[#E6EDF3] hover:bg-[#1A2130]"
                }`}
              >
                {label}
              </Link>
            )
          })}
        </div>
      )}
    </nav>
  )
}
