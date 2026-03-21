import { Activity } from "lucide-react"
import Link from "next/link"

export function Footer() {
  return (
    <footer className="border-t border-[#1E2A38] bg-[#0B0F14]">
      <div className="max-w-7xl mx-auto px-6 py-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Brand */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-[#00FF9C]" />
              <span className="text-xs font-bold tracking-widest uppercase text-[#E6EDF3]">
                Finance<span className="text-[#00FF9C]">Bro</span>
              </span>
            </div>
            <p className="text-[11px] text-[#8B949E] leading-relaxed max-w-xs">
              Quantitative research and trading platform built with statistical rigor,
              walk-forward validation, and risk-aware portfolio construction.
            </p>
          </div>

          {/* Navigation */}
          <div className="flex flex-col gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-[#4CC9F0] mb-1">
              Navigation
            </span>
            {[
              { href: "/", label: "Home" },
              { href: "/pipeline", label: "Pipeline" },
              { href: "/terminal", label: "Terminal" },
              { href: "/about", label: "About" },
              { href: "/contact", label: "Contact" },
            ].map(({ href, label }) => (
              <Link
                key={href}
                href={href}
                className="text-xs text-[#8B949E] hover:text-[#E6EDF3] transition-colors w-fit"
              >
                {label}
              </Link>
            ))}
          </div>

          {/* Tech */}
          <div className="flex flex-col gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-[#4CC9F0] mb-1">
              Stack
            </span>
            <div className="flex flex-wrap gap-2">
              {["FastAPI", "React", "PostgreSQL", "Celery", "XGBoost", "PyTorch", "Alpaca"].map(
                (tech) => (
                  <span
                    key={tech}
                    className="px-2 py-0.5 text-[10px] font-medium text-[#8B949E] border border-[#1E2A38] rounded-sm"
                  >
                    {tech}
                  </span>
                )
              )}
            </div>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-[#1E2A38] flex items-center justify-between">
          <span className="text-[10px] text-[#8B949E]">
            &copy; {new Date().getFullYear()} FinanceBro. Built for quantitative research.
          </span>
          <span className="text-[10px] text-[#1E2A38] font-mono">v0.1.0</span>
        </div>
      </div>
    </footer>
  )
}
