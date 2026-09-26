"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { Activity, Cpu, CheckCircle2, AlertCircle } from "lucide-react"

export function Navbar() {
  const [backendStatus, setBackendStatus] = useState<"checking" | "connected" | "unavailable">("checking")

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
        const res = await fetch(`${apiUrl}/api/health`, { cache: "no-store" })
        if (res.ok) {
          const data = await res.json()
          if (data.models_loaded) {
            setBackendStatus("connected")
            return
          }
        }
        setBackendStatus("unavailable")
      } catch {
        // Fallback: try relative proxy route /api/health
        try {
          const res2 = await fetch("/api/health", { cache: "no-store" })
          if (res2.ok) {
            setBackendStatus("connected")
            return
          }
        } catch {
          // Ignore
        }
        setBackendStatus("unavailable")
      }
    }

    checkHealth()
    const interval = setInterval(checkHealth, 15000)
    return () => clearInterval(interval)
  }, [])

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-white/10 bg-background/80 backdrop-blur-xl">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2 text-xl font-bold tracking-tight text-white hover:opacity-90 transition-opacity">
          <Activity className="h-6 w-6 text-cyan-400" />
          <span>DERMA<span className="text-cyan-400">AI</span></span>
          <span className="hidden sm:inline-block ml-2 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-2.5 py-0.5 text-xs font-medium text-cyan-300">
            Research Demonstrator
          </span>
        </Link>
        
        <div className="hidden lg:flex items-center gap-6 text-sm font-medium text-muted-foreground">
          <Link href="#analyze" className="hover:text-white transition-colors">Classifier Demo</Link>
          <Link href="#gradcam" className="hover:text-white transition-colors">Grad-CAM</Link>
          <Link href="#benchmarks" className="hover:text-white transition-colors">Research Benchmarks</Link>
          <Link href="#model-info" className="hover:text-white transition-colors">Architecture</Link>
          <Link href="#disclaimer" className="hover:text-white transition-colors">Research Disclaimer</Link>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Real Backend Status Indicator */}
          <div className="flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium border bg-black/40">
            {backendStatus === "connected" ? (
              <>
                <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span>
                <span className="text-emerald-400">Demo backend connected</span>
              </>
            ) : backendStatus === "checking" ? (
              <>
                <span className="h-2 w-2 rounded-full bg-amber-400 animate-pulse"></span>
                <span className="text-amber-400">Connecting backend...</span>
              </>
            ) : (
              <>
                <span className="h-2 w-2 rounded-full bg-rose-500"></span>
                <span className="text-rose-400">Backend unavailable</span>
              </>
            )}
          </div>

          <a
            href="#analyze"
            className="hidden sm:inline-flex items-center justify-center rounded-full bg-cyan-600 px-4 py-1.5 text-xs font-semibold text-white shadow-sm hover:bg-cyan-500 transition-colors"
          >
            Launch Demo
          </a>
        </div>
      </div>
    </nav>
  )
}
