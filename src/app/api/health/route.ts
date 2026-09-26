import { NextResponse } from "next/server"

export async function GET() {
  const backendUrl = process.env.DERMAAI_BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  try {
    const res = await fetch(`${backendUrl}/api/health`, {
      cache: "no-store",
    })
    if (!res.ok) {
      return NextResponse.json(
        { status: "error", message: `Backend returned HTTP ${res.status}` },
        { status: res.status }
      )
    }
    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json(
      { status: "unavailable", error: "Failed to connect to FastAPI backend" },
      { status: 503 }
    )
  }
}
