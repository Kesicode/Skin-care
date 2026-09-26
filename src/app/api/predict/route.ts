import { NextResponse } from "next/server"

export async function POST(request: Request) {
  const backendUrl = process.env.DERMAAI_BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  try {
    const formData = await request.formData()
    const file = formData.get("image")

    if (!file) {
      return NextResponse.json({ success: false, error: "No image file provided." }, { status: 400 })
    }

    const backendResponse = await fetch(`${backendUrl}/api/predict`, {
      method: "POST",
      body: formData,
    })

    const data = await backendResponse.json()
    return NextResponse.json(data, { status: backendResponse.status })
  } catch (error) {
    console.error("FastAPI backend connection error:", error)
    return NextResponse.json(
      {
        success: false,
        error: "Unable to connect to DermaAI inference backend. Please ensure the FastAPI server is running.",
        error_code: "BACKEND_UNAVAILABLE"
      },
      { status: 503 }
    )
  }
}
