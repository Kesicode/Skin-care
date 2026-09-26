"use client"

import { useState, useRef, useEffect } from "react"
import Image from "next/image"
import {
  UploadCloud,
  FileImage,
  Layers,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  Info,
  CheckCircle2,
  XCircle,
  BarChart3,
  Cpu,
  RefreshCw,
  ExternalLink
} from "lucide-react"

// Types matching backend schemas
interface PredictionDetail {
  class_index: number
  class_code: string
  class_name: string
}

interface TopPrediction {
  class_code: string
  class_name: string
  probability: number
}

interface GradCAMData {
  available: boolean
  target_class: string
  target_layer: string
  experiment8: string | null
  experiment13: string | null
  notice: string
}

interface PredictionResponse {
  success: boolean
  prediction: PredictionDetail
  confidence: number
  probabilities: Record<string, number>
  top3: TopPrediction[]
  ensemble: {
    experiment8_weight: number
    experiment13_weight: number
  }
  gradcam: GradCAMData
  research_notice: string
  experiment15_note: string
}

const CLASS_DESCRIPTIONS: Record<string, { label: string; full: string }> = {
  akiec: { label: "AKIEC", full: "Actinic Keratosis / Intraepithelial Carcinoma" },
  bcc: { label: "BCC", full: "Basal Cell Carcinoma" },
  bkl: { label: "BKL", full: "Benign Keratosis (Solar Lentigo / Seborrheic Keratosis)" },
  df: { label: "DF", full: "Dermatofibroma" },
  mel: { label: "MEL", full: "Melanoma" },
  nv: { label: "NV", full: "Melanocytic Nevus" },
  vasc: { label: "VASC", full: "Vascular Lesion" },
}

export default function DermaAIDemoPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [loadingStage, setLoadingStage] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [isModelInfoOpen, setIsModelInfoOpen] = useState(false)
  const [isEvaluationContextOpen, setIsEvaluationContextOpen] = useState(false)
  const [backendConnected, setBackendConnected] = useState<boolean | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)

  // Check health on mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
        const res = await fetch(`${apiUrl}/api/health`)
        if (res.ok) {
          const data = await res.json()
          setBackendConnected(data.models_loaded)
          return
        }
      } catch {
        // Fallback to proxy
        try {
          const res2 = await fetch("/api/health")
          if (res2.ok) {
            const data2 = await res2.json()
            setBackendConnected(data2.models_loaded)
            return
          }
        } catch {
          // Ignore
        }
      }
      setBackendConnected(false)
    }

    checkBackend()
  }, [])

  const handleFile = (file: File) => {
    setErrorMessage(null)
    setResult(null)

    if (!file.type.startsWith("image/")) {
      setErrorMessage("Please select a valid image file (JPG, JPEG, PNG).")
      return
    }

    if (file.size > 10 * 1024 * 1024) {
      setErrorMessage("Uploaded file exceeds the maximum allowed size of 10 MB.")
      return
    }

    setSelectedFile(file)
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragOver(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleRemoveImage = () => {
    setSelectedFile(null)
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }
    setPreviewUrl(null)
    setResult(null)
    setErrorMessage(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const runPrediction = async () => {
    if (!selectedFile) return

    setErrorMessage(null)
    setResult(null)

    // Stage 1: Loading image
    setLoadingStage("Loading image...")

    const formData = new FormData()
    formData.append("image", selectedFile)

    try {
      // Stage 2: Running ensemble
      setTimeout(() => setLoadingStage("Running ensemble..."), 300)
      setTimeout(() => setLoadingStage("Computing class probabilities..."), 700)
      setTimeout(() => setLoadingStage("Generating Grad-CAM..."), 1100)

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
      let response: Response

      try {
        response = await fetch(`${apiUrl}/api/predict`, {
          method: "POST",
          body: formData,
        })
      } catch (err) {
        // Fallback to Next.js API proxy
        response = await fetch("/api/predict", {
          method: "POST",
          body: formData,
        })
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.error || "Unable to analyze this image. Please try another valid dermoscopic image.")
      }

      const data: PredictionResponse = await response.json()
      setResult(data)
    } catch (err: any) {
      setErrorMessage(err.message || "Unable to analyze this image. Please try another valid dermoscopic image.")
    } finally {
      setLoadingStage(null)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* ── Top Research Alert Banner ── */}
      <div className="bg-amber-950/80 border-b border-amber-500/30 px-4 py-2 text-xs md:text-sm text-amber-200 text-center flex items-center justify-center gap-2">
        <Info className="h-4 w-4 shrink-0 text-amber-400" />
        <span>
          <strong>Academic Research Demonstrator:</strong> Strictly for image-level classification on retrospective HAM10000 data. Not for clinical diagnosis.
        </span>
      </div>

      {/* ── Hero Header ── */}
      <header className="relative pt-12 pb-10 px-4 border-b border-white/5 bg-gradient-to-b from-slate-900 to-slate-950">
        <div className="container mx-auto max-w-5xl text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/60 border border-cyan-500/30 text-cyan-300 text-xs font-medium mb-4">
            <Cpu className="h-3.5 w-3.5 text-cyan-400" />
            <span>Frozen Ensemble (85% Exp8 + 15% Exp13)</span>
          </div>

          <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight text-white mb-4">
            Derma<span className="text-cyan-400">AI</span>
          </h1>

          <p className="text-lg md:text-xl text-slate-300 font-light max-w-3xl mx-auto mb-2">
            AI-Assisted Dermoscopic Image Classification Research Demonstration
          </p>

          <p className="text-sm text-slate-400 max-w-2xl mx-auto">
            Interactive software demonstration evaluating the frozen Experiment 14 dual-model probability ensemble
            with verified Grad-CAM attention visualizations on the HAM10000 7-class lesion dataset.
          </p>
        </div>
      </header>

      {/* ── Main Container ── */}
      <main className="container mx-auto max-w-5xl px-4 py-8 flex-1 flex flex-col gap-10">

        {/* ── 1. Upload & Analysis Section ── */}
        <section id="analyze" className="scroll-mt-24">
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 md:p-8 shadow-xl shadow-black/40">
            <h2 className="text-xl font-bold text-white mb-2 flex items-center gap-2">
              <UploadCloud className="h-5 w-5 text-cyan-400" />
              Upload Dermoscopic Image
            </h2>
            <p className="text-sm text-slate-400 mb-6">
              Select or drop a dermoscopic lesion image to run the frozen ensemble pipeline. Note: this research model was trained specifically on dermoscopic images and does not evaluate standard consumer photos.
            </p>

            {/* Drag & Drop Zone */}
            {!previewUrl ? (
              <div
                onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
                onDragLeave={() => setIsDragOver(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-10 md:p-14 text-center cursor-pointer transition-all duration-200 flex flex-col items-center justify-center gap-3 ${
                  isDragOver
                    ? "border-cyan-400 bg-cyan-950/20"
                    : "border-slate-700 bg-slate-950/50 hover:border-slate-500 hover:bg-slate-950"
                }`}
                tabIndex={0}
                role="button"
                aria-label="Upload a dermoscopic image"
                onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click() }}
              >
                <div className="h-16 w-16 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-cyan-400 mb-2">
                  <FileImage className="h-8 w-8" />
                </div>
                <div className="text-base font-semibold text-slate-200">
                  Upload a dermoscopic image
                </div>
                <div className="text-xs text-slate-400">
                  Drag and drop here, or click to browse (JPG, JPEG, PNG • max 10 MB)
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      handleFile(e.target.files[0])
                    }
                  }}
                />
              </div>
            ) : (
              <div className="flex flex-col md:flex-row items-center gap-6 bg-slate-950/60 border border-slate-800 rounded-xl p-4 md:p-6">
                <div className="relative w-48 h-48 rounded-lg overflow-hidden border border-slate-700 bg-black shrink-0">
                  <img
                    src={previewUrl}
                    alt="Uploaded dermoscopic preview"
                    className="w-full h-full object-cover"
                  />
                </div>

                <div className="flex-1 flex flex-col gap-3 text-center md:text-left">
                  <div>
                    <h3 className="font-semibold text-slate-200 text-sm">Selected File</h3>
                    <p className="text-xs text-slate-400 truncate max-w-sm">{selectedFile?.name}</p>
                    <p className="text-xs text-slate-500">
                      {selectedFile ? (selectedFile.size / 1024).toFixed(1) : 0} KB • Ready for evaluation preprocessing (224×224)
                    </p>
                  </div>

                  <div className="flex flex-wrap items-center justify-center md:justify-start gap-3 mt-2">
                    <button
                      onClick={runPrediction}
                      disabled={loadingStage !== null}
                      className="px-6 py-2.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-medium text-sm flex items-center gap-2 shadow-lg shadow-cyan-900/30 transition-all focus:outline-none focus:ring-2 focus:ring-cyan-400"
                    >
                      {loadingStage ? (
                        <>
                          <RefreshCw className="h-4 w-4 animate-spin" />
                          <span>{loadingStage}</span>
                        </>
                      ) : (
                        <>
                          <Layers className="h-4 w-4" />
                          <span>Analyze Image</span>
                        </>
                      )}
                    </button>

                    <button
                      onClick={handleRemoveImage}
                      disabled={loadingStage !== null}
                      className="px-4 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-300 font-medium text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-slate-500"
                    >
                      Remove Image
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Error Message Display */}
            {errorMessage && (
              <div className="mt-4 p-4 rounded-xl bg-rose-950/60 border border-rose-500/40 text-rose-200 text-sm flex items-start gap-3">
                <XCircle className="h-5 w-5 text-rose-400 shrink-0 mt-0.5" />
                <div>
                  <div className="font-semibold text-rose-300">Analysis Notice</div>
                  <div className="text-xs md:text-sm">{errorMessage}</div>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* ── 2. Prediction Results Dashboard ── */}
        {result && (
          <section className="flex flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Primary Result & Top-3 Cards */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Primary Prediction Card */}
              <div className="lg:col-span-7 bg-slate-900/90 border border-slate-800 rounded-2xl p-6 md:p-8 flex flex-col justify-between shadow-xl">
                <div>
                  <div className="text-xs font-bold tracking-wider text-cyan-400 uppercase mb-2">
                    MODEL PREDICTION
                  </div>
                  <h3 className="text-3xl md:text-4xl font-black text-white mb-1">
                    {result.prediction.class_name}
                  </h3>
                  <div className="text-sm text-slate-400 mb-6 flex items-center gap-2">
                    <span>Dataset class code:</span>
                    <span className="font-mono px-2 py-0.5 rounded bg-slate-800 text-cyan-300 text-xs font-semibold">
                      {result.prediction.class_code}
                    </span>
                    <span className="text-xs text-slate-500">• Index {result.prediction.class_index}</span>
                  </div>

                  {/* Confidence Display */}
                  <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/80 mb-6">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-slate-400 font-medium">Model confidence</span>
                      <span className="text-xl font-bold font-mono text-cyan-400">
                        {(result.confidence * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                      <div
                        className="bg-cyan-500 h-full rounded-full transition-all duration-700"
                        style={{ width: `${Math.min(100, result.confidence * 100)}%` }}
                      ></div>
                    </div>
                    <div className="text-[11px] text-slate-500 mt-2">
                      Softmax score from the 85/15 ensemble. Not a clinically calibrated disease probability.
                    </div>
                  </div>

                  {/* Class-Specific Mandatory Framing */}
                  {result.prediction.class_code === "mel" ? (
                    <div className="p-3.5 rounded-lg bg-amber-950/40 border border-amber-500/30 text-amber-200 text-xs leading-relaxed flex items-start gap-2.5">
                      <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
                      <div>
                        <strong>Predicted class: Melanoma (mel).</strong> This is an image-level research prediction and is not a medical diagnosis. Biopsy and clinical examination are required for medical confirmation.
                      </div>
                    </div>
                  ) : (
                    <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-slate-300 text-xs leading-relaxed flex items-start gap-2.5">
                      <Info className="h-4 w-4 text-slate-400 shrink-0 mt-0.5" />
                      <div>
                        <strong>Predicted class: {result.prediction.class_name} ({result.prediction.class_code}).</strong> The system reports the retrospective dataset class prediction only.
                      </div>
                    </div>
                  )}

                  {/* Low Confidence Warning */}
                  {result.confidence < 0.60 && (
                    <div className="mt-3 p-3 rounded-lg bg-amber-950/30 border border-amber-600/30 text-amber-300 text-xs flex items-start gap-2">
                      <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
                      <span>
                        The model has relatively low confidence in this prediction. Consider the result uncertain and avoid interpreting it as a diagnosis.
                      </span>
                    </div>
                  )}
                </div>

                <div className="mt-6 pt-4 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-500">
                  <span>Ensemble rule: 0.85 P8 + 0.15 P13</span>
                  <span>Decision: Standard argmax</span>
                </div>
              </div>

              {/* Top-3 Predictions Distribution Card */}
              <div className="lg:col-span-5 bg-slate-900/90 border border-slate-800 rounded-2xl p-6 md:p-8 flex flex-col justify-between shadow-xl">
                <div>
                  <div className="text-xs font-bold tracking-wider text-slate-400 uppercase mb-2">
                    MODEL PROBABILITY DISTRIBUTION
                  </div>
                  <h3 className="text-xl font-bold text-white mb-1">Top-3 Predictions</h3>
                  <p className="text-xs text-slate-400 mb-6">
                    Highest ensemble output probabilities among the 7 candidate lesion classes.
                  </p>

                  <div className="space-y-4">
                    {result.top3.map((item, idx) => (
                      <div key={item.class_code} className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
                        <div className="flex items-center justify-between text-sm mb-1.5">
                          <span className="font-semibold text-slate-200 flex items-center gap-2">
                            <span className="text-xs font-mono text-cyan-400">#{idx + 1}</span>
                            <span>{item.class_name}</span>
                          </span>
                          <span className="font-mono font-bold text-cyan-300 text-sm">
                            {(item.probability * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-700 ${
                              idx === 0 ? "bg-cyan-400" : idx === 1 ? "bg-cyan-600" : "bg-slate-500"
                            }`}
                            style={{ width: `${Math.min(100, item.probability * 100)}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="mt-6 pt-4 border-t border-slate-800/80 text-[11px] text-slate-500 leading-normal">
                  Percentages reflect raw normalized model outputs and do not represent patient risk likelihood.
                </div>
              </div>
            </div>

            {/* ── 3. Full 7-Class Probability Horizontal Distribution ── */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 md:p-8 shadow-xl">
              <div className="flex flex-col md:flex-row md:items-center justify-between mb-6">
                <div>
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <BarChart3 className="h-5 w-5 text-cyan-400" />
                    Seven-Class Probability Distribution
                  </h3>
                  <p className="text-xs text-slate-400 mt-1">
                    Normalized probabilities for all 7 lesion categories generated by the frozen probability ensemble.
                  </p>
                </div>
                <div className="text-xs text-slate-500 font-mono mt-2 md:mt-0">
                  Total sum: {(Object.values(result.probabilities).reduce((a, b) => a + b, 0) * 100).toFixed(1)}%
                </div>
              </div>

              <div className="space-y-3.5">
                {Object.entries(result.probabilities).map(([code, prob]) => {
                  const meta = CLASS_DESCRIPTIONS[code] || { label: code.toUpperCase(), full: code }
                  const isTop = code === result.prediction.class_code
                  const pct = (prob * 100).toFixed(2)

                  return (
                    <div key={code} className="grid grid-cols-12 gap-3 items-center text-xs md:text-sm">
                      <div className="col-span-4 sm:col-span-3 font-mono font-medium text-slate-300 truncate">
                        <span className={isTop ? "text-cyan-400 font-bold" : "text-slate-400"}>
                          {meta.label}
                        </span>
                        <span className="hidden sm:inline text-xs text-slate-500 ml-1.5">
                          ({meta.full.split(" ")[0]})
                        </span>
                      </div>

                      <div className="col-span-6 sm:col-span-7">
                        <div className="w-full bg-slate-950 h-3 rounded-full overflow-hidden border border-slate-800 p-0.5">
                          <div
                            className={`h-full rounded-full transition-all duration-700 ${
                              isTop ? "bg-cyan-400" : "bg-slate-600"
                            }`}
                            style={{ width: `${Math.max(2, prob * 100)}%` }}
                          ></div>
                        </div>
                      </div>

                      <div className="col-span-2 text-right font-mono font-bold text-slate-300">
                        {pct}%
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* ── 4. Grad-CAM Model Attention Visualizations ── */}
            <div id="gradcam" className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 md:p-8 shadow-xl scroll-mt-24">
              <div className="mb-6">
                <div className="text-xs font-bold tracking-wider text-cyan-400 uppercase mb-1">
                  EXPLAINABILITY INSPECTION
                </div>
                <h3 className="text-2xl font-bold text-white">Model Attention Visualization (Grad-CAM)</h3>
                <p className="text-sm text-slate-400 mt-1 max-w-3xl leading-relaxed">
                  Grad-CAM highlights image regions that contributed strongly to the selected class score (
                  <span className="font-mono text-cyan-300">{result.prediction.class_name}</span>) in each component model.
                  Visualizations are computed on the verified CBAM attention layer (<code className="text-xs text-slate-300">model.attention</code>, feature map shape [1, 960, 7, 7]).
                </p>
              </div>

              {/* Side-by-side or stacked on mobile */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                {/* Original Image */}
                <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 flex flex-col items-center">
                  <div className="text-xs font-semibold text-slate-300 mb-3 text-center">
                    Original Dermoscopic Image
                  </div>
                  <div className="relative w-full aspect-square rounded-lg overflow-hidden border border-slate-800 bg-black">
                    {previewUrl && (
                      <img
                        src={previewUrl}
                        alt="Original dermoscopy input"
                        className="w-full h-full object-cover"
                      />
                    )}
                  </div>
                  <div className="text-[11px] text-slate-500 mt-3 text-center">
                    Preprocessed to 224×224 RGB
                  </div>
                </div>

                {/* Experiment 8 Grad-CAM */}
                <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 flex flex-col items-center">
                  <div className="text-xs font-semibold text-cyan-400 mb-3 text-center">
                    Grad-CAM — Experiment 8 component
                  </div>
                  <div className="relative w-full aspect-square rounded-lg overflow-hidden border border-slate-800 bg-black">
                    {result.gradcam.experiment8 ? (
                      <img
                        src={result.gradcam.experiment8}
                        alt="Grad-CAM visualization for Experiment 8"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="h-full flex items-center justify-center text-xs text-slate-500">
                        Overlay unavailable
                      </div>
                    )}
                  </div>
                  <div className="text-[11px] text-slate-400 mt-3 text-center">
                    Tempered Class Weighting (85% weight)
                  </div>
                </div>

                {/* Experiment 13 Grad-CAM */}
                <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 flex flex-col items-center">
                  <div className="text-xs font-semibold text-cyan-400 mb-3 text-center">
                    Grad-CAM — Experiment 13 component
                  </div>
                  <div className="relative w-full aspect-square rounded-lg overflow-hidden border border-slate-800 bg-black">
                    {result.gradcam.experiment13 ? (
                      <img
                        src={result.gradcam.experiment13}
                        alt="Grad-CAM visualization for Experiment 13"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="h-full flex items-center justify-center text-xs text-slate-500">
                        Overlay unavailable
                      </div>
                    )}
                  </div>
                  <div className="text-[11px] text-slate-400 mt-3 text-center">
                    Targeted Melanoma Weight (15% weight)
                  </div>
                </div>
              </div>

              {/* Interpretability Notice Box */}
              <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 text-xs text-slate-400 leading-relaxed flex items-start gap-3">
                <Info className="h-4 w-4 text-cyan-400 shrink-0 mt-0.5" />
                <div>
                  <strong>Interpretability Notice:</strong> The ensemble determines the displayed prediction. Grad-CAM maps show class-specific activation patterns from each frozen component model. Grad-CAM is an interpretability aid, not proof of clinical reasoning or biological causality.
                </div>
              </div>
            </div>
          </section>
        )}

        {/* ── 5. Research Transparency Panel (Collapsible) ── */}
        <section id="model-info" className="scroll-mt-24">
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
            <button
              onClick={() => setIsModelInfoOpen(!isModelInfoOpen)}
              className="w-full px-6 py-5 flex items-center justify-between text-left hover:bg-slate-850 transition-colors focus:outline-none"
              aria-expanded={isModelInfoOpen}
            >
              <div className="flex items-center gap-3">
                <Cpu className="h-5 w-5 text-cyan-400" />
                <div>
                  <h3 className="text-base font-bold text-white">About this model (Research Transparency)</h3>
                  <p className="text-xs text-slate-400">Technical specification of the frozen dual-model architecture</p>
                </div>
              </div>
              {isModelInfoOpen ? <ChevronUp className="h-5 w-5 text-slate-400" /> : <ChevronDown className="h-5 w-5 text-slate-400" />}
            </button>

            {isModelInfoOpen && (
              <div className="px-6 pb-6 pt-2 border-t border-slate-800 text-xs md:text-sm text-slate-300 flex flex-col gap-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                    <span className="text-slate-500 block text-xs">Dataset</span>
                    <span className="font-semibold text-slate-200">HAM10000 (Retrospective)</span>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                    <span className="text-slate-500 block text-xs">Classes</span>
                    <span className="font-semibold text-slate-200">7 Lesion Categories</span>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                    <span className="text-slate-500 block text-xs">Architecture</span>
                    <span className="font-semibold text-slate-200">MobileNetV3-Large + CBAM</span>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                    <span className="text-slate-500 block text-xs">Parameters per Model</span>
                    <span className="font-semibold text-slate-200">4,326,297 (Frozen)</span>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                    <span className="text-slate-500 block text-xs">Input Dimensions</span>
                    <span className="font-semibold text-slate-200">224 × 224 RGB</span>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                    <span className="text-slate-500 block text-xs">Ensemble Formula</span>
                    <span className="font-semibold text-slate-200">0.85 · P8 + 0.15 · P13</span>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs leading-relaxed text-slate-400">
                  <strong className="text-slate-200 block mb-1">Decision Rule & Threshold Verification:</strong>
                  The system uses standard argmax decision (<code className="text-cyan-300">argmax(P_ens)</code>). Experiment 15 evaluated post-hoc melanoma decision threshold tuning on the ensemble probabilities and confirmed that the optimal threshold (<code className="text-cyan-300">theta* = 0.50</code>) produced exactly zero changed predictions relative to the natural argmax rule.
                </div>
              </div>
            )}
          </div>
        </section>

        {/* ── 6. Project Research Benchmark Panel ── */}
        <section id="benchmarks" className="scroll-mt-24">
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 md:p-8 shadow-xl">
            <div className="flex flex-col md:flex-row md:items-center justify-between mb-4">
              <div>
                <div className="text-xs font-bold tracking-wider text-cyan-400 uppercase mb-1">
                  OFFICIAL METRICS
                </div>
                <h3 className="text-2xl font-bold text-white">Project Research Benchmark</h3>
                <p className="text-xs text-slate-400 mt-1">
                  Held-out benchmark metrics evaluated on the 988-image project test split (Experiment 14).
                </p>
              </div>
              <div className="mt-3 md:mt-0 inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-800 text-xs text-slate-300 font-mono">
                <span>N = 988 Test Images</span>
              </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
              {[
                { label: "Accuracy", value: "80.57%", sub: "796 / 988 correct" },
                { label: "Balanced Accuracy", value: "75.69%", sub: "Unweighted class mean" },
                { label: "Macro F1", value: "70.51%", sub: "Unweighted F1 average" },
                { label: "Weighted F1", value: "81.06%", sub: "Prevalence-weighted" },
                { label: "Melanoma Recall", value: "50.94%", sub: "54 / 106 detected" },
                { label: "Melanoma F1", value: "52.43%", sub: "Precision: 54.00%" },
              ].map((m, i) => (
                <div key={i} className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 text-center">
                  <div className="text-xs text-slate-400 mb-1">{m.label}</div>
                  <div className="text-xl md:text-2xl font-black font-mono text-cyan-400">{m.value}</div>
                  <div className="text-[10px] text-slate-500 mt-1">{m.sub}</div>
                </div>
              ))}
            </div>

            {/* Mandatory Disclaimers on Benchmark */}
            <div className="space-y-2.5 text-xs text-slate-400 leading-relaxed border-t border-slate-800 pt-4">
              <p>
                <strong>Evaluation Context:</strong> These are image-level benchmark results on the project&apos;s HAM10000 test split. They are <strong>NOT</strong> clinical diagnostic accuracy and should not be interpreted as clinical performance.
              </p>
              <p>
                <strong>Repeated Evaluation Disclosure:</strong> The project test split was evaluated repeatedly during the sequential research program; therefore this benchmark is not an independent external validation result.
              </p>
              <p className="text-slate-500 text-[11px]">
                Population Separation: The final benchmark metrics above reflect the 988-image project test split. The Phase 16 Grad-CAM explainability and error decomposition were conducted on the separate 1,010-image validation split.
              </p>
            </div>
          </div>
        </section>

        {/* ── 7. Narrative Research Timeline ── */}
        <section className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-6 md:p-8">
          <h3 className="text-lg font-bold text-white mb-4">Research Development Timeline</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 text-xs">
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/60">
              <span className="font-semibold text-cyan-400 block mb-1">Experiments 1–6</span>
              <span className="text-slate-400">Baseline architectures, loss strategies, augmentations, and CBAM integration.</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/60">
              <span className="font-semibold text-cyan-400 block mb-1">Experiments 7–8</span>
              <span className="text-slate-400">Gradual unfreezing fine-tuning and tempered class weighting (alpha=0.75).</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/60">
              <span className="font-semibold text-cyan-400 block mb-1">Experiments 9–13</span>
              <span className="text-slate-400">TTA, resolution trials (256x256), and melanoma-targeted weighting (1.20x).</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/60">
              <span className="font-semibold text-cyan-400 block mb-1">Experiments 14–18</span>
              <span className="text-slate-400">Probability ensemble (85/15), threshold audit, Grad-CAM analysis, and demo engineering.</span>
            </div>
          </div>
        </section>

        {/* ── 8. High-Importance Research Disclaimer Banner ── */}
        <section id="disclaimer" className="scroll-mt-24">
          <div className="rounded-2xl border-2 border-amber-500/40 bg-amber-950/20 p-6 md:p-8 text-amber-200">
            <div className="flex items-start gap-4">
              <AlertTriangle className="h-6 w-6 text-amber-400 shrink-0 mt-1" />
              <div className="space-y-2">
                <h3 className="text-lg font-bold text-amber-300 tracking-wide uppercase">
                  RESEARCH USE ONLY
                </h3>
                <p className="text-xs md:text-sm leading-relaxed text-amber-200/90">
                  This system is an academic research demonstrator for image-level classification of dermoscopic images. It is not a certified medical diagnostic device, has not undergone clinical trial evaluation, and must not be used as a substitute for professional clinical diagnosis, biopsy, dermoscopic examination, or physician decision-making in patient care.
                </p>
                <p className="text-xs md:text-sm text-amber-300 font-semibold">
                  Model predictions may be incorrect. Never disregard or delay seeking professional medical advice because of information presented by this demonstrator.
                </p>
              </div>
            </div>
          </div>
        </section>

      </main>

      {/* ── Footer ── */}
      <footer className="border-t border-white/10 bg-slate-950 py-8 px-4 text-center text-xs text-slate-500">
        <div className="container mx-auto max-w-5xl flex flex-col sm:flex-row items-center justify-between gap-4">
          <div>
            DermaAI Research Demonstrator • HAM10000 7-Class Image Classification
          </div>
          <div className="flex items-center gap-4 text-slate-400">
            <span>Frozen System</span>
            <span>•</span>
            <span>Zero Retraining</span>
            <span>•</span>
            <span>Academic Demonstration</span>
          </div>
        </div>
      </footer>
    </div>
  )
}
