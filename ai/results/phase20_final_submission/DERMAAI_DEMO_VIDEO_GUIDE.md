# DermaAI: Demo Video Recording Guide & Storyboard (2–3 Minutes)

This document provides the exact storyboard, screen recording cues, and narration alignment for recording `DERMAAI_DEMO_VIDEO.mp4`.

---

## 1. Video Technical Specifications
- **Target Duration**: 2 minutes 45 seconds (±15 seconds)
- **Resolution**: 1080p (1920x1080) at 30 or 60 fps
- **Audio**: Clear 48 kHz stereo voiceover matching `FINAL_PITCH_SCRIPT.md`
- **Output Filename**: `DERMAAI_DEMO_VIDEO.mp4`

---

## 2. Storyboard & Scene Progression

| Scene # | Time | Visual Feed | Audio Narration / Dialogue |
|---|---|---|---|
| **Scene 1: Title & Hook** | 0:00 – 0:20 | `HACKATHON_FINAL_PPT.pptx` (Slide 1–2) | Introduce DermaAI, explain the clinical challenge of dermoscopy and the class imbalance dilemma. |
| **Scene 2: Dataset & Imbalance** | 0:20 – 0:45 | `HACKATHON_FINAL_PPT.pptx` (Slide 3–4) | Explain HAM10000 7-class distribution, lesion-level split (seed 42), and population separation ($N=988$ test vs $N=1,010$ val). |
| **Scene 3: Methodology & Architecture** | 0:45 – 1:10 | `HACKATHON_FINAL_PPT.pptx` (Slide 5–6) | Show MobileNetV3-Large + CBAM attention ($4,326,297$ parameters) and 15 controlled experiments. |
| **Scene 4: The 85/15 Ensemble** | 1:10 – 1:30 | `HACKATHON_FINAL_PPT.pptx` (Slide 6 & 8) | Explain $P_{\text{final}} = 0.85 P_8 + 0.15 P_{13}$, argmax decision, and audited benchmark results ($80.57\%$ Acc, $75.69\%$ Bal Acc, $52.43\%$ Mel F1). |
| **Scene 5: Live Application Screen** | 1:30 – 1:55 | Screen recording of `http://localhost:3000` | Open DermaAI, point out research banner, drag and drop non-test lesion image, click "Analyze Lesion Image". |
| **Scene 6: Real-Time Results** | 1:55 – 2:15 | Screen recording of DermaAI Results | Show predicted class, confidence score, top-3 distribution, and 7-class probability bars summing to 1.00. |
| **Scene 7: Dual Grad-CAM** | 2:15 – 2:35 | Screen recording of Grad-CAM panel | Explain side-by-side heatmaps: Exp 8 global border context vs Exp 13 focal atypical pigment network. |
| **Scene 8: Error Analysis & Rigor** | 2:35 – 2:50 | `HACKATHON_FINAL_PPT.pptx` (Slide 9) | Transparent review of melanoma misclassifications (26 nevi, 19 benign keratoses). |
| **Scene 9: Limitations & Closing** | 2:50 – 3:00 | `HACKATHON_FINAL_PPT.pptx` (Slide 13–14) | Emphasize **RESEARCH USE ONLY** notice, lack of clinical trials, and final project conclusion. |

---

## 3. Recording Instructions (OBS Studio / Screen Recorder)
1. Set display scaling to 100% and resolution to 1920x1080.
2. Open Chrome to `http://localhost:3000` and PowerPoint in full-screen slide show mode.
3. Start recording while reading aloud from `FINAL_PITCH_SCRIPT.md`.
4. Use smooth, deliberate mouse movements when dragging the image into the dropzone.
5. Export MP4 with H.264 video codec and AAC audio codec.
