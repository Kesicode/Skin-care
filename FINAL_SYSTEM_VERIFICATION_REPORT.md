# DermaAI -- Final System Verification Report

**Generated:** 2026-09-27T06:14:56Z  
**Overall:** `FAIL`  
**Checks:** 120 PASS / 1 FAIL / 0 SKIP / 121 Total

---

## Results

| # | Title | Status | Detail |
|---|-------|--------|--------|
| 01 | Working tree clean | FAIL | Modified: ?? FINAL_SYSTEM_VERIFICATION.json ?? FINAL_SYSTEM_VERIFICATION_REPORT.md ?? final_system_verification.py |
| 01 | Latest commit | PASS | 0e29530 feat(dermaai): complete phase 20 - final hackathon submission package and presentation materials |
| 01 | Remote -> GitHub | PASS | origin	https://github.com/Kesicode/Skin-care.git (fetch) |
| 02 | .gitignore exists | PASS |  |
| 02 | *.pt or models/ ignored | PASS |  |
| 02 | node_modules ignored | PASS |  |
| 02 | .next build ignored | PASS |  |
| 02 | __pycache__ ignored | PASS |  |
| 02 | phase17 results allowed | PASS |  |
| 02 | phase20 results allowed | PASS |  |
| 03 | Exp8 checkpoint exists | PASS | 17,512,600 bytes |
| 03 | Exp13 checkpoint exists | PASS | 17,512,921 bytes |
| 04 | Exp8 SHA-256 matches spec | PASS | ccc579cb2087545fa1118b75... |
| 04 | Exp13 SHA-256 matches spec | PASS | 9fab1b1218fddc86e060f322... |
| 05 | Exp8 size matches spec | PASS | 17,512,600 bytes |
| 05 | Exp13 size matches spec | PASS | 17,512,921 bytes |
| 06 | DermaAI_MobileNetV3 importable from ai.src.model | PASS |  |
| 07 | Exp8 param count = 4,326,297 | PASS |  |
| 08 | Exp13 param count = 4,326,297 | PASS |  |
| 09 | Exp8 loads without error | PASS |  |
| 10 | Exp13 loads without error | PASS |  |
| 11 | Exp8 in eval() mode | PASS |  |
| 11 | Exp8 all params frozen | PASS |  |
| 12 | Exp13 in eval() mode | PASS |  |
| 12 | Exp13 all params frozen | PASS |  |
| 13 | Exp8 has model.attention (CBAM) | PASS |  |
| 14 | Exp13 has model.attention (CBAM) | PASS |  |
| 15 | Exp8 output shape=(1,7) | PASS |  |
| 16 | Exp13 output shape=(1,7) | PASS |  |
| 17 | shape=[1,3,224,224] | PASS |  |
| 18 | Normalized range [2.249,2.640] (expected normalized values) | PASS |  |
| 19 | Exp8 softmax sum=1.00000000 | PASS |  |
| 20 | Exp13 softmax sum=1.00000000 | PASS |  |
| 21 | Ensemble formula correctly applied | PASS |  |
| 21 | Weights sum=1.0000000000 (=1.0 exactly) | PASS |  |
| 22 | Ensemble sum=1.00000012 | PASS |  |
| 23 | Argmax->class 5 (nv), conf=0.2402 | PASS |  |
| 24 | EXP8_WEIGHT = 0.85 in config.py | PASS |  |
| 24 | EXP13_WEIGHT = 0.15 in config.py | PASS |  |
| 24 | inference.py uses weight constants from config | PASS |  |
| 25 | Exp8 hooks fire, heatmap=(7,7) | PASS |  |
| 26 | Exp13 hooks fire, heatmap=(7,7) | PASS |  |
| 27 | model.attention shape=(1, 960, 7, 7) | PASS |  |
| 28 | Range=[0.0000,1.0000] in [0,1] | PASS |  |
| 29 | Valid base64 PNG (200x200) | PASS |  |
| 30 | Two identical runs produce identical output | PASS |  |
| 31 | CLASSES has 7 entries | PASS |  |
| 31 | All indices 0-6 in order | PASS |  |
| 31 | CLASS_CODES=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'] | PASS |  |
| 32 | akiec: OK | PASS |  |
| 32 | bcc: OK | PASS |  |
| 32 | bkl: OK | PASS |  |
| 32 | df: OK | PASS |  |
| 32 | mel: OK | PASS |  |
| 32 | nv: OK | PASS |  |
| 32 | vasc: OK | PASS |  |
| 33 | EXP8_WEIGHT=0.85 | PASS |  |
| 33 | EXP13_WEIGHT=0.15 | PASS |  |
| 34 | Exp8 SHA-256 constant correct | PASS |  |
| 34 | Exp13 SHA-256 constant correct | PASS |  |
| 35 | EXPECTED_PARAMS_COUNT=4,326,297 | PASS |  |
| 36 | Notice contains 'RESEARCH USE ONLY' | PASS |  |
| 36 | Notice contains 'academic' | PASS |  |
| 36 | Notice contains 'NOT a certified medical diagnostic' | PASS |  |
| 36 | Notice contains 'substitute' | PASS |  |
| 37 | Exp15 note in inference.py | PASS |  |
| 37 | Exp15 note in schemas.py | PASS |  |
| 38 | config imports OK | PASS |  |
| 38 | schemas imports OK | PASS |  |
| 38 | utils imports OK | PASS |  |
| 38 | preprocessing imports OK | PASS |  |
| 39 | app is FastAPI instance | PASS |  |
| 39 | Routes: ['/openapi.json', '/docs', '/docs/oauth2-redirect', '/redoc', '/api/health', '/api/models/info', '/api/model-info', '/api/predict'] | PASS |  |
| 40 | status=200, models_loaded=True | PASS |  |
| 40 | status="ok" | PASS |  |
| 40 | ensemble.experiment8_weight=0.85 | PASS |  |
| 41 | status=200 | PASS |  |
| 41 |   architecture=DermaAI_MobileNetV3 | PASS |  |
| 41 |   backbone=MobileNetV3-Large | PASS |  |
| 41 |   attention=CBAM | PASS |  |
| 41 |   num_classes=7 | PASS |  |
| 41 |   parameters_per_model=4326297 | PASS |  |
| 41 |   decision_rule=argmax | PASS |  |
| 41 | classes=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'] | PASS |  |
| 42 | Alias returns 200 | PASS |  |
| 43 | Returns 200 for valid JPEG | PASS |  |
| 44 | Rejects text/plain with 400 | PASS |  |
| 45 | 'success' present | PASS |  |
| 45 | 'prediction' present | PASS |  |
| 45 | 'confidence' present | PASS |  |
| 45 | 'probabilities' present | PASS |  |
| 45 | 'top3' present | PASS |  |
| 45 | 'ensemble' present | PASS |  |
| 45 | 'gradcam' present | PASS |  |
| 45 | 'research_notice' present | PASS |  |
| 46 | probabilities dict has 7 entries | PASS |  |
| 46 | sum=1.00000005~1.0 | PASS |  |
| 46 | All 7 class codes present | PASS |  |
| 47 | Notice contains 'RESEARCH USE ONLY' | PASS |  |
| 47 | Notice contains 'academic' | PASS |  |
| 47 | Notice contains 'NOT a certified medical diagnostic' | PASS |  |
| 47 | Notice contains 'substitute' | PASS |  |
| 48 | gradcam.available=True | PASS |  |
| 48 | target_layer="model.attention" | PASS |  |
| 48 | gradcam.experiment8 is valid base64 PNG | PASS |  |
| 48 | gradcam.experiment13 is valid base64 PNG | PASS |  |
| 49 | No training-set references in inference/service/main | PASS |  |
| 50 | No test-split references in demo/inference code | PASS |  |
| 51 | .next/ exists | PASS |  |
| 51 | BUILD_ID=tazf8XwIcqcVxxUsgzHsc | PASS |  |
| 51 | .next/static/ exists | PASS |  |
| 52 | src\app\page.tsx (39,056 bytes) | PASS |  |
| 52 | src\app\api\predict\route.ts (1,006 bytes) | PASS |  |
| 52 | src\app\api\health\route.ts (693 bytes) | PASS |  |
| 52 | src\app\layout.tsx (1,057 bytes) | PASS |  |
| 53 | scripts.dev="next dev" | PASS |  |
| 53 | scripts.build="next build" | PASS |  |
| 53 | scripts.start="next start" | PASS |  |
| 53 | "next"="16.3.5" | PASS |  |
| 53 | "react"="19.2.8" | PASS |  |
| 53 | "react-dom"="19.2.8" | PASS |  |

---

## Frozen Spec

| Item | Value |
|------|-------|
| Exp8 SHA-256 | `ccc579cb2087545fa1118b7551015bec17c0682ec04066268d3a46aa5d23be2c` |
| Exp13 SHA-256 | `9fab1b1218fddc86e060f322506bafa691429ef2620d7ff1c26041dc7b52ef21` |
| Exp8 size | 17,512,600 bytes |
| Exp13 size | 17,512,921 bytes |
| Params/model | 4,326,297 |
| Ensemble | 0.85*Exp8 + 0.15*Exp13 |
| Input | 224x224 |
| Classes | akiec, bcc, bkl, df, mel, nv, vasc |
| Grad-CAM target | model.attention [1,960,7,7] |

> READ-ONLY AUDIT. No weights, checkpoints, or code were modified.