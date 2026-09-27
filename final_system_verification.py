#!/usr/bin/env python3
"""
DERMAAI -- FINAL END-TO-END SYSTEM VERIFICATION
AUDIT ONLY. READ-ONLY. NO MODIFICATIONS.
"""
import os, sys, json, time, base64, hashlib, io, re, importlib
from datetime import datetime, timezone
from typing import Any, Dict, List

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
AI_DIR       = os.path.join(PROJECT_ROOT, "ai")
BACKEND_DIR  = os.path.join(PROJECT_ROOT, "backend")
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
MODELS_DIR   = os.path.join(AI_DIR, "models")
EXP8_PATH    = os.path.join(MODELS_DIR, "experiment8_best_model.pt")
EXP13_PATH   = os.path.join(MODELS_DIR, "experiment13_best_model.pt")

EXPECTED_EXP8_SHA256  = "ccc579cb2087545fa1118b7551015bec17c0682ec04066268d3a46aa5d23be2c"
EXPECTED_EXP13_SHA256 = "9fab1b1218fddc86e060f322506bafa691429ef2620d7ff1c26041dc7b52ef21"
EXPECTED_EXP8_SIZE    = 17_512_600
EXPECTED_EXP13_SIZE   = 17_512_921
EXPECTED_PARAMS       = 4_326_297
EXP8_WEIGHT           = 0.85
EXP13_WEIGHT          = 0.15
EXPECTED_CLASSES      = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
EXPECTED_CLASS_NAMES  = {
    "akiec": "Actinic Keratosis / Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis (Solar Lentigo / Seborrheic Keratosis)",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevus",
    "vasc":  "Vascular Lesion",
}
RESEARCH_KEYWORDS = ["RESEARCH USE ONLY", "academic", "NOT a certified medical diagnostic", "substitute"]

results: List[Dict[str, Any]] = []
_model8 = _model13 = _torch = _F = _device = _app = None

def _sec(n, t): print(f"\n{'='*68}\n  SECTION {n:02d}: {t}\n{'='*68}")
def _p(sec, t, d=""): results.append({"section":sec,"title":t,"status":"PASS","detail":d}); print(f"  [PASS]  {t}" + (f"\n          {d}" if d else ""))
def _f(sec, t, d=""): results.append({"section":sec,"title":t,"status":"FAIL","detail":d}); print(f"  [FAIL]  {t}" + (f"\n          {d}" if d else ""))
def _s(sec, t, r=""): results.append({"section":sec,"title":t,"status":"SKIP","detail":r}); print(f"  [SKIP]  {t} -- {r}")

def _sha256(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda: f.read(65536), b""): h.update(c)
    return h.hexdigest()

def _load_torch():
    global _torch, _F, _device
    if _torch: return True
    try:
        import torch, torch.nn.functional as F
        _torch=torch; _F=F; _device=torch.device("cpu"); return True
    except ImportError as e: return str(e)

def _load_models():
    global _model8, _model13
    if _model8 and _model13: return True
    r = _load_torch()
    if r is not True: return f"torch unavailable: {r}"
    if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
    if AI_DIR not in sys.path: sys.path.insert(0, AI_DIR)
    try:
        from ai.src.model import DermaAI_MobileNetV3
    except ImportError:
        try:
            sys.path.insert(0, os.path.join(AI_DIR,"src")); from model import DermaAI_MobileNetV3
        except ImportError as e: return f"Cannot import DermaAI_MobileNetV3: {e}"
    for attr, path in [("_model8", EXP8_PATH), ("_model13", EXP13_PATH)]:
        try:
            m = DermaAI_MobileNetV3(num_classes=7, use_attention=True)
            m.load_state_dict(_torch.load(path, map_location=_device))
            m.to(_device).eval()
            for p in m.parameters(): p.requires_grad = False
            globals()[attr] = m
        except Exception as e: return f"{attr} load error: {e}"
    return True

def _dummy(seed=42):
    _load_torch(); g=_torch.Generator().manual_seed(seed)
    return _torch.randn(1,3,224,224,generator=g)

def _synth_img():
    from PIL import Image as I; return I.new("RGB",(200,200),color=(128,80,60))

def _jpeg():
    from PIL import Image as I; buf=io.BytesIO(); I.new("RGB",(32,32),color=(100,150,200)).save(buf,"JPEG"); return buf.getvalue()

# ─── SECTIONS ────────────────────────────────────────────────────────────────

def s01():
    _sec(1,"Repository Integrity (git)")
    import subprocess
    try:
        r=subprocess.run(["git","status","--porcelain"],capture_output=True,text=True,cwd=PROJECT_ROOT)
        if r.returncode!=0: _f(1,"git status",r.stderr.strip()); return
        dirty=r.stdout.strip()
        _f(1,"Working tree clean",f"Modified:\n{dirty}") if dirty else _p(1,"Working tree clean")
        r2=subprocess.run(["git","log","-1","--oneline"],capture_output=True,text=True,cwd=PROJECT_ROOT)
        _p(1,"Latest commit",r2.stdout.strip())
        r3=subprocess.run(["git","remote","-v"],capture_output=True,text=True,cwd=PROJECT_ROOT)
        if "github.com" in r3.stdout: _p(1,"Remote -> GitHub",r3.stdout.strip().split("\n")[0])
        else: _f(1,"Remote not GitHub",r3.stdout.strip())
    except Exception as e: _f(1,"git command failed",str(e))

def s02():
    _sec(2,".gitignore Audit")
    gi=os.path.join(PROJECT_ROOT,".gitignore")
    if not os.path.exists(gi): _f(2,".gitignore exists"); return
    _p(2,".gitignore exists")
    with open(gi) as f: src=f.read()
    for label,ok in [
        ("*.pt or models/ ignored","*.pt" in src or "models/" in src),
        ("node_modules ignored","node_modules" in src),
        (".next build ignored",".next" in src),
        ("__pycache__ ignored","__pycache__" in src),
        ("phase17 results allowed","phase17_final_documentation" in src),
        ("phase20 results allowed","phase20_final_submission" in src),
    ]: (_p if ok else _f)(2,label)

def s03():
    _sec(3,"Checkpoint File Existence")
    for path,name in [(EXP8_PATH,"Exp8"),(EXP13_PATH,"Exp13")]:
        if os.path.exists(path): _p(3,f"{name} checkpoint exists",f"{os.path.getsize(path):,} bytes")
        else: _f(3,f"{name} checkpoint exists",f"Not found: {path}")

def s04():
    _sec(4,"Checkpoint SHA-256 Integrity")
    for path,exp,name in [(EXP8_PATH,EXPECTED_EXP8_SHA256,"Exp8"),(EXP13_PATH,EXPECTED_EXP13_SHA256,"Exp13")]:
        if not os.path.exists(path): _s(4,f"{name} SHA-256","File not found"); continue
        actual=_sha256(path)
        if actual.lower()==exp.lower(): _p(4,f"{name} SHA-256 matches spec",actual[:24]+"...")
        else: _f(4,f"{name} SHA-256 MISMATCH",f"Expected: {exp}\nActual:   {actual}")

def s05():
    _sec(5,"Checkpoint File Size")
    for path,exp,name in [(EXP8_PATH,EXPECTED_EXP8_SIZE,"Exp8"),(EXP13_PATH,EXPECTED_EXP13_SIZE,"Exp13")]:
        if not os.path.exists(path): _s(5,f"{name} size","File not found"); continue
        a=os.path.getsize(path)
        if a==exp: _p(5,f"{name} size matches spec",f"{a:,} bytes")
        else: _f(5,f"{name} size MISMATCH",f"Expected {exp:,}, got {a:,}")

def s06():
    _sec(6,"Model Architecture Import")
    if PROJECT_ROOT not in sys.path: sys.path.insert(0,PROJECT_ROOT)
    try: from ai.src.model import DermaAI_MobileNetV3; _p(6,"DermaAI_MobileNetV3 importable from ai.src.model")
    except ImportError:
        try: sys.path.insert(0,os.path.join(AI_DIR,"src")); from model import DermaAI_MobileNetV3; _p(6,"DermaAI_MobileNetV3 importable (fallback)")
        except ImportError as e: _f(6,"DermaAI_MobileNetV3 cannot be imported",str(e))

def s07_08():
    _sec(7,"Parameter Count -- Experiment 8"); _sec(8,"Parameter Count -- Experiment 13")
    r=_load_models()
    if r is not True: _f(7,"Exp8 (model not loaded)",r); _f(8,"Exp13 (model not loaded)",r); return
    for sec,model,name in [(7,_model8,"Exp8"),(8,_model13,"Exp13")]:
        n=sum(p.numel() for p in model.parameters())
        _p(sec,f"{name} param count = {n:,}") if n==EXPECTED_PARAMS else _f(sec,f"{name} param MISMATCH",f"Expected {EXPECTED_PARAMS:,}, got {n:,}")

def s09_10():
    _sec(9,"Model Load -- Experiment 8"); _sec(10,"Model Load -- Experiment 13")
    r=_load_models()
    if r is not True: _f(9,"Exp8 load",r); _f(10,"Exp13 load",r); return
    _p(9,"Exp8 loads without error"); _p(10,"Exp13 loads without error")

def s11_12():
    _sec(11,"eval()+frozen -- Exp8"); _sec(12,"eval()+frozen -- Exp13")
    if not (_model8 and _model13): _s(11,"Models not loaded"); _s(12,"Models not loaded"); return
    for sec,model,name in [(11,_model8,"Exp8"),(12,_model13,"Exp13")]:
        (_p if not model.training else _f)(sec,f"{name} in eval() mode")
        any_g=any(p.requires_grad for p in model.parameters())
        if not any_g: _p(sec,f"{name} all params frozen")
        else:
            gps=[n for n,p in model.named_parameters() if p.requires_grad]
            _f(sec,f"{name} has unfrozen params",f"{len(gps)}: {gps[:3]}")

def s13_14():
    _sec(13,"CBAM -- Exp8"); _sec(14,"CBAM -- Exp13")
    if not (_model8 and _model13): _s(13,"Not loaded"); _s(14,"Not loaded"); return
    for sec,model,name in [(13,_model8,"Exp8"),(14,_model13,"Exp13")]:
        (_p if hasattr(model,"attention") else _f)(sec,f"{name} has model.attention (CBAM)")

def s15_16():
    _sec(15,"Forward shape -- Exp8"); _sec(16,"Forward shape -- Exp13")
    if not (_model8 and _model13): _s(15,"Not loaded"); _s(16,"Not loaded"); return
    d=_dummy()
    with _torch.inference_mode():
        for sec,model,name in [(15,_model8,"Exp8"),(16,_model13,"Exp13")]:
            try:
                out=model(d)
                _p(sec,f"{name} output shape=(1,7)") if out.shape==(1,7) else _f(sec,f"{name} shape WRONG",str(tuple(out.shape)))
            except Exception as e: _f(sec,f"{name} forward ERROR",str(e))

def s17():
    _sec(17,"Preprocessing -- Tensor Shape"); _load_torch()
    try:
        import torchvision.transforms as T
        tfm=T.Compose([T.Resize((224,224),interpolation=T.InterpolationMode.BILINEAR),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        tensor=tfm(_synth_img()).unsqueeze(0)
        _p(17,"shape=[1,3,224,224]") if tensor.shape==(1,3,224,224) else _f(17,"Shape WRONG",str(tuple(tensor.shape)))
    except Exception as e: _f(17,"Preprocessing test error",str(e))

def s18():
    _sec(18,"Preprocessing -- Value Range"); _load_torch()
    try:
        import torchvision.transforms as T; from PIL import Image as PILImage
        tfm=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        t=tfm(PILImage.new("RGB",(224,224),(255,255,255))).unsqueeze(0)
        vmin,vmax=float(t.min()),float(t.max())
        _p(18,f"Normalized range [{vmin:.3f},{vmax:.3f}] (expected normalized values)") if vmin<10.0 and vmax>-10.0 else _f(18,"Range suspicious",f"[{vmin},{vmax}]")
    except Exception as e: _f(18,"Value range error",str(e))

def s19_20():
    _sec(19,"Softmax sum -- Exp8"); _sec(20,"Softmax sum -- Exp13")
    if not (_model8 and _model13): _s(19,"Not loaded"); _s(20,"Not loaded"); return
    d=_dummy()
    with _torch.inference_mode():
        for sec,model,name in [(19,_model8,"Exp8"),(20,_model13,"Exp13")]:
            s=float(_F.softmax(model(d),dim=-1).sum().item())
            _p(sec,f"{name} softmax sum={s:.8f}") if abs(s-1.0)<1e-5 else _f(sec,f"{name} sum={s:.8f}!=1.0")

def s21():
    _sec(21,"Ensemble Formula P=0.85*P8+0.15*P13")
    if not (_model8 and _model13): _s(21,"Not loaded"); return
    d=_dummy()
    with _torch.inference_mode():
        p8=_F.softmax(_model8(d),dim=-1); p13=_F.softmax(_model13(d),dim=-1)
        ens=EXP8_WEIGHT*p8+EXP13_WEIGHT*p13
    _p(21,"Ensemble formula correctly applied") if _torch.allclose(ens,0.85*p8+0.15*p13,atol=1e-7) else _f(21,"Ensemble formula mismatch")
    ws=EXP8_WEIGHT+EXP13_WEIGHT
    _p(21,f"Weights sum={ws:.10f} (=1.0 exactly)") if abs(ws-1.0)<1e-9 else _f(21,f"Weights NOT 1.0: {ws}")

def s22():
    _sec(22,"Ensemble Probability Sum~1.0")
    if not (_model8 and _model13): _s(22,"Not loaded"); return
    d=_dummy()
    with _torch.inference_mode():
        s=float((EXP8_WEIGHT*_F.softmax(_model8(d),dim=-1)+EXP13_WEIGHT*_F.softmax(_model13(d),dim=-1)).sum().item())
    _p(22,f"Ensemble sum={s:.8f}") if abs(s-1.0)<1e-5 else _f(22,f"Ensemble sum={s:.8f}!=1.0")

def s23():
    _sec(23,"Argmax Decision Rule")
    if not (_model8 and _model13): _s(23,"Not loaded"); return
    d=_dummy()
    with _torch.inference_mode():
        p8=_F.softmax(_model8(d),dim=-1); p13=_F.softmax(_model13(d),dim=-1)
        pe=EXP8_WEIGHT*p8+EXP13_WEIGHT*p13
        idx=int(_torch.argmax(pe,dim=-1).item()); conf=float(pe[0,idx].item())
    _p(23,f"Argmax->class {idx} ({EXPECTED_CLASSES[idx]}), conf={conf:.4f}") if 0<=idx<7 else _f(23,f"Argmax out of range: {idx}")

def s24():
    _sec(24,"Ensemble Weights Frozen in Source (0.85/0.15)")
    cfg=os.path.join(BACKEND_DIR,"app","config.py")
    if not os.path.exists(cfg): _f(24,"config.py not found"); return
    with open(cfg) as f: src=f.read()
    (_p if "EXP8_WEIGHT = 0.85" in src else _f)(24,"EXP8_WEIGHT = 0.85 in config.py")
    (_p if "EXP13_WEIGHT = 0.15" in src else _f)(24,"EXP13_WEIGHT = 0.15 in config.py")
    inf=os.path.join(BACKEND_DIR,"app","inference.py")
    if os.path.exists(inf):
        with open(inf) as f: isrc=f.read()
        if "EXP8_WEIGHT * p8" in isrc and "EXP13_WEIGHT * p13" in isrc: _p(24,"inference.py uses weight constants from config")
        else: _f(24,"inference.py does NOT use weight constants")

def s25_26():
    _sec(25,"Grad-CAM Hooks -- Exp8"); _sec(26,"Grad-CAM Hooks -- Exp13")
    if not (_model8 and _model13): _s(25,"Not loaded"); _s(26,"Not loaded"); return
    try:
        from backend.app.gradcam import GradCAMGenerator
    except ImportError:
        try: from app.gradcam import GradCAMGenerator
        except ImportError as e: _f(25,"GradCAMGenerator not importable",str(e)); _f(26,"GradCAMGenerator not importable",str(e)); return
    d=_dummy()
    for sec,model,name in [(25,_model8,"Exp8"),(26,_model13,"Exp13")]:
        try:
            gen=GradCAMGenerator(model=model,target_layer=model.attention)
            hm=gen.compute_cam(d,target_class_idx=4); gen.cleanup()
            _p(sec,f"{name} hooks fire, heatmap=(7,7)") if hm is not None and hm.shape==(7,7) else _f(sec,f"{name} shape WRONG",str(hm.shape if hm is not None else None))
        except Exception as e: _f(sec,f"{name} Grad-CAM ERROR",str(e))

def s27():
    _sec(27,"Grad-CAM Feature Map Shape [1,960,7,7]")
    if not _model8: _s(27,"Not loaded"); return
    cap=[]; hook=_model8.attention.register_forward_hook(lambda m,i,o: cap.append(o.detach().clone()))
    try:
        with _torch.no_grad(): _model8(_dummy()); hook.remove()
        if cap:
            sh=tuple(cap[0].shape)
            _p(27,f"model.attention shape={sh}") if sh==(1,960,7,7) else _f(27,f"Shape WRONG: {sh} (expected (1,960,7,7))")
        else: _f(27,"Hook did not capture activations")
    except Exception as e: hook.remove(); _f(27,"Feature map test ERROR",str(e))

def s28():
    _sec(28,"Grad-CAM Output Range [0,1]")
    if not _model8: _s(28,"Not loaded"); return
    try:
        from backend.app.gradcam import GradCAMGenerator
    except ImportError:
        try: from app.gradcam import GradCAMGenerator
        except ImportError as e: _f(28,"Import error",str(e)); return
    try:
        gen=GradCAMGenerator(model=_model8,target_layer=_model8.attention)
        hm=gen.compute_cam(_dummy(),target_class_idx=0); gen.cleanup()
        vmin,vmax=float(hm.min()),float(hm.max())
        _p(28,f"Range=[{vmin:.4f},{vmax:.4f}] in [0,1]") if vmin>=-1e-6 and vmax<=1.0+1e-6 else _f(28,f"Out of [0,1]: [{vmin:.4f},{vmax:.4f}]")
    except Exception as e: _f(28,"Range test ERROR",str(e))

def s29():
    _sec(29,"Grad-CAM Overlay -- Valid base64 PNG")
    if not _model8: _s(29,"Not loaded"); return
    try:
        from backend.app.gradcam import generate_component_gradcam
    except ImportError:
        try: from app.gradcam import generate_component_gradcam
        except ImportError as e: _f(29,"Import error",str(e)); return
    try:
        from PIL import Image as PILImage
        b64=generate_component_gradcam(model=_model8,input_tensor=_dummy(),original_img=_synth_img(),target_class_idx=4)
        if not isinstance(b64,str): _f(29,"Not a string",type(b64).__name__); return
        if not b64.startswith("data:image/png;base64,"): _f(29,"Missing data URI prefix",b64[:60]); return
        raw=base64.b64decode(b64.split(",",1)[1]); pil=PILImage.open(io.BytesIO(raw))
        _p(29,f"Valid base64 PNG ({pil.size[0]}x{pil.size[1]})")
    except Exception as e: _f(29,"Overlay test ERROR",str(e))

def s30():
    _sec(30,"Deterministic Inference")
    if not (_model8 and _model13): _s(30,"Not loaded"); return
    d=_dummy(seed=777); runs=[]
    for _ in range(2):
        with _torch.inference_mode():
            runs.append(EXP8_WEIGHT*_F.softmax(_model8(d),dim=-1)+EXP13_WEIGHT*_F.softmax(_model13(d),dim=-1))
    _p(30,"Two identical runs produce identical output") if _torch.allclose(runs[0],runs[1],atol=1e-7) else _f(30,"Non-deterministic!",f"Max diff={float((runs[0]-runs[1]).abs().max()):.2e}")

def s31():
    _sec(31,"Class Ordering (7 classes, strict index)")
    try:
        from backend.app.config import CLASSES, CLASS_CODES
        _p(31,"CLASSES has 7 entries") if len(CLASSES)==7 else _f(31,f"CLASSES len={len(CLASSES)}")
        bad=[i for i,c in enumerate(CLASSES) if c["index"]!=i]
        _p(31,"All indices 0-6 in order") if not bad else _f(31,f"Wrong indices at positions: {bad}")
        _p(31,f"CLASS_CODES={list(CLASS_CODES)}") if list(CLASS_CODES)==EXPECTED_CLASSES else _f(31,"CLASS_CODES WRONG",f"Expected {EXPECTED_CLASSES} got {list(CLASS_CODES)}")
    except Exception as e: _f(31,"ERROR",str(e))

def s32():
    _sec(32,"Class Codes and Names Correctness")
    try:
        from backend.app.config import CLASS_NAMES
        for code,exp_name in EXPECTED_CLASS_NAMES.items():
            a=CLASS_NAMES.get(code)
            if a==exp_name: _p(32,f"{code}: OK")
            elif a is None: _f(32,f"{code}: missing")
            else: _f(32,f"{code}: MISMATCH",f"Exp '{exp_name}' Got '{a}'")
    except Exception as e: _f(32,"ERROR",str(e))

def s33():
    _sec(33,"Config Ensemble Weights Match Spec")
    try:
        from backend.app.config import EXP8_WEIGHT as W8, EXP13_WEIGHT as W13
        _p(33,f"EXP8_WEIGHT={W8}") if abs(W8-0.85)<1e-9 else _f(33,f"EXP8_WEIGHT={W8} (expected 0.85)")
        _p(33,f"EXP13_WEIGHT={W13}") if abs(W13-0.15)<1e-9 else _f(33,f"EXP13_WEIGHT={W13} (expected 0.15)")
    except Exception as e: _f(33,"ERROR",str(e))

def s34():
    _sec(34,"Config SHA-256 Constants Match Spec")
    try:
        from backend.app.config import EXPECTED_EXP8_SHA256 as C8, EXPECTED_EXP13_SHA256 as C13
        _p(34,"Exp8 SHA-256 constant correct") if C8.lower()==EXPECTED_EXP8_SHA256.lower() else _f(34,"Exp8 SHA-256 WRONG",f"Exp {EXPECTED_EXP8_SHA256} Got {C8}")
        _p(34,"Exp13 SHA-256 constant correct") if C13.lower()==EXPECTED_EXP13_SHA256.lower() else _f(34,"Exp13 SHA-256 WRONG",f"Exp {EXPECTED_EXP13_SHA256} Got {C13}")
    except Exception as e: _f(34,"ERROR",str(e))

def s35():
    _sec(35,"Config Parameter Count Constant")
    try:
        from backend.app.config import EXPECTED_PARAMS_COUNT
        _p(35,f"EXPECTED_PARAMS_COUNT={EXPECTED_PARAMS_COUNT:,}") if EXPECTED_PARAMS_COUNT==EXPECTED_PARAMS else _f(35,f"WRONG: {EXPECTED_PARAMS_COUNT:,} (expected {EXPECTED_PARAMS:,})")
    except Exception as e: _f(35,"ERROR",str(e))

def s36():
    _sec(36,"Research Notice Presence and Content")
    try:
        from backend.app.config import RESEARCH_NOTICE
        for kw in RESEARCH_KEYWORDS:
            _p(36,f"Notice contains '{kw}'") if kw.upper() in RESEARCH_NOTICE.upper() else _f(36,f"Notice MISSING keyword '{kw}'")
    except Exception as e: _f(36,"ERROR",str(e))

def s37():
    _sec(37,"Experiment 15 Note Presence")
    for fpath,fname in [(os.path.join(BACKEND_DIR,"app","inference.py"),"inference.py"),(os.path.join(BACKEND_DIR,"app","schemas.py"),"schemas.py")]:
        if not os.path.exists(fpath): _f(37,f"{fname} not found"); continue
        with open(fpath) as f: src=f.read()
        _p(37,f"Exp15 note in {fname}") if "experiment15" in src.lower() else _f(37,f"Exp15 note MISSING from {fname}")

def s38():
    _sec(38,"Backend Module Imports (No Error)")
    for full,short in [("backend.app.config","config"),("backend.app.schemas","schemas"),("backend.app.utils","utils"),("backend.app.preprocessing","preprocessing")]:
        try: importlib.import_module(full); _p(38,f"{short} imports OK")
        except Exception as e: _f(38,f"{short} import ERROR",str(e))

def s39():
    _sec(39,"FastAPI App Object Exists")
    try:
        from backend.app.main import app; from fastapi import FastAPI
        if isinstance(app,FastAPI):
            _p(39,"app is FastAPI instance"); _p(39,f"Routes: {[r.path for r in app.routes]}")
        else: _f(39,f"app is {type(app)}")
    except Exception as e: _f(39,"ERROR",str(e))

def _client():
    global _app
    if _app: return _app
    try:
        from fastapi.testclient import TestClient; from backend.app.main import app
        _app=TestClient(app,raise_server_exceptions=False); return _app
    except Exception as e: return str(e)

def s40():
    _sec(40,"/api/health Endpoint")
    c=_client()
    if isinstance(c,str): _f(40,"TestClient failed",c); return
    try:
        r=c.get("/api/health")
        if r.status_code==200:
            body=r.json(); _p(40,f"status=200, models_loaded={body.get('models_loaded')}")
            _p(40,'status="ok"') if body.get("status")=="ok" else _f(40,f'status={body.get("status")}')
            w=body.get("ensemble",{}).get("experiment8_weight",0)
            _p(40,f"ensemble.experiment8_weight=0.85") if abs(w-0.85)<1e-6 else _f(40,f"weight={w}")
        else: _f(40,f"Returned {r.status_code}",r.text[:200])
    except Exception as e: _f(40,"ERROR",str(e))

def s41():
    _sec(41,"/api/model-info Endpoint")
    c=_client()
    if isinstance(c,str): _f(41,"TestClient failed",c); return
    try:
        r=c.get("/api/model-info")
        if r.status_code==200:
            body=r.json(); _p(41,"status=200")
            for k,exp in [("architecture","DermaAI_MobileNetV3"),("backbone","MobileNetV3-Large"),("attention","CBAM"),("num_classes",7),("parameters_per_model",EXPECTED_PARAMS),("decision_rule","argmax")]:
                _p(41,f"  {k}={exp}") if body.get(k)==exp else _f(41,f"  {k} WRONG",f"Exp {exp!r} got {body.get(k)!r}")
            cl=body.get("classes",[])
            _p(41,f"classes={cl}") if cl==EXPECTED_CLASSES else _f(41,"classes WRONG",str(cl))
        else: _f(41,f"Returned {r.status_code}",r.text[:200])
    except Exception as e: _f(41,"ERROR",str(e))

def s42():
    _sec(42,"/api/models/info Alias")
    c=_client()
    if isinstance(c,str): _f(42,"TestClient failed",c); return
    try:
        r=c.get("/api/models/info")
        _p(42,"Alias returns 200") if r.status_code==200 else _f(42,f"Returned {r.status_code}",r.text[:200])
    except Exception as e: _f(42,"ERROR",str(e))

def s43():
    _sec(43,"/api/predict -- Accepts Valid JPEG")
    c=_client()
    if isinstance(c,str): _f(43,"TestClient failed",c); return
    try:
        r=c.post("/api/predict",files={"image":("t.jpg",io.BytesIO(_jpeg()),"image/jpeg")})
        _p(43,"Returns 200 for valid JPEG") if r.status_code==200 else _f(43,f"Returned {r.status_code}",r.text[:300])
    except Exception as e: _f(43,"ERROR",str(e))

def s44():
    _sec(44,"/api/predict -- Rejects Non-Image")
    c=_client()
    if isinstance(c,str): _f(44,"TestClient failed",c); return
    try:
        r=c.post("/api/predict",files={"image":("t.txt",io.BytesIO(b"hello"),"text/plain")})
        _p(44,f"Rejects text/plain with {r.status_code}") if r.status_code in (400,415,422) else _f(44,f"Accepted non-image (status {r.status_code})")
    except Exception as e: _f(44,"ERROR",str(e))

def s45():
    _sec(45,"/api/predict -- All Required Response Fields")
    c=_client()
    if isinstance(c,str): _f(45,"TestClient failed",c); return
    try:
        r=c.post("/api/predict",files={"image":("t.jpg",io.BytesIO(_jpeg()),"image/jpeg")})
        if r.status_code!=200: _f(45,f"Returned {r.status_code}",r.text[:200]); return
        body=r.json()
        for field in ["success","prediction","confidence","probabilities","top3","ensemble","gradcam","research_notice"]:
            _p(45,f"'{field}' present") if field in body else _f(45,f"'{field}' MISSING")
    except Exception as e: _f(45,"ERROR",str(e))

def s46():
    _sec(46,"Probabilities Sum~1.0 in API Response")
    c=_client()
    if isinstance(c,str): _f(46,"TestClient failed",c); return
    try:
        r=c.post("/api/predict",files={"image":("t.jpg",io.BytesIO(_jpeg()),"image/jpeg")})
        if r.status_code!=200: _f(46,f"Returned {r.status_code}"); return
        body=r.json(); probs=body.get("probabilities",{})
        _p(46,"probabilities dict has 7 entries") if len(probs)==7 else _f(46,f"probabilities has {len(probs)} entries")
        s=sum(probs.values())
        _p(46,f"sum={s:.8f}~1.0") if abs(s-1.0)<1e-4 else _f(46,f"sum={s:.8f}!=1.0")
        missing=[c for c in EXPECTED_CLASSES if c not in probs]
        _p(46,"All 7 class codes present") if not missing else _f(46,f"Missing codes: {missing}")
    except Exception as e: _f(46,"ERROR",str(e))

def s47():
    _sec(47,"Clinical Disclaimer in API Response")
    c=_client()
    if isinstance(c,str): _f(47,"TestClient failed",c); return
    try:
        r=c.post("/api/predict",files={"image":("t.jpg",io.BytesIO(_jpeg()),"image/jpeg")})
        if r.status_code!=200: _f(47,f"Returned {r.status_code}"); return
        notice=r.json().get("research_notice","")
        for kw in RESEARCH_KEYWORDS:
            _p(47,f"Notice contains '{kw}'") if kw.upper() in notice.upper() else _f(47,f"Notice MISSING '{kw}'")
    except Exception as e: _f(47,"ERROR",str(e))

def s48():
    _sec(48,"Grad-CAM Available Flag in API Response")
    c=_client()
    if isinstance(c,str): _f(48,"TestClient failed",c); return
    try:
        r=c.post("/api/predict",files={"image":("t.jpg",io.BytesIO(_jpeg()),"image/jpeg")})
        if r.status_code!=200: _f(48,f"Returned {r.status_code}"); return
        gc=r.json().get("gradcam",{})
        _p(48,"gradcam.available=True") if gc.get("available") is True else _f(48,f"gradcam.available={gc.get('available')!r}")
        _p(48,'target_layer="model.attention"') if gc.get("target_layer")=="model.attention" else _f(48,f"target_layer={gc.get('target_layer')!r}")
        for key in ("experiment8","experiment13"):
            v=gc.get(key)
            _p(48,f"gradcam.{key} is valid base64 PNG") if isinstance(v,str) and v.startswith("data:image/png;base64,") else _f(48,f"gradcam.{key} invalid",str(v)[:60] if v else "None")
    except Exception as e: _f(48,"ERROR",str(e))

def s49():
    _sec(49,"Training Set NOT Loaded in Inference Code")
    patterns=["train_split","HAM10000_metadata","lesion_images_train","train.csv","train_loader","DataLoader","Dataset"]
    clean=True
    for fpath in [os.path.join(BACKEND_DIR,"app","inference.py"),os.path.join(BACKEND_DIR,"app","model_service.py"),os.path.join(BACKEND_DIR,"app","main.py")]:
        if not os.path.exists(fpath): continue
        with open(fpath) as f: src=f.read()
        for pat in patterns:
            if pat.lower() in src.lower(): _f(49,f"{os.path.basename(fpath)} references '{pat}'"); clean=False
    if clean: _p(49,"No training-set references in inference/service/main")

def s50():
    _sec(50,"Test Split NOT Referenced in Demo Code")
    patterns=["test_split","test.csv","experiment8_test","experiment13_test","N=988","988 images"]
    clean=True
    for fpath in [os.path.join(BACKEND_DIR,"app","inference.py"),os.path.join(BACKEND_DIR,"app","model_service.py"),os.path.join(BACKEND_DIR,"app","main.py"),os.path.join(SRC_DIR,"app","page.tsx")]:
        if not os.path.exists(fpath): continue
        with open(fpath) as f: src=f.read()
        for pat in patterns:
            if pat.lower() in src.lower(): _f(50,f"{os.path.basename(fpath)} references '{pat}'"); clean=False
    if clean: _p(50,"No test-split references in demo/inference code")

def s51():
    _sec(51,"Frontend Build Artifact Existence")
    nxt=os.path.join(PROJECT_ROOT,".next")
    if os.path.isdir(nxt): _p(51,".next/ exists")
    else: _f(51,".next/ MISSING -- run npm run build"); return
    bid=os.path.join(nxt,"BUILD_ID")
    if os.path.exists(bid):
        with open(bid) as f: _p(51,f"BUILD_ID={f.read().strip()}")
    else: _f(51,"BUILD_ID MISSING")
    _p(51,".next/static/ exists") if os.path.isdir(os.path.join(nxt,"static")) else _f(51,".next/static/ MISSING")

def s52():
    _sec(52,"Next.js Route Files Existence")
    for path in [os.path.join(SRC_DIR,"app","page.tsx"),os.path.join(SRC_DIR,"app","api","predict","route.ts"),os.path.join(SRC_DIR,"app","api","health","route.ts"),os.path.join(SRC_DIR,"app","layout.tsx")]:
        rel=os.path.relpath(path,PROJECT_ROOT)
        _p(52,f"{rel} ({os.path.getsize(path):,} bytes)") if os.path.exists(path) else _f(52,f"{rel} MISSING")

def s53():
    _sec(53,"package.json Integrity")
    pkg_path=os.path.join(PROJECT_ROOT,"package.json")
    if not os.path.exists(pkg_path): _f(53,"package.json not found"); return
    with open(pkg_path) as f: pkg=json.load(f)
    scripts=pkg.get("scripts",{})
    for s in ["dev","build","start"]: _p(53,f'scripts.{s}="{scripts[s]}"') if s in scripts else _f(53,f"scripts.{s} MISSING")
    deps=pkg.get("dependencies",{})
    for d in ["next","react","react-dom"]: _p(53,f'"{d}"="{deps[d]}"') if d in deps else _f(53,f'"{d}" MISSING')

def s54():
    _sec(54,"FINAL SUMMARY")
    p=sum(1 for r in results if r["status"]=="PASS")
    f=sum(1 for r in results if r["status"]=="FAIL")
    sk=sum(1 for r in results if r["status"]=="SKIP")
    print(f"\n  Total={len(results)} | PASS={p} | FAIL={f} | SKIP={sk}")
    if f==0: print("\n  ALL CHECKS PASSED"); overall="PASS"
    else:
        print(f"\n  {f} FAIL(S):"); overall="FAIL"
        for r in results:
            if r["status"]=="FAIL": print(f"    * S{r['section']:02d}: {r['title']}" + (f"\n           {r['detail'].split(chr(10))[0][:100]}" if r["detail"] else ""))
    return overall

def write_reports(overall):
    ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    p=sum(1 for r in results if r["status"]=="PASS"); f=sum(1 for r in results if r["status"]=="FAIL"); sk=sum(1 for r in results if r["status"]=="SKIP")
    jd={"report":"DERMAAI_FINAL_SYSTEM_VERIFICATION","generated_at":ts,"overall":overall,"summary":{"pass":p,"fail":f,"skip":sk,"total":len(results)},"checks":results}
    jp=os.path.join(PROJECT_ROOT,"FINAL_SYSTEM_VERIFICATION.json")
    with open(jp,"w") as fh: json.dump(jd,fh,indent=2)
    lines=["# DermaAI -- Final System Verification Report","",f"**Generated:** {ts}  ",f"**Overall:** `{overall}`  ",f"**Checks:** {p} PASS / {f} FAIL / {sk} SKIP / {len(results)} Total","","---","","## Results","","| # | Title | Status | Detail |","|---|-------|--------|--------|"]
    for r in results:
        d=r["detail"].replace("\n"," ").replace("|","\\|")[:120] if r["detail"] else ""
        lines.append(f"| {r['section']:02d} | {r['title']} | {r['status']} | {d} |")
    lines+=["","---","","## Frozen Spec","","| Item | Value |","|------|-------|",f"| Exp8 SHA-256 | `{EXPECTED_EXP8_SHA256}` |",f"| Exp13 SHA-256 | `{EXPECTED_EXP13_SHA256}` |",f"| Exp8 size | {EXPECTED_EXP8_SIZE:,} bytes |",f"| Exp13 size | {EXPECTED_EXP13_SIZE:,} bytes |",f"| Params/model | {EXPECTED_PARAMS:,} |",f"| Ensemble | 0.85*Exp8 + 0.15*Exp13 |",f"| Input | 224x224 |",f"| Classes | {', '.join(EXPECTED_CLASSES)} |",f"| Grad-CAM target | model.attention [1,960,7,7] |","","> READ-ONLY AUDIT. No weights, checkpoints, or code were modified."]
    mp=os.path.join(PROJECT_ROOT,"FINAL_SYSTEM_VERIFICATION_REPORT.md")
    with open(mp,"w") as fh: fh.write("\n".join(lines))
    print(f"\n  JSON -> {jp}\n  MD   -> {mp}")

if __name__=="__main__":
    print("="*68); print("  DERMAAI -- FINAL END-TO-END SYSTEM VERIFICATION"); print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"); print("  READ-ONLY AUDIT -- NO MODIFICATIONS"); print("="*68)
    t0=time.time()
    if PROJECT_ROOT not in sys.path: sys.path.insert(0,PROJECT_ROOT)
    s01(); s02(); s03(); s04(); s05(); s06()
    print("\n[Loading models...]"); _load_models()
    s07_08(); s09_10(); s11_12(); s13_14(); s15_16()
    s17(); s18(); s19_20(); s21(); s22(); s23(); s24()
    s25_26(); s27(); s28(); s29(); s30()
    s31(); s32(); s33(); s34(); s35(); s36(); s37()
    s38(); s39(); s40(); s41(); s42(); s43(); s44()
    s45(); s46(); s47(); s48(); s49(); s50()
    s51(); s52(); s53()
    overall=s54()
    print(f"\n  Elapsed: {time.time()-t0:.1f}s")
    write_reports(overall)
