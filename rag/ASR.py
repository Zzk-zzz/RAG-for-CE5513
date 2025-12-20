# -*- coding: utf-8 -*-
# E:\Code\rag\ASR.py

import os, sys, re, json, subprocess, cv2, numpy as np, shutil
from pathlib import Path
from PIL import Image
import imagehash
from tqdm import tqdm

# =========================
# CUDA / cuDNN DLL 注入 & 自检（在 import faster_whisper 之前）
# =========================
def _add_cuda_dll_dirs_and_check():
    added = []
    sp = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    cand_dirs = [
        sp / "cuda_runtime" / "bin",
        sp / "cublas" / "bin",
        sp / "cudnn" / "bin",
    ]
    for d in cand_dirs:
        try:
            if d.exists():
                os.add_dll_directory(str(d))
                added.append(str(d))
        except Exception:
            pass
    if added:
        print("[NVIDIA] DLL search paths added:")
        for p in added:
            print("   -", p)

    def _check(name: str):
        try:
            __import__("ctypes").WinDLL(name)
            print(f"[CHECK] {name} ... OK")
            return True
        except Exception as e:
            print(f"[CHECK] {name} ... FAIL ({e})")
            return False

    _check("cudart64_12.dll")
    _check("cublas64_12.dll")
    _check("cublasLt64_12.dll")
    _check("cudnn64_9.dll")
    _check("cudnn_ops64_9.dll")
    ok_infer = _check("cudnn_cnn_infer64_9.dll")
    # 缺 infer 子库 → 关闭 cuDNN 路径，faster-whisper 仍可走 GPU（无 cuDNN）
    if not ok_infer:
        print("[NVIDIA] cudnn_cnn_infer64_9.dll missing; set CT2_DISABLE_CUDNN=1 (GPU will still be used).")
        os.environ.setdefault("CT2_DISABLE_CUDNN", "1")

_add_cuda_dll_dirs_and_check()

from faster_whisper import WhisperModel

# -------------------- 环境缓存（放 E 盘） --------------------
os.environ.setdefault("HF_HOME", r"E:\AIcache\hf")
os.environ.setdefault("TRANSFORMERS_CACHE", r"E:\AIcache\hf\transformers")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", r"E:\AIcache\sentence")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# -------------------- 关键路径 --------------------
BASE      = r"E:\Code\pdf2latex_formal"
LATEX     = os.path.join(BASE, "merged_1-4.tex")       # 含四讲的 LaTeX（带 % <label>_page_N 标记）
TEMP_PDFS = os.path.join(BASE, "temp_pages")
IMAGES    = os.path.join(BASE, "images")
SLIDE_PNG = os.path.join(BASE, "slide_images")         # 渲染页 PNG（命名为 <page_label>.png）

VIDEO_PATHS = [
    r"E:\Code\Wednesday_ 15 January 2025 at 5_54_50 pm_default_8ee4e38d.mp4",
    r"E:\Code\Wednesday_ 22 January 2025 at 5_57_29 pm_default_ea687d7b.mp4",
]

OUT_DIR      = r"E:\Code\outputs"
PROFILES     = os.path.join(BASE,    "page_profiles.jsonl")
ENRICHED     = os.path.join(OUT_DIR, "page_profiles_enriched.jsonl")
ALIGNMENTS   = os.path.join(OUT_DIR, "alignments.jsonl")
TEX_ANNOTATE = os.path.join(OUT_DIR, "merged_with_asr.tex")

POPPLER_ROOT = r"E:\Code\poppler-25.07.0\Library\bin"
WHISPER_MODEL_DIR = r"E:\AIcache\whisper_models"

# -------------------- 讲次白名单（双保险；03/04 必须下划线） --------------------
LECTURE_PREFIXES = [
    "Lecture 01_",
    "Lecture_02_",
    "Lecture_03_",
    "Lecture_04_",
]
def _keep_label(label: str) -> bool:
    return any(label.startswith(pfx) for pfx in LECTURE_PREFIXES)

# -------------------- 主要参数（兼顾完整与稳健） --------------------
LEFT_FRACTION        = 0.66   # 左 2/3 是 PPT
SAMPLE_FPS           = 2      # 帧哈希采样频率（页锚更密集）
HASH_FUNC            = imagehash.phash
HASH_THRESHOLD       = 10     # 略放宽匹配阈值，捕更多页
MIN_STABLE_SEC       = 2.0
MERGE_GAP_SEC        = 3.0

# Whisper/ASR：优先 GPU，失败回退 CPU；贪心解码；强制英文；注入术语
ASR_MODEL            = "small.en"
ASR_DEVICE_PREFS     = ["cuda", "cpu"]   # 依次尝试
ASR_COMPUTE_FOR_DEV  = {"cuda": "float16", "cpu": "int8"}
ASR_BEAM_SIZE        = 1                 # 贪心
ASR_LANGUAGE         = "en"
ASR_INITIAL_PROMPT   = (
    "CE5513 plastic analysis of structures, plastic hinge, collapse mechanism, "
    "Ronan Point, Hyatt Regency walkway, yield, moment capacity, kinematic approach, "
    "limit analysis, interaction diagram, incremental elastoplastic analysis, linear programming, "
    "finite element analysis, beam section behaviour"
)

# 写回策略：确保零丢失
ASSIGN_OFFSLIDE_TO_NEAREST = True     # 未命中 → 贴最近页
NEAREST_ASSIGN_MAX_GAP     = 120.0    # 秒，先取大，保证不漏
FORCE_ASSIGN_WHEN_NO_PAGE  = True     # 仍无匹配 → 强行贴“绝对最近页”

# 阶段开关（避免重复渲染 PNG）
STAGES = dict(
    build_profiles=False,
    render_slide_png=False,
    align_and_asr=True,
    write_tex=True,
    build_asr_index=False,
    demo_gemini=False
)

# -------------------- 小工具 --------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def write_jsonl(path: str, records):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def find_poppler_exe(root, names=("pdftoppm.exe","pdftoppm","pdftocairo.exe","pdftocairo")):
    if root and os.path.exists(root):
        for n in names:
            p = os.path.join(root, n)
            if os.path.exists(p):
                return p
        for dp, _, files in os.walk(root):
            for n in names:
                if n in files:
                    return os.path.join(dp, n)
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None

# -------------------- A: LaTeX → page_profiles --------------------
PAGE_MARKER_RE = re.compile(r"^%\s*(.*?)_page_(\d+)\s*$", flags=re.MULTILINE)
INCG_RE        = re.compile(r"\\includegraphics(?:\[[^\]]*\])?{([^}]+)}")
CMD_RE         = re.compile(r"\\[a-zA-Z@]+(\s*\[[^\]]*\])?(\s*{[^{}]*})?")
MATH_RE        = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")
BRACE_RE       = re.compile(r"[{}]")

def tex_to_plain(page_tex: str) -> str:
    t = MATH_RE.sub(" ", page_tex)
    t = CMD_RE.sub(" ", t)
    t = BRACE_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def find_pages(tex: str):
    ms = list(PAGE_MARKER_RE.finditer(tex))
    pages = []
    if not ms:
        parts = re.split(r"\\clearpage", tex)
        for i, chunk in enumerate(parts, 1):
            chunk = chunk.strip()
            if chunk:
                pages.append((f"page_{i:03d}", chunk, f"page_{i}"))
        return pages
    for i, m in enumerate(ms):
        start = m.end()
        end = ms[i+1].start() if i+1 < len(ms) else len(tex)
        base, num = m.group(1), int(m.group(2))
        page_id = f"page_{num:03d}"
        page_label = f"{base}_page_{num}"
        body = tex[start:end].strip()
        if body:
            pages.append((page_id, body, page_label))
    return pages

def build_profiles():
    tex = read_text(LATEX)
    pages = find_pages(tex)
    recs = []
    for pid, body, label in pages:
        if not _keep_label(label):
            continue
        images = [n.strip().rstrip('}').strip() for n in INCG_RE.findall(body)]
        plain  = tex_to_plain(body)
        preview = plain[:360]
        recs.append({
            "page_id": pid,
            "page_label": label,
            "text_chars": len(plain),
            "text_preview": preview,
            "images": images,
            "source": {"latex_file": LATEX, "images_dir": IMAGES},
            "asr_spans": [],
            "offslide_spans": [],
        })
    write_jsonl(PROFILES, recs)
    print(f"[OK] page_profiles.jsonl -> {PROFILES} ({len(recs)} pages)")

# -------------------- B: PDF → PNG --------------------
def render_temp_pages_to_png():
    exe = find_poppler_exe(POPPLER_ROOT)
    if not exe:
        raise RuntimeError("找不到 pdftoppm/pdftocairo，请检查 POPPLER_ROOT。")
    ensure_dir(SLIDE_PNG)
    pdfs = sorted(Path(TEMP_PDFS).glob("*.pdf"))
    if not pdfs:
        print(f"[WARN] {TEMP_PDFS} 下未找到逐页 PDF。")
        return

    exe_name = os.path.basename(exe).lower()
    use_cairo = exe_name.startswith("pdftocairo")

    for p in tqdm(pdfs, desc="Rendering pages"):
        m = re.match(r"^(.*)_page_(\d+)\.pdf$", p.name, flags=re.IGNORECASE)
        if m:
            title = m.group(1); num = int(m.group(2))
            page_label = f"{title}_page_{num}"
        else:
            page_label = p.stem

        if not _keep_label(page_label):
            continue

        outprefix = os.path.join(SLIDE_PNG, page_label)
        target = Path(outprefix + ".png")
        if target.exists():
            try: target.unlink()
            except Exception: pass

        if use_cairo:
            cmd = [exe, "-png", "-r", "180", str(p), outprefix + ".png"]
            subprocess.run(cmd, check=True)
        else:
            try:
                cmd = [exe, "-png", "-r", "180", "-singlefile", str(p), outprefix]
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                cmd = [exe, "-png", "-r", "180", str(p), outprefix]
                subprocess.run(cmd, check=True)
                gen = Path(outprefix + "-1.png")
                if gen.exists():
                    if target.exists():
                        try: target.unlink()
                        except Exception: pass
                    gen.replace(target)

    n = len(list(Path(SLIDE_PNG).glob("*.png")))
    print(f"[OK] slide images -> {SLIDE_PNG} ({n} pngs)")

# -------------------- C: 对齐 + ASR --------------------
def load_profiles_json():
    profs = []
    if not os.path.exists(PROFILES):
        return profs
    with open(PROFILES, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                profs.append(json.loads(line))
    return profs

def load_ref_hashes():
    refs = []
    if not os.path.isdir(SLIDE_PNG):
        return refs
    for p in sorted(Path(SLIDE_PNG).glob("*.png")):
        if not _keep_label(p.stem):
            continue
        img = Image.open(str(p)).convert("L")
        refs.append((p.stem, HASH_FUNC(img)))  # stem = page_label
    return refs

def iter_frame_hashes(video_path: str, sample_fps: int, left_fraction: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Open video failed: {video_path}")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step  = max(int(fps // sample_fps), 1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cut_w = int(width * left_fraction)
    hashes = []
    idx = 0
    with tqdm(total=(total//step + 1), desc=f"Frames {Path(video_path).name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            if idx % step == 0:
                crop = frame[:, :cut_w]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                h = HASH_FUNC(Image.fromarray(gray))
                t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
                hashes.append((t, h))
                pbar.update(1)
            idx += 1
    cap.release()
    return hashes

def match_to_pages(frame_hashes, ref_hashes):
    hits = {}
    for (t, h) in frame_hashes:
        best, dist = None, 1e9
        for name, rh in ref_hashes:
            d = h - rh
            if d < dist:
                best, dist = name, d
        if best is not None and dist <= HASH_THRESHOLD:
            hits.setdefault(best, []).append(t)
    spans = []
    for name, ts in hits.items():
        ts = sorted(ts)
        s = ts[0]; prev = ts[0]
        for cur in ts[1:]:
            if cur - prev > (1.0 / max(SAMPLE_FPS,1)) * 2.5:
                if prev - s >= MIN_STABLE_SEC:
                    spans.append((name, s, prev))
                s = cur
            prev = cur
        if prev - s >= MIN_STABLE_SEC:
            spans.append((name, s, prev))
    merged = []
    for name in sorted(set(n for (n,_,_) in spans)):
        segs = sorted([(s,e) for (n,s,e) in spans if n==name])
        cur_s, cur_e = segs[0]
        for (s,e) in segs[1:]:
            if s - cur_e <= MERGE_GAP_SEC:
                cur_e = max(cur_e, e)
            else:
                merged.append((name, cur_s, cur_e))
                cur_s, cur_e = s,e
        merged.append((name, cur_s, cur_e))
    # (label, start, end, conf)
    return [(lab, s, e, 0.9) for (lab,s,e) in merged]

def _init_whisper():
    last_err = None
    for dev in ASR_DEVICE_PREFS:
        try:
            print(f"[ASR] trying device={dev} compute={ASR_COMPUTE_FOR_DEV[dev]} model={ASR_MODEL}")
            m = WhisperModel(
                ASR_MODEL,
                device=dev,
                compute_type=ASR_COMPUTE_FOR_DEV[dev],
                download_root=WHISPER_MODEL_DIR,
                num_workers=1
            )
            print(f"[ASR] OK on device={dev}")
            return m, dev
        except Exception as e:
            print(f"[ASR] fail on {dev}: {e}")
            last_err = e
    raise RuntimeError(f"Whisper init failed on all devices: {last_err}")

def transcribe(video_path: str):
    model, dev = _init_whisper()
    segments, info = model.transcribe(
        video_path,
        beam_size=ASR_BEAM_SIZE,        # 贪心（=1）够快
        vad_filter=True,
        language=ASR_LANGUAGE,          # 强制英文
        task="transcribe",
        condition_on_previous_text=True,
        initial_prompt=ASR_INITIAL_PROMPT
    )
    out = []
    for seg in segments:
        out.append({"t_start": float(seg.start), "t_end": float(seg.end), "text": seg.text.strip()})
    print(f"[ASR] {Path(video_path).name}: {len(out)} segments  (device={dev})")
    return out

def _nearest_page_and_gap(mid_t: float, page_spans):
    """
    返回 (最邻近 span, 距离 gap秒)；若命中区间则 gap=0
    """
    best = None
    best_gap = 1e12
    for (lab, t0, t1, conf) in page_spans:
        if t0 <= mid_t <= t1:
            return (lab, t0, t1, conf), 0.0
        gap = min(abs(mid_t - t0), abs(mid_t - t1))
        if gap < best_gap:
            best_gap = gap
            best = (lab, t0, t1, conf)
    return best, best_gap

def assign_asr(asr, page_spans):
    out = []
    for s in asr:
        mid = 0.5*(s["t_start"] + s["t_end"])
        # 1) 命中某页区间
        cands = [(lab,t0,t1,conf) for (lab,t0,t1,conf) in page_spans if t0 <= mid <= t1]
        if cands:
            cands.sort(key=lambda x: (x[2]-x[1]), reverse=True)  # 取区间更长者
            lab, t0, t1, conf = cands[0]
            ss = dict(s); ss["page_label_guess"] = lab; ss["page_conf"] = conf
            out.append(ss); continue

        # 2) 最近页（阈值内）
        nn, gap = _nearest_page_and_gap(mid, page_spans) if page_spans else (None, 1e12)
        if nn and gap <= NEAREST_ASSIGN_MAX_GAP:
            lab, t0, t1, conf = nn
            ss = dict(s); ss["page_label_guess"] = lab; ss["page_conf"] = conf
            out.append(ss); continue

        # 3) 兜底：仍无匹配 → 强行贴绝对最近页（保证零丢失）
        if FORCE_ASSIGN_WHEN_NO_PAGE and nn:
            lab, t0, t1, conf = nn
            ss = dict(s); ss["page_label_guess"] = lab; ss["page_conf"] = 0.0
            out.append(ss); continue

        # 4) 实在无页（几乎不会发生）→ 保留未贴页
        ss = dict(s); ss["page_label_guess"] = None; ss["page_conf"] = 0.0
        out.append(ss)
    return out

def align_and_asr():
    ensure_dir(OUT_DIR)
    profiles = load_profiles_json()
    enriched = {p["page_label"]: p for p in profiles}
    for p in enriched.values():
        p["asr_spans"] = []
        p["offslide_spans"] = []

    ref = load_ref_hashes()
    if not ref:
        print("[WARN] 未找到 slide_images/*.png，视觉匹配不会进行。")

    all_align = []
    for v in VIDEO_PATHS:
        print(f"\n== Video: {v}")
        fh = iter_frame_hashes(v, SAMPLE_FPS, LEFT_FRACTION)
        spans = match_to_pages(fh, ref) if ref else []
        asr = transcribe(v)
        assigned = assign_asr(asr, spans) if spans else [
            {"t_start":a["t_start"],"t_end":a["t_end"],"text":a["text"],"page_label_guess":None,"page_conf":0.0}
            for a in asr
        ]
        for s in assigned:
            lab = s.get("page_label_guess")
            rec = {
                "video": v,
                "t_start": s["t_start"],
                "t_end": s["t_end"],
                "text": s["text"],
                "conf": s.get("page_conf", 0.0),
            }
            if lab and lab in enriched:
                enriched[lab]["asr_spans"].append(rec)
            else:
                enriched.setdefault("__OFFSLIDE__", {"page_label":"__OFFSLIDE__","asr_spans":[],"offslide_spans":[]})
                enriched["__OFFSLIDE__"]["asr_spans"].append(rec)

        all_align.append({
            "video": v,
            "page_spans": [{"label":lab,"t_start":s,"t_end":e,"conf":c} for (lab,s,e,c) in spans],
            "asr": assigned
        })

    write_jsonl(ALIGNMENTS, all_align)
    write_jsonl(ENRICHED, [v for k,v in enriched.items() if k!="__OFFSLIDE__"])
    print(f"[OK] alignments -> {ALIGNMENTS}")
    print(f"[OK] enriched profiles -> {ENRICHED}")

# -------------------- D: 写回 LaTeX（零省略，按时序全写） --------------------
SPECIAL_MAP = {
    '\\': r'\textbackslash{}',
    '&':  r'\&',
    '%':  r'\%',
    '$':  r'\$',
    '#':  r'\#',
    '_':  r'\_',
    '{':  r'\{',
    '}':  r'\}',
    '~':  r'\textasciitilde{}',
    '^':  r'\textasciicircum{}',
}
def latex_escape(s: str) -> str:
    return ''.join(SPECIAL_MAP.get(ch, ch) for ch in s)

def fmt_time(t):
    t = int(t)
    h = t//3600; m=(t%3600)//60; s=t%60
    return f"{h:02d}:{m:02d}:{s:02d}" if h>0 else f"{m:02d}:{s:02d}"

def write_tex():
    tex = read_text(LATEX)

    # page_label -> 全量 ASR（按时间排序，无上限）
    page2asr = {}
    with open(ENRICHED, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            page2asr[p["page_label"]] = sorted(
                p.get("asr_spans", []),
                key=lambda x: (x["t_start"], x["t_end"])
            )

    out = []
    last = 0
    for m in PAGE_MARKER_RE.finditer(tex):
        out.append(tex[last:m.end()])

        # 插在 \clearpage 之后
        after_marker_pos = m.end()
        clear_m = re.search(r'\\clearpage', tex[after_marker_pos:])
        insert_pos = after_marker_pos + (clear_m.end() if clear_m else 0)
        out.append(tex[after_marker_pos:insert_pos])

        label = f"{m.group(1)}_page_{int(m.group(2))}"
        if not _keep_label(label):
            last = insert_pos
            continue

        spans = page2asr.get(label, [])
        if spans:
            block_parts = []
            for s in spans:
                t0, t1 = fmt_time(s["t_start"]), fmt_time(s["t_end"])
                txt = latex_escape(s["text"])
                block_parts.append(
                    f"\n% {label} | t={t0}-{t1}\n"
                    f"\\paragraph*{{ASR}}\n"
                    f"\\begin{{quote}}\\small {txt}\\end{{quote}}\n"
                )
            out.append("".join(block_parts))
        last = insert_pos

    out.append(tex[last:])
    ensure_dir(os.path.dirname(TEX_ANNOTATE))
    with open(TEX_ANNOTATE, "w", encoding="utf-8") as f:
        f.write("".join(out))
    print(f"[OK] wrote annotated LaTeX -> {TEX_ANNOTATE}")

# -------------------- （可选）ASR 索引 + Demo --------------------
def build_asr_index():
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except Exception as e:
        print("[WARN] 缺少 sentence-transformers 或 faiss-cpu：", e)
        return None

    docs, meta = [], []
    with open(ENRICHED, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            p = json.loads(line)
            label = p["page_label"]
            if not _keep_label(label): continue
            for sp in p.get("asr_spans", []):
                docs.append(sp["text"])
                meta.append({"page_label": label, "t_start": sp["t_start"], "t_end": sp["t_end"]})
    if not docs:
        print("[WARN] 暂无 ASR 片段可索引。")
        return None

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    import faiss
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    print(f"[OK] ASR index built: {len(docs)} segments")
    return dict(index=index, emb=emb, docs=docs, meta=meta)

def demo_ask_gemini(store, query, top_k=10):
    if store is None:
        print("[ERR] 没有可用索引。")
        return
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    D, I = store["index"].search(q, top_k)
    I = I[0].tolist(); D = D[0].tolist()
    parts, cited = [], []
    for rank, (i, score) in enumerate(zip(I, D), 1):
        text = store["docs"][i]; meta = store["meta"][i]
        label = meta["page_label"]; 
        parts.append(f"[{rank}] source={label}\n{text}")
        cited.append(label)
    context = "\n\n---\n\n".join(parts)
    prompt = (
        "You are a precise assistant for CE5513 lectures.\n"
        "Answer ONLY using the provided lecture transcript excerpts. If not enough, say you don't know.\n"
        "Cite page labels explicitly when relevant.\n\n"
        f"Excerpts:\n{context}\n\nQuestion: {query}\n"
    )
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("[WARN] GEMINI_API_KEY 未设置。")
        print("(Relevant sources: " + ", ".join(sorted(set(cited))) + ")")
        return
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        print("\n[Answer]\n" + resp.text.strip())
    except Exception as e:
        print("[ERR] Gemini 失败：", e)
        print("(Relevant sources: " + ", ".join(sorted(set(cited))) + ")")

# -------------------- Main --------------------
if __name__ == "__main__":
    print("== CE5513 ASR + Alignment ==")
    if STAGES["build_profiles"]:
        build_profiles()
    if STAGES["render_slide_png"]:
        render_temp_pages_to_png()
    if STAGES["align_and_asr"]:
        align_and_asr()
    if STAGES["write_tex"]:
        write_tex()

    store = None
    if STAGES["build_asr_index"]:
        store = build_asr_index()
    if STAGES["demo_gemini"]:
        demo_ask_gemini(store, query="What is a plastic hinge and why does redundancy matter?")
    print("== Done ==")
