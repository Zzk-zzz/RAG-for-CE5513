# qa_from_latex.py
# RAG-style Q&A over a page-marked LaTeX file (e.g., merged_with_asr_mavils.tex)
# - Local embeddings (Sentence-Transformers) + FAISS retrieval
# - Gemini generation (reads API key from env or gemini_key.txt next to this script / cwd)
# - Removes inline "(Lecture_xxx_page_n,...)" citations from the model output and appends sources at the end
#
# Usage:
#   python qa_from_latex.py --tex "E:\Code\outputs\merged_with_asr_mavils.tex"
#   python qa_from_latex.py --tex "E:\Code\outputs\merged_with_asr_mavils.tex" --pdf "E:\Code\pdf2latex_formal\merged_1-4.pdf"
#
# Requirements:
#   pip install -U sentence-transformers faiss-cpu google-genai numpy

from __future__ import annotations

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

# -----------------------------
# Defaults (override by CLI)
# -----------------------------
DEFAULT_TEX = r"E:\Code\outputs\merged_with_asr_mavils.tex"
DEFAULT_PDF = ""  # optional

DEFAULT_RETRIEVE_TOP_K = 15
DEFAULT_CONTEXT_TOP_N = 10
DEFAULT_MAX_CHARS_PER_CHUNK = 1400
DEFAULT_MAX_TOTAL_CHARS = 14000

DEFAULT_PARTS_PER_PAGE = 2
DEFAULT_OVERLAP_RATIO = 0.30

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "gemini-2.5-flash"


# -----------------------------
# Key loading (env -> gemini_key.txt)
# -----------------------------
def load_gemini_key() -> str:
    k = os.getenv("GEMINI_API_KEY", "").strip()
    if k:
        return k

    here = Path(__file__).resolve().parent
    p = here / "gemini_key.txt"
    if p.exists():
        return p.read_text(encoding="utf-8", errors="ignore").strip()

    p2 = Path.cwd() / "gemini_key.txt"
    if p2.exists():
        return p2.read_text(encoding="utf-8", errors="ignore").strip()

    p3 = Path.home() / ".gemini_key.txt"
    if p3.exists():
        return p3.read_text(encoding="utf-8", errors="ignore").strip()

    return ""


# -----------------------------
# LaTeX helpers
# -----------------------------
PAGE_MARKER_RE = re.compile(r"^%\s*(.*?)_page_(\d+)\s*$", flags=re.MULTILINE)
INCG_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?{([^}]+)}")
MATH_RE = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")
CMD_RE = re.compile(r"\\[a-zA-Z@]+(\s*\[[^\]]*\])?(\s*{[^{}]*})?")
BRACE_RE = re.compile(r"[{}]")


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def find_pages(tex: str) -> List[Tuple[str, str, str, int]]:
    """
    Split by '% <label>_page_<num>' markers.
    Returns: [(page_id, page_text, page_label, page_num)]
    """
    markers = list(PAGE_MARKER_RE.finditer(tex))
    pages: List[Tuple[str, str, str, int]] = []
    if markers:
        for i, m in enumerate(markers):
            start = m.end()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(tex)
            base = m.group(1)
            num = int(m.group(2))
            page_id = f"page_{num:03d}"
            page_label = f"{base}_page_{num}"
            body = tex[start:end].strip()
            if body:
                pages.append((page_id, body, page_label, num))
    else:
        # fallback: split by \clearpage
        parts = re.split(r"\\clearpage", tex)
        for i, chunk in enumerate(parts, start=1):
            chunk = chunk.strip()
            if chunk:
                page_id = f"page_{i:03d}"
                pages.append((page_id, chunk, page_id, i))
    return pages


def extract_section_title(page_text: str) -> str:
    pat = re.compile(r"\\(?:sub)?section\*{([^}]*)}")
    found = pat.findall(page_text)
    return (found[-1].strip() if found else "Unknown Section")


def clean_for_embedding(page_tex: str) -> str:
    t = INCG_RE.sub(" IMAGE ", page_tex)
    t = MATH_RE.sub(" MATH ", t)
    t = CMD_RE.sub(" ", t)
    t = BRACE_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def clean_for_context(page_tex: str) -> str:
    t = re.sub(r"\\includegraphics(?:\[[^\]]*\])?{[^}]+}", "", page_tex)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def chunk_by_page(tex: str, parts_per_page: int, overlap_ratio: float) -> List[Dict[str, Any]]:
    pages = find_pages(tex)
    out: List[Dict[str, Any]] = []

    for page_id, page_tex, page_label, page_num in pages:
        section = extract_section_title(page_tex)

        embed_base = clean_for_embedding(page_tex)
        raw_base = clean_for_context(page_tex)

        words = embed_base.split()
        if not words:
            continue

        n = len(words)
        part_size = max(1, n // max(parts_per_page, 1))
        overlap = int(part_size * max(0.0, min(overlap_ratio, 0.95)))

        i = 0
        while i < n:
            start = i
            end = min(i + part_size, n)
            embed_chunk = " ".join(words[start:end]).strip()
            if embed_chunk:
                out.append(
                    dict(
                        embed_text=embed_chunk,
                        raw_text=raw_base,
                        page_id=page_id,
                        page_label=page_label,
                        page_num=int(page_num),
                        section=section,
                    )
                )
            if end == n:
                break
            i = end - overlap

    return out


# -----------------------------
# Embeddings + FAISS
# -----------------------------
_local_model = None


def _ensure_embed_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _local_model


def embed_texts(texts: List[str]) -> np.ndarray:
    model = _ensure_embed_model()
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype(np.float32)


def build_index(vectors: np.ndarray):
    import faiss
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def search(index, query: str, top_k: int):
    q = embed_texts([query])
    D, I = index.search(q, top_k)
    hits = []
    for j, idx in enumerate(I[0].tolist()):
        if idx < 0:
            continue
        hits.append((idx, float(D[0][j])))
    return hits


# -----------------------------
# Context building + Answering
# -----------------------------
def safe_trim(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + " ..."


def dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x2 = x.strip()
        if not x2:
            continue
        if x2 not in seen:
            seen.add(x2)
            out.append(x2)
    return out


def build_context(
    chunks: List[Dict[str, Any]],
    hit_indices: List[int],
    context_top_n: int,
    max_chars_per_chunk: int,
    max_total_chars: int,
) -> Tuple[str, List[str], List[int]]:
    chosen = hit_indices[:context_top_n]
    source_labels = [chunks[i]["page_label"] for i in chosen]
    page_nums = [int(chunks[i]["page_num"]) for i in chosen]

    parts = []
    total = 0
    for k, i in enumerate(chosen, start=1):
        label = chunks[i]["page_label"]
        section = chunks[i]["section"]
        raw = safe_trim(chunks[i]["raw_text"], max_chars_per_chunk)

        piece = f"[{k}] source={label} | section={section}\n{raw}"
        if total + len(piece) > max_total_chars:
            parts.append("...[truncated due to context length]...")
            break
        parts.append(piece)
        total += len(piece) + 2

    return "\n\n---\n\n".join(parts), dedup_preserve_order(source_labels), page_nums


_CIT_PAREN_RE = re.compile(r"\(([^)]{1,250})\)")


def strip_inline_citations(answer_text: str) -> Tuple[str, List[str]]:
    collected: List[str] = []

    def repl(m: re.Match) -> str:
        inner = m.group(1)
        if "_page_" not in inner:
            return m.group(0)
        for part in inner.split(","):
            s = part.strip()
            if "_page_" in s:
                collected.append(s)
        return ""

    cleaned = _CIT_PAREN_RE.sub(repl, answer_text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s+\n", "\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned, dedup_preserve_order(collected)


def gemini_generate(prompt: str, model_name: str) -> str:
    api_key = load_gemini_key()
    if not api_key:
        raise RuntimeError("Gemini API key not found (set GEMINI_API_KEY or create gemini_key.txt next to this script).")

    try:
        from google import genai
    except Exception as e:
        raise RuntimeError("google-genai not installed. Run: pip install -U google-genai") from e

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model=model_name, contents=prompt)
    text = getattr(resp, "text", None)
    if not text:
        return str(resp)
    return text.strip()


def make_open_pdf_hint(pdf_path: str, page_num: int) -> str:
    return f'start "" "{pdf_path}#page={page_num}"'


def answer_question(
    query: str,
    chunks: List[Dict[str, Any]],
    index,
    pdf_path: str,
    model_name: str,
    retrieve_top_k: int,
    context_top_n: int,
    max_chars_per_chunk: int,
    max_total_chars: int,
) -> str:
    hits = search(index, query, retrieve_top_k)
    hit_indices = [i for (i, _score) in hits]

    print("\n[Retrieved]")
    for rank, (i, score) in enumerate(hits, start=1):
        label = chunks[i]["page_label"]
        sec = chunks[i]["section"]
        preview = safe_trim(chunks[i]["embed_text"], 140).replace("\n", " ")
        extra = ""
        if pdf_path:
            extra = f' | open: {make_open_pdf_hint(pdf_path, int(chunks[i]["page_num"]))}'
        print(f"#{rank} score={score:.4f} source={label} section={sec} preview={preview}{extra}")

    context_text, retrieved_sources, _retrieved_pages = build_context(
        chunks,
        hit_indices,
        context_top_n=context_top_n,
        max_chars_per_chunk=max_chars_per_chunk,
        max_total_chars=max_total_chars,
    )

    prompt = (
        "You are a precise assistant for structural mechanics lecture notes.\n"
        "Answer the question using ONLY the provided excerpts.\n"
        "Do NOT put source citations in-line.\n"
        "At the very end, output a single line that starts with 'SOURCES: ' followed by comma-separated source labels.\n"
        "If the excerpts do not contain the answer, say you don't know, and still output 'SOURCES: ' with the best-matching sources.\n\n"
        f"EXCERPTS:\n{context_text}\n\n"
        f"QUESTION: {query}\n"
    )

    raw = gemini_generate(prompt, model_name=model_name)

    # 1) strip inline parentheses citations anyway (if model disobeys)
    cleaned, inline_sources = strip_inline_citations(raw)

    # 2) parse SOURCES line if present
    sources: List[str] = []
    m = re.search(r"(?im)^\s*SOURCES:\s*(.+?)\s*$", cleaned)
    if m:
        tail = m.group(1).strip()
        cleaned = re.sub(r"(?im)^\s*SOURCES:\s*.+?\s*$", "", cleaned).strip()
        for part in tail.split(","):
            s = part.strip()
            if s:
                sources.append(s)

    # 3) unify sources (prefer model -> inline -> retrieved)
    if sources:
        sources = dedup_preserve_order(sources)
    elif inline_sources:
        sources = dedup_preserve_order(inline_sources)
    else:
        sources = dedup_preserve_order(retrieved_sources)

    out = cleaned.strip()
    if sources:
        out = out + "\n\n(" + ", ".join(sources) + ")"

    # Optional: add "open PDF page" helper commands
    if pdf_path and sources:
        page_hints = []
        for s in sources[:6]:
            mm = re.search(r"_page_(\d+)\s*$", s)
            if mm:
                page_hints.append(make_open_pdf_hint(pdf_path, int(mm.group(1))))
        if page_hints:
            out += "\n\n[Open PDF]\n" + "\n".join(page_hints)

    return out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=DEFAULT_TEX, help="Path to the LaTeX file (page-marked).")
    ap.add_argument("--pdf", default=DEFAULT_PDF, help="Optional PDF path for 'open page' hints.")
    ap.add_argument("--model", default=GEN_MODEL, help="Gemini model name.")
    ap.add_argument("--k", type=int, default=DEFAULT_RETRIEVE_TOP_K, help="Retrieve top K.")
    ap.add_argument("--n", type=int, default=DEFAULT_CONTEXT_TOP_N, help="Context top N.")
    ap.add_argument("--parts_per_page", type=int, default=DEFAULT_PARTS_PER_PAGE)
    ap.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP_RATIO)
    ap.add_argument("--max_chars_per_chunk", type=int, default=DEFAULT_MAX_CHARS_PER_CHUNK)
    ap.add_argument("--max_total_chars", type=int, default=DEFAULT_MAX_TOTAL_CHARS)
    args = ap.parse_args()

    tex_path = args.tex
    if not Path(tex_path).exists():
        print(f"[ERR] LaTeX file not found: {tex_path}")
        sys.exit(1)

    tex = read_text(tex_path)
    chunk_objs = chunk_by_page(tex, parts_per_page=args.parts_per_page, overlap_ratio=args.overlap)

    if not chunk_objs:
        print("[ERR] No chunks produced. Check page markers or input LaTeX.")
        sys.exit(1)

    embed_texts_list = [c["embed_text"] for c in chunk_objs]
    print(f"[INFO] pages={len({c['page_label'] for c in chunk_objs})} chunks={len(chunk_objs)} embedding_model={EMBED_MODEL_NAME}")

    vecs = embed_texts(embed_texts_list)
    index = build_index(vecs)
    print("[INFO] FAISS index built.\n")

    pdf_path = (args.pdf or "").strip()
    if pdf_path and not Path(pdf_path).exists():
        print(f"[WARN] PDF path not found (will ignore): {pdf_path}")
        pdf_path = ""

    while True:
        q = input("Question (q to quit): ").strip()
        if not q:
            continue
        if q.lower() == "q":
            break

        try:
            ans = answer_question(
                q,
                chunk_objs,
                index,
                pdf_path=pdf_path,
                model_name=args.model,
                retrieve_top_k=args.k,
                context_top_n=args.n,
                max_chars_per_chunk=args.max_chars_per_chunk,
                max_total_chars=args.max_total_chars,
            )
        except Exception as e:
            print(f"\n[ERR] {e}")
            continue

        print("\n[Answer]\n" + ans)
        print("=" * 80)


if __name__ == "__main__":
    main()
