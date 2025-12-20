# -*- coding: utf-8 -*-
import argparse, os, glob, json
from pathlib import Path
import fitz
import pandas as pd
from collections import Counter

def load_one(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

def load_mapping(xlsx_path):
    df = pd.read_excel(xlsx_path)
    # 兼容不同列名
    cols = {c.lower(): c for c in df.columns}
    k = cols.get('key') or list(df.columns)[0]
    v = cols.get('value') or list(df.columns)[1]
    df = df[[k, v]].rename(columns={k:'t', v:'slide'})
    df = df.dropna()
    df['t'] = df['t'].astype(float)
    df['slide'] = df['slide'].astype(int)
    return df.sort_values('t', kind='mergesort').reset_index(drop=True)

def main(prefix, pdf_path):
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    ocr_xlsx = load_one(prefix + "_ocr_*.xlsx")
    aud_xlsx = load_one(prefix + "_audiomatching_*.xlsx")
    img_xlsx = load_one(prefix + "_image_matching_*.xlsx")
    merged_mean = load_one(prefix + "_mean_matching_all_*.xlsx")
    merged_max  = load_one(prefix + "_max_matching_all_*.xlsx")
    merged_wsum = load_one(prefix + "_weighted_sum_matching_all_*.xlsx")

    rows = []
    for tag, f in [('ocr', ocr_xlsx), ('audio', aud_xlsx), ('image', img_xlsx),
                   ('mean', merged_mean), ('max', merged_max), ('wsum', merged_wsum)]:
        if f:
            df = load_mapping(f)
            rows.append((tag, f, len(df)))
    print("[FOUND]", rows)

    # 统计覆盖 & 分歧
    candidates = [(tag, load_mapping(f)) for tag, f, _ in rows]
    if not candidates:
        print("No mapping xlsx found under prefix.")
        return

    # PDF页数
    n_pages = None
    if os.path.isfile(pdf_path):
        try:
            n_pages = len(fitz.open(pdf_path))
        except Exception:
            pass

    # 对齐到同一时间轴集合
    all_t = sorted({t for _, df in candidates for t in df['t'].tolist()})
    align = {}
    for tag, df in candidates:
        m = dict(zip(df['t'].round(3), df['slide']))
        align[tag] = {round(t,3): m.get(round(t,3), None) for t in all_t}

    # 分歧统计
    disagree = 0
    agree = 0
    votes = []
    for t in all_t:
        vals = [align[tag][round(t,3)] for tag, _ in candidates if align[tag][round(t,3)] is not None]
        if not vals:
            continue
        c = Counter(vals).most_common()
        if len(c) == 1 or (len(c)>1 and c[0][1] > c[1][1]):
            agree += 1
        else:
            disagree += 1
        votes.append((t, dict(Counter(vals))))

    # 覆盖页数（以出现过的 slide 值计）
    covered = set()
    for tag, df in candidates:
        covered |= set(df['slide'].tolist())
    coverage = None
    if n_pages:
        coverage = f"{len(covered)}/{n_pages} ({len(covered)/n_pages:.1%})"
    else:
        coverage = f"{len(covered)} (unknown total pages)"

    # 报告
    report = Path(prefix + "_verify_report.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"PDF pages: {n_pages}\n")
        for tag, fpath, n in rows:
            f.write(f"{tag:>6}: {fpath} | {n} timestamps\n")
        f.write(f"\nCoverage (unique slides matched): {coverage}\n")
        total_pairs = agree + disagree
        if total_pairs:
            f.write(f"Agreement: {agree}/{total_pairs} ({agree/max(1,total_pairs):.1%})\n")
            f.write(f"Disagreement: {disagree}/{total_pairs} ({disagree/max(1,total_pairs):.1%})\n")
        f.write("\nSamples of votes (first 20):\n")
        for t, v in votes[:20]:
            f.write(f"  t={t:.2f}s -> {v}\n")
    print("[OK] wrote", report)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="结果前缀，如 E:\\Code\\outputs\\mavils_lecture01")
    ap.add_argument("--pdf_path", required=True, help="同一场的PDF路径")
    args = ap.parse_args()
    main(args.prefix, args.pdf_path)
