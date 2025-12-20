# E:\Code\rag\quick_check.py
import json, os, re, sys
ENRICHED = r"E:\Code\outputs\page_profiles_enriched.jsonl"

def non_ascii_ratio(s): 
    if not s: return 0.0
    na = sum(1 for ch in s if ord(ch) > 127)
    return na/len(s)

pages = {}
with open(ENRICHED, "r", encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if not line: continue
        p=json.loads(line)
        lab=p["page_label"]
        spans=sorted(p.get("asr_spans",[]), key=lambda x:(x["t_start"],x["t_end"]))
        pages[lab]=spans

print(f"[OK] loaded {len(pages)} pages from enriched JSONL")
bad = []
gap_thr = 1.0  # 秒：统计超过这个阈值的空挡
for lab, spans in sorted(pages.items()):
    if not spans: 
        bad.append((lab,"NO_ASR",0))
        continue
    gaps=[]
    non_en=[]
    for i,s in enumerate(spans):
        txt=s.get("text","")
        if non_ascii_ratio(txt)>0.2:
            non_en.append(i)
        if i>0:
            prev=spans[i-1]
            gap = s["t_start"] - prev["t_end"]
            if gap > gap_thr: gaps.append(round(gap,2))
    if gaps or non_en:
        bad.append((lab, f"gaps>{gap_thr}s:{gaps}" if gaps else "gaps:[]", len(non_en)))

print("\n[Summary of pages with potential issues]")
for lab, gaps, nbad in bad[:60]:
    print(f"- {lab:55s} | {gaps:20s} | non-ASCII segs: {nbad}")
print(f"\n[Done] flagged pages: {len(bad)}")
