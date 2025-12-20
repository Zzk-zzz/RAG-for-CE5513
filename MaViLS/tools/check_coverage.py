# E:\Code\MaViLS\tools\check_coverage.py
import re
from pathlib import Path

TEX = r"E:\Code\outputs\merged_with_asr_mavils.tex"

marker = re.compile(r"^%\s*(.*?)_page_(\d+)\s*$", re.MULTILINE)
has_asr = re.compile(r"\\paragraph\*\{ASR\}", re.MULTILINE)

text = Path(TEX).read_text(encoding="utf-8", errors="replace")
ms = list(marker.finditer(text))

pages = []
for i, m in enumerate(ms):
    start = m.end()
    end = ms[i+1].start() if i+1 < len(ms) else len(text)
    label = f"{m.group(1)}_page_{int(m.group(2))}"
    chunk = text[start:end]
    pages.append((label, bool(has_asr.search(chunk))))

total = len(pages)
with_asr = sum(1 for _, ok in pages if ok)
no_asr = [(lab) for lab, ok in pages if not ok]

print(f"[STAT] pages={total}  pages_with_ASR={with_asr}  pages_without_ASR={total-with_asr}")
if no_asr:
    print("[NO ASR PAGES] (first 30)")
    for lab in no_asr[:30]:
        print(" -", lab)
