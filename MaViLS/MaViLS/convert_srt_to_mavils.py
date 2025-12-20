# convert_srt_to_mavils.py
import re, sys, pathlib

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
out = []

with src.open('r', encoding='utf-8', errors='ignore') as f:
    blocks = re.split(r'\r?\n\r?\n+', f.read().strip())

for b in blocks:
    lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
    if not lines: 
        continue
    # 可能第1行是序号
    if re.fullmatch(r'\d+', lines[0]): 
        lines = lines[1:]
    if not lines: 
        continue
    # 时间行：00:00:13,260 --> 00:00:14,200
    m = re.match(r'(\d{2}:\d{2}:\d{2})[,.](\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2})[,.](\d{3})', lines[0])
    if not m:
        # 有些工具用 00:00:13.260 也行，尽量兼容
        m = re.match(r'(\d{2}:\d{2}:\d{2})[.](\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2})[.](\d{3})', lines[0])
    if not m:
        continue
    s_hms, s_ms, e_hms, e_ms = m.groups()
    text = " ".join(lines[1:])  # 同一字幕块多行并成一行
    text = re.sub(r'\s+', ' ', text).strip()
    # 生成 MaViLS 需要的单行：方括号 + 小数点毫秒
    out.append(f'[{s_hms}.{s_ms} --> {e_hms}.{e_ms}] {text}')

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open('w', encoding='utf-8') as f:
    f.write("\n".join(out) + "\n")
print(f"Converted {len(out)} lines -> {dst}")
