import re, os
from pathlib import Path

SRC = r"E:\Code\pdf2latex_formal\merged.tex"
DST = r"E:\Code\pdf2latex_formal\merged_1-4.tex"

# 匹配形如：% Lecture 01_xxx_page_4 或 % Lecture_02_xxx_page_17
MARKER = re.compile(r"^%\s*Lecture[ _]0?(\d+)_.*?_page_(\d+)\s*$", re.MULTILINE)

def main():
    with open(SRC, "r", encoding="utf-8", errors="replace") as f:
        tex = f.read()

    # 找到所有页标记
    it = list(MARKER.finditer(tex))
    if not it:
        raise SystemExit("未在 merged.tex 里找到页标记（% Lecture ... _page_...）。")

    # 找到 \begin{document}（用于拼接导言区）
    m_begin = re.search(r"\\begin{document}", tex)
    if not m_begin:
        raise SystemExit("未找到 \\begin{document}")

    preamble = tex[:m_begin.end()]  # 包含 \begin{document}

    # 把每个标记到下一个标记（或文件末尾）的块切出来
    chunks = []
    for i, m in enumerate(it):
        start = m.start()
        end = it[i+1].start() if i+1 < len(it) else len(tex)
        chunks.append((m, tex[start:end]))

    # 只保留 Lecture <= 4 的块
    kept_blocks = []
    for m, block in chunks:
        lec = int(m.group(1))  # Lecture 序号
        if lec <= 4:
            kept_blocks.append(block)

    if not kept_blocks:
        raise SystemExit("Lecture 1-4 未匹配到任何页面块，检查标记格式是否一致。")

    # 生成新 tex：导言区 + 选中块 + \end{document}
    out = []
    out.append(preamble)
    out.append("\n")
    out.extend(kept_blocks)

    # 确保有 \end{document}
    if not re.search(r"\\end{document}", "".join(out)):
        out.append("\n\\end{document}\n")

    Path(DST).write_text("".join(out), encoding="utf-8")
    print(f"[OK] 写出前四讲：{DST}")

if __name__ == "__main__":
    main()
