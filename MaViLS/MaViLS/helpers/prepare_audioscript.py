import os
import re
import sys
import pandas as pd
sys.path.append('../')

def generate_output_dict_by_word(file):
    """generates dictionary with time of video recording as key and audio text said by the lecturer at this time as value.

    Args:
        file (str): file name/path to audioscript which was generated in such a format: [00:00:13.260 --> 00:00:14.200] class
                    The audioscript stores a word said in every row together with the exact time of the video recording (start and stop).
                    For generating these audioscripts please refer to e.g. to this repo: https://github.com/SYSTRAN/faster-whisper
        output_file (str, optional): output file path/name. Defaults to '../data/unlabeled_ground_truth/output_file.xlsx'.

    Returns:
        dict: dictionary with time of video recording as key and audio text as value by WORD
    """
    with open(file, encoding='utf-8') as f:
        input_text = f.read()

    # Regular expression pattern to match timestamps
    pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d{3})\s-->\s(\d{2}:\d{2}:\d{2}\.\d{3})\] (.+?)\n'

    # Find all matches in the input text
    matches = re.findall(pattern, input_text)

    # Initialize dictionary to store converted timestamps and corresponding text
    output_dict = {}

    # Convert timestamps to seconds and populate the dictionary
    for match in matches:
        start_time = match[0]
        end_time = match[1]
        text = match[2]
        end_seconds = int(start_time[:2]) * 3600  + int(end_time[3:5]) * 60 + float(end_time[6:])
        
        output_dict[end_seconds] = text  

    return output_dict


def _hms_to_sec(hms: str) -> float:
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)

def _parse_srt(text: str):
    """解析标准SRT：块状+逗号毫秒，返回[(t0,t1,txt), ...]"""
    items = []
    blocks = re.split(r"\r?\n\r?\n+", text.strip())
    for b in blocks:
        lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        if not lines:
            continue
        if re.fullmatch(r"\d+", lines[0]):  # 去掉序号行
            lines = lines[1:]
        if not lines:
            continue
        m = re.match(r"(\d{2}:\d{2}:\d{2})[,.](\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2})[,.](\d{3})", lines[0])
        if not m:
            continue
        sh, sms, eh, ems = m.group(1), m.group(2), m.group(3), m.group(4)
        t0 = _hms_to_sec(sh) + int(sms) / 1000.0
        t1 = _hms_to_sec(eh) + int(ems) / 1000.0
        txt = " ".join(lines[1:]).strip()
        if txt:
            items.append((t0, t1, txt))
    return items

def _parse_bracket_lines(text: str):
    """解析方括号行：[hh:mm:ss.mmm --> hh:mm:ss.mmm] sentence"""
    items = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        m = re.match(
            r"\[(\d{2}:\d{2}:\d{2})[.,](\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2})[.,](\d{3})\]\s*(.+)$",
            ln,
        )
        if not m:
            continue
        sh, sms, eh, ems, txt = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5).strip()
        t0 = _hms_to_sec(sh) + int(sms) / 1000.0
        t1 = _hms_to_sec(eh) + int(ems) / 1000.0
        if txt:
            items.append((t0, t1, txt))
    return items

def generate_output_dict_by_sentence(file):
    """
    返回 { mid_seconds(float): sentence(str), ... }
    兼容两种输入：标准SRT（块状）或方括号单行格式。
    读不到则返回空 dict（主流程会继续）。
    """
    if not file or not os.path.exists(file) or os.path.getsize(file) == 0:
        return {}

    with open(file, "r", encoding="utf-8-sig", errors="ignore") as f:
        text = f.read()

    items = _parse_srt(text)
    if not items:
        items = _parse_bracket_lines(text)

    out = {}
    for t0, t1, txt in items:
        mid = (t0 + t1) / 2.0
        out[mid] = txt
    return out
