# E:\Code\MaViLS\make_srt.py
import os, math, subprocess
from pathlib import Path

# =========================
# User paths
# =========================
OUT_DIR = r"E:\Code\outputs"

VID1 = r"E:\Code\Wednesday_ 15 January 2025 at 5_54_50 pm_default_8ee4e38d.mp4"
VID2 = r"E:\Code\Wednesday_ 22 January 2025 at 5_57_29 pm_default_ea687d7b.mp4"

OUT_SRT_1 = os.path.join(OUT_DIR, "lecture01.srt")
OUT_SRT_2 = os.path.join(OUT_DIR, "lecture02.srt")

# ffmpeg：你采纳的固定目录（脚本注入 PATH）
FFMPEG_DIR = r"E:\Code\MaViLS\tools\ffmpeg\bin"
FFMPEG_EXE = os.path.join(FFMPEG_DIR, "ffmpeg.exe")

# 缓存尽量放 E 盘（避免 C 盘爆）
os.environ.setdefault("HF_HOME", r"E:\AIcache\hf")
os.environ.setdefault("TRANSFORMERS_CACHE", r"E:\AIcache\hf\transformers")
os.environ.setdefault("PIP_CACHE_DIR", r"E:\AIcache\pip")
WHISPER_MODEL_DIR = r"E:\AIcache\whisper_models"

# =========================
# ASR params (safe defaults)
# =========================
ASR_MODEL = "small.en"
ASR_DEVICE = "cpu"          # 想试 GPU 就改成 "cuda"（但你之前 CUDA/cuDNN 折腾很多，先 CPU 最稳）
ASR_COMPUTE_TYPE = "int8"   # cpu 建议 int8；cuda 建议 float16
ASR_BEAM_SIZE = 3           # 1=贪心更快；3 更稳一点
ASR_LANGUAGE = "en"

ASR_INITIAL_PROMPT = (
    "CE5513 plastic analysis of structures, plastic hinge, collapse mechanism, "
    "Ronan Point, Hyatt Regency walkway, yield, moment capacity, kinematic approach, "
    "limit analysis, interaction diagram, incremental elastoplastic analysis, linear programming, "
    "finite element analysis, beam section behaviour"
)

# =========================
# SRT utils
# =========================
def sec_to_ts(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    ms = int(round((sec - math.floor(sec)) * 1000))
    s  = int(math.floor(sec))
    h  = s // 3600
    m  = (s % 3600) // 60
    s  = s % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(items, out_path: str):
    items = sorted(items, key=lambda x: (x["t_start"], x["t_end"]))
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(items, 1):
            f.write(f"{i}\n")
            f.write(f"{sec_to_ts(seg['t_start'])} --> {sec_to_ts(seg['t_end'])}\n")
            f.write((seg.get("text") or "").strip().replace("\r", "") + "\n\n")
    print(f"[OK] wrote {out_path} ({len(items)} segments)")

# =========================
# ffmpeg + whisper
# =========================
def ensure_ffmpeg():
    if os.path.isdir(FFMPEG_DIR):
        os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")
    if os.path.exists(FFMPEG_EXE):
        return FFMPEG_EXE
    raise FileNotFoundError(f"ffmpeg.exe not found: {FFMPEG_EXE}")

def extract_wav(video_path: str, wav_path: str):
    ffmpeg = ensure_ffmpeg()
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)

    # 标准化：16kHz mono PCM，ASR 最稳
    cmd = [
        ffmpeg, "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        wav_path
    ]
    print("[FFMPEG]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def transcribe_wav_to_segments(wav_path: str):
    from faster_whisper import WhisperModel

    compute = ASR_COMPUTE_TYPE
    if ASR_DEVICE.lower() == "cuda":
        compute = "float16"

    print(f"[ASR] model={ASR_MODEL} device={ASR_DEVICE} compute={compute} beam={ASR_BEAM_SIZE}")
    model = WhisperModel(
        ASR_MODEL,
        device=ASR_DEVICE,
        compute_type=compute,
        download_root=WHISPER_MODEL_DIR
    )

    segments, info = model.transcribe(
        wav_path,
        beam_size=ASR_BEAM_SIZE,
        vad_filter=True,
        language=ASR_LANGUAGE,
        task="transcribe",
        condition_on_previous_text=True,
        initial_prompt=ASR_INITIAL_PROMPT
    )

    out = []
    for seg in segments:
        txt = (seg.text or "").strip()
        if not txt:
            continue
        out.append({"t_start": float(seg.start), "t_end": float(seg.end), "text": txt})
    return out

def video_to_srt(video_path: str, out_srt: str):
    audio_cache = os.path.join(OUT_DIR, "_audio_cache")
    wav_path = os.path.join(audio_cache, Path(video_path).stem + ".wav")

    extract_wav(video_path, wav_path)
    segs = transcribe_wav_to_segments(wav_path)
    write_srt(segs, out_srt)

# =========================
# main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    video_to_srt(VID1, OUT_SRT_1)
    video_to_srt(VID2, OUT_SRT_2)
    print("[DONE] lecture01.srt + lecture02.srt generated.")

if __name__ == "__main__":
    main()
