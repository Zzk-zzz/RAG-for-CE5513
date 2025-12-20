# -*- coding: utf-8 -*-
"""
MaViLS matching - Zikun fixed edition (multi-PDF enabled)
关键修复/增强：
1) 新增 --file_paths（支持多份 PDF），内存里顺序拼接成一个“长文档”
2) SRT 时间 -> 抽帧间隔过滤（MIN_GAP）防止内存爆、速度慢
3) 视频帧原地 resize + 可选左侧裁剪（只保留 PPT 区域）
4) PDF 转图只在内存，不写盘
5) SentenceTransformer 输出统一转 numpy
6) jump_penalty 改为 float
7) OCR 语言可配，默认 'eng'
8) 结束时清理中间产物（ocr/image/audio 三路的 xlsx，可选）
"""

import os
import gc
import sys
import argparse
from datetime import datetime
from pathlib import Path
import glob, shutil

import numpy as np
import pandas as pd
import cv2
import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, SwiftFormerModel

# 本仓库 helper
sys.path.append('../')
from helpers.prepare_audioscript import generate_output_dict_by_sentence
from helpers.utils import (
    calculate_dp_with_jumps,
    compute_similarity_matrix,
    create_video_frames,
    extract_features_from_images,
    calculate_sift_normalized_similarity,
    gradient_descent_with_adam
)

# Tesseract 路径（保持你本地配置）
from local_settings import path_to_tesseract
import pytesseract
pytesseract.pytesseract.tesseract_cmd = path_to_tesseract

# =========================
# 可调参数
# =========================
MIN_GAP_SEC = 0.9
FRAME_W, FRAME_H = 960, 540
CROP_LEFT_FRACTION = 0.66
TESS_LANGS = "eng"

# =========================
# 工具
# =========================
def _is_float_like(x):
    try:
        float(x); return True
    except Exception:
        return False

def _ensure_parent_dir(path_str: str):
    p = Path(path_str)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)

def _log_time(tag: str, t0: datetime, append_file: str = "time.txt"):
    with open(append_file, 'a', encoding='utf-8') as f:
        f.write(f"{tag}: {datetime.now() - t0}\n")

def _safe_rm(p):
    try:
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.isfile(p):
            os.remove(p)
    except Exception:
        pass

def _cleanup_outputs(file_name: str):
    out_dir = os.path.dirname(file_name) or "."
    trash_globs = [
        f"{file_name}_ocr_*.xlsx",
        f"{file_name}_image_matching_*.xlsx",
        f"{file_name}_audiomatching_*.xlsx",
        os.path.join(out_dir, "time.txt"),
    ]
    for pat in trash_globs:
        for f in glob.glob(pat):
            _safe_rm(f)

    # 清 MaViLS 仓库根部的临时目录（如果曾被原版脚本创建）
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for d in [os.path.join(repo_root, "pdf_images"),
              os.path.join(repo_root, "frame_images")]:
        _safe_rm(d)

    print("[CLEAN] Removed temp xlsx (ocr/image/audio) and temp image dirs; kept fused xlsx + SRT.")

def _parse_multi_paths(s: str):
    """
    解析 --file_paths；支持用 ; 或 , 分隔，自动去除空白。
    """
    parts = []
    for tok in (s or "").replace(";", ",").split(","):
        tok = tok.strip().strip('"').strip("'")
        if tok:
            parts.append(tok)
    return parts

def _load_pdfs_to_images(pdf_paths: list[str]):
    """
    依次读取多份 PDF，转为 BGR 图像列表 + PIL 图像列表，并返回总页数。
    """
    text_pdf = []
    pdf_images_cv2 = []
    pil_images = []

    for one_pdf in pdf_paths:
        if not os.path.isfile(one_pdf):
            raise FileNotFoundError(f"PDF not found: {one_pdf}")
        pdf = fitz.open(one_pdf)
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            text_pdf.append(page.get_text())

            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = img[:, :, :3]
            if img.ndim == 2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            pdf_images_cv2.append(img_bgr)
            pil_images.append(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
        pdf.close()

    return text_pdf, pdf_images_cv2, pil_images

# =========================
# 主流程
# =========================
def main(
    video_path: str,
    file_name: str,
    file_path: str = None,
    audio_script: str = None,
    autoimage_name: str = 'MBZUAI/swiftformer-xs',
    sentence_model_name: str = 'sentence-transformers/distiluse-base-multilingual-cased',
    jump_penalty: float = 0.1,
    merge_method: str = 'max',
    sift: bool = False,
    file_paths: list[str] = None,         # 新增：多 PDF
    keep_intermediate: bool = False       # 新增：是否保留三路单表
):
    """
    运行 MaViLS 三路匹配。支持：
      - 单 PDF：file_path
      - 多 PDF：file_paths（顺序拼接）
    """

    # ---- 前置检查 ----
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not audio_script or not os.path.isfile(audio_script):
        raise FileNotFoundError(f"Audio script not found: {audio_script}")

    # 解析 PDF 列表（多优先）
    pdf_list = []
    if file_paths:
        pdf_list = file_paths
    elif file_path:
        pdf_list = [file_path]
    else:
        raise ValueError("Please provide --file_path or --file_paths (multiple).")

    for p in pdf_list:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"PDF not found: {p}")

    _ensure_parent_dir(file_name + "_dummy.touch")

    jump_penalty_string = str(jump_penalty).replace('.', 'comma')

    with open('time.txt', 'a', encoding='utf-8') as f:
        f.write("== Time for alignment algorithms ==\n")
    t0 = datetime.now()

    # =======================
    # 1) 解析字幕 -> 时间点（做抽帧用）
    # =======================
    output_dict = generate_output_dict_by_sentence(audio_script)  # {time_str/float: sentence}
    interval_raw = sorted(float(t) for t in output_dict.keys() if _is_float_like(t))
    if not interval_raw:
        raise ValueError("No valid timestamps parsed from audio_script.")

    interval_list = []
    prev = -1e9
    for t in interval_raw:
        if t - prev >= MIN_GAP_SEC:
            interval_list.append(t); prev = t
    print(f"[INFO] intervals after gap filter: {len(interval_list)}  (from {len(interval_raw)})")

    # =======================
    # 2) 抽帧
    # =======================
    frames = create_video_frames(video_path, interval_list)
    resized_frames = frames
    for i in tqdm(range(len(resized_frames)), desc='video frames are resized'):
        f = cv2.resize(resized_frames[i], (FRAME_W, FRAME_H))
        if 0.1 <= CROP_LEFT_FRACTION <= 1.0:
            cut = int(FRAME_W * CROP_LEFT_FRACTION)
            f = f[:, :cut]
        resized_frames[i] = f
    gc.collect()

    # =======================
    # 3) 多 PDF -> 图像 + 文本（内存）
    # =======================
    text_pdf, pdf_images_cv2, pil_images = _load_pdfs_to_images(pdf_list)

    # =======================
    # 4) 文本/视觉特征
    # =======================
    sentence_model = SentenceTransformer(sentence_model_name)

    # 只保留过滤后的字幕句子
    sentences_audio = [output_dict.get(str(t), output_dict.get(t, "")) for t in interval_list]
    audio_features = sentence_model.encode(
        sentences_audio, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
    ).detach().cpu().numpy()

    ocr_config = "--oem 1 --psm 6"
    image_texts = [
        pytesseract.image_to_string(im, lang=TESS_LANGS, config=ocr_config)
        for im in tqdm(pil_images, desc='text is extracted from slide images')
    ]
    text_features = sentence_model.encode(
        image_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
    ).detach().cpu().numpy()

    # 音频句子 vs 幻灯片 OCR
    similarity_matrix_audio = compute_similarity_matrix(audio_features, text_features)
    optimal_path_audio, _ = calculate_dp_with_jumps(similarity_matrix_audio, jump_penalty)
    _log_time("Time for audio algorithm", t0)

    # 视觉模型
    image_processor = AutoImageProcessor.from_pretrained(autoimage_name)
    image_model = SwiftFormerModel.from_pretrained(autoimage_name)

    pdf_proc = [image_processor(im, return_tensors="pt") for im in tqdm(pdf_images_cv2, desc='pdf images are processed')]
    vid_proc = [image_processor(im, return_tensors="pt") for im in tqdm(resized_frames, desc='video frames are processed')]

    features_pdf = np.array(extract_features_from_images(pdf_proc, image_model))
    features_frames = np.array(extract_features_from_images(vid_proc, image_model))

    similarity_matrix_image = compute_similarity_matrix(features_frames, features_pdf)
    optimal_path_image, _ = calculate_dp_with_jumps(similarity_matrix_image, jump_penalty)
    _log_time("Time for image algorithm", t0)

    # OCR 视频帧
    frame_texts = [
        pytesseract.image_to_string(
            Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)), lang=TESS_LANGS, config=ocr_config
        )
        for frm in tqdm(resized_frames, desc='text is extracted from video frames')
    ]
    frame_features = sentence_model.encode(
        frame_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
    ).detach().cpu().numpy()

    similarity_matrix_ocr = compute_similarity_matrix(frame_features, text_features)
    optimal_path_ocr, _ = calculate_dp_with_jumps(similarity_matrix_ocr, jump_penalty)
    _log_time("Time for ocr algorithm", t0)

    # =======================
    # 5) 写三路结果
    # =======================
    def _path_to_df(opt_path):
        return { interval_list[i0]: (j0 + 1) for (i0, j0) in opt_path }

    df_ocr   = pd.DataFrame(list(_path_to_df(optimal_path_ocr).items()),   columns=['Key', 'Value'])
    df_audio = pd.DataFrame(list(_path_to_df(optimal_path_audio).items()), columns=['Key', 'Value'])
    df_img   = pd.DataFrame(list(_path_to_df(optimal_path_image).items()), columns=['Key', 'Value'])

    df_ocr.to_excel(f'{file_name}_ocr_{jump_penalty_string}.xlsx', index=False, engine='openpyxl')
    df_audio.to_excel(f'{file_name}_audiomatching_{jump_penalty_string}.xlsx', index=False, engine='openpyxl')
    df_img.to_excel(f'{file_name}_image_matching_{jump_penalty_string}.xlsx', index=False, engine='openpyxl')

    # =======================
    # 6) 融合
    # =======================
    if merge_method == 'mean':
        sim_merged = np.mean((similarity_matrix_ocr, similarity_matrix_audio, similarity_matrix_image), axis=0)
        opt_path, _ = calculate_dp_with_jumps(sim_merged, jump_penalty)
        pd.DataFrame(list(_path_to_df(opt_path).items()), columns=['Key','Value'])\
          .to_excel(f'{file_name}_mean_matching_all_{jump_penalty_string}.xlsx', index=False, engine='openpyxl')

    elif merge_method == 'max':
        sim_merged = np.max((similarity_matrix_ocr, similarity_matrix_audio, similarity_matrix_image), axis=0)
        opt_path, _ = calculate_dp_with_jumps(sim_merged, jump_penalty)
        pd.DataFrame(list(_path_to_df(opt_path).items()), columns=['Key','Value'])\
          .to_excel(f'{file_name}_max_matching_all_{jump_penalty_string}.xlsx', index=False, engine='openpyxl')

    elif merge_method == 'weighted_sum':
        matrices = [similarity_matrix_ocr, similarity_matrix_audio, similarity_matrix_image]
        weights = gradient_descent_with_adam(matrices, jump_penalty, num_iterations=50)
        print("Optimal Weights:", weights)
        sim_merged = weights[0]*similarity_matrix_ocr + weights[1]*similarity_matrix_audio + weights[2]*similarity_matrix_image
        opt_path, _ = calculate_dp_with_jumps(sim_merged, jump_penalty)
        pd.DataFrame(list(_path_to_df(opt_path).items()), columns=['Key','Value'])\
          .to_excel(f'{file_name}_weighted_sum_matching_all_{jump_penalty_string}_50iterations_with_adam.xlsx', index=False, engine='openpyxl')

    elif merge_method == 'all':
        # mean
        sim_merged = np.mean((similarity_matrix_ocr, similarity_matrix_audio, similarity_matrix_image), axis=0)
        opt_path, _ = calculate_dp_with_jumps(sim_merged, jump_penalty)
        pd.DataFrame(list(_path_to_df(opt_path).items()), columns=['Key','Value'])\
          .to_excel(f'{file_name}_mean_matching_all_{jump_penalty_string}.xlsx', index=False, engine='openpyxl')
        # max
        sim_merged = np.max((similarity_matrix_ocr, similarity_matrix_audio, similarity_matrix_image), axis=0)
        opt_path, _ = calculate_dp_with_jumps(sim_merged, jump_penalty)
        pd.DataFrame(list(_path_to_df(opt_path).items()), columns=['Key','Value'])\
          .to_excel(f'{file_name}_max_matching_all_{jump_penalty_string}.xlsx', index=False, engine='openpyxl')
        # weighted
        matrices = [similarity_matrix_ocr, similarity_matrix_audio, similarity_matrix_image]
        weights = gradient_descent_with_adam(matrices, jump_penalty)
        print("Optimal Weights:", weights)
        sim_merged = weights[0]*similarity_matrix_ocr + weights[1]*similarity_matrix_audio + weights[2]*similarity_matrix_image
        opt_path, _ = calculate_dp_with_jumps(sim_merged, jump_penalty)
        pd.DataFrame(list(_path_to_df(opt_path).items()), columns=['Key','Value'])\
          .to_excel(f'{file_name}_weighted_sum_matching_all_{jump_penalty_string}.xlsx', index=False, engine='openpyxl')

    _log_time("== Total time ==", t0)

    if not keep_intermediate:
        try:
            _cleanup_outputs(file_name)
        except Exception as e:
            print(f"[CLEAN][WARN] {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Match video frames to lecture slides (multi-PDF enabled)")

    parser.add_argument('--sentence_model', default='sentence-transformers/distiluse-base-multilingual-cased', type=str)
    parser.add_argument('--jump_penalty', default=0.1, type=float)
    parser.add_argument('--autoimage_name', default="MBZUAI/swiftformer-xs", type=str)
    parser.add_argument('--merge_method', default='max', type=str, choices=['mean','max','weighted_sum','all'])
    parser.add_argument('--sift', action='store_true')

    parser.add_argument('--audio_script', type=str, required=True, help='Path to subtitle file')
    parser.add_argument('--file_path', type=str, help='Single PDF path (for backward compatibility)')
    parser.add_argument('--file_paths', type=str, help='Multiple PDF paths separated by ; or ,')
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--file_name', type=str, required=True, help='Output file prefix (without extension)')
    parser.add_argument('--keep_intermediate', action='store_true', help='Keep ocr/image/audio single-lane xlsx')

    args = parser.parse_args()

    pdf_list = _parse_multi_paths(args.file_paths) if args.file_paths else ( [args.file_path] if args.file_path else [] )

    main(
        video_path=args.video_path,
        file_name=args.file_name,
        file_path=args.file_path,
        audio_script=args.audio_script,
        sift=args.sift,
        merge_method=args.merge_method,
        autoimage_name=args.autoimage_name,
        jump_penalty=args.jump_penalty,
        sentence_model_name=args.sentence_model,
        file_paths=pdf_list,
        keep_intermediate=args.keep_intermediate
    )
