# E:\Code\MaViLS\tools\split_by_wrap.py
import os, argparse
import pandas as pd

def load_xy(xlsx):
    df = pd.read_excel(xlsx)
    cols = {c.lower(): c for c in df.columns}
    if 'key' not in cols or 'value' not in cols:
        raise ValueError(f"{xlsx} 需要包含列 Key / Value")
    df = df[[cols['key'], cols['value']]].rename(columns={cols['key']:'Key', cols['value']:'Value'})
    # 确保按时间排序
    df = df.sort_values('Key', kind='mergesort').reset_index(drop=True)
    return df

def find_first_wrap(values, drop_tol=10):
    # 返回首次“明显回落”的下标 i（即 values[i] << values[i-1]）
    prev = None
    for i, v in enumerate(values):
        if prev is not None and (prev - v) >= drop_tol:
            return i
        prev = v
    return None  # 没检测到

def split_and_save(xlsx_l01, xlsx_l02, drop_tol=10):
    df1 = load_xy(xlsx_l01)
    df2 = load_xy(xlsx_l02)

    i1 = find_first_wrap(df1['Value'].tolist(), drop_tol=drop_tol)
    i2 = find_first_wrap(df2['Value'].tolist(), drop_tol=drop_tol)

    # L01：要的是“回绕之前”的那一段
    df1_keep = df1.iloc[:i1] if i1 is not None else df1.copy()
    # L02：要的是“回绕之后”的那一段
    df2_keep = df2.iloc[i2:] if i2 is not None else df2.copy()

    out1 = os.path.splitext(xlsx_l01)[0] + "_trimmed_v2.xlsx"
    out2 = os.path.splitext(xlsx_l02)[0] + "_trimmed_v2.xlsx"
    df1_keep.to_excel(out1, index=False)
    df2_keep.to_excel(out2, index=False)

    print(f"[L01] {xlsx_l01}")
    print(f"      rows={len(df1)}, wrap_idx={i1}, kept={len(df1_keep)} -> {out1}")
    print(f"[L02] {xlsx_l02}")
    print(f"      rows={len(df2)}, wrap_idx={i2}, kept={len(df2_keep)} -> {out2}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--l01", required=True, help="视频1对 Lecture01 的 xlsx")
    ap.add_argument("--l02", required=True, help="同一视频对 Lecture02 的 xlsx")
    ap.add_argument("--drop_tol", type=int, default=10, help="判定回落的阈值(默认10页)")
    args = ap.parse_args()
    split_and_save(args.l01, args.l02, drop_tol=args.drop_tol)
