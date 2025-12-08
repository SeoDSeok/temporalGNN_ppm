"""
BPI Challenge 2016 → Temporal GNN 입력 생성 (per-case by SessionID, with tqdm)
- 케이스 기준: SessionID
- 노드: 이벤트(event)
- 노드 피처: [one-hot(activity) | one-hot(feature2)]
  * feature2 자동 선택 우선순위: HandlingChannelID → VHOST → AgeCategory → Gender → URL_FILE → Office_U → Office_W
- 엣지: 같은 case 내 시간순 연속 이벤트 (src -> dst)
- 엣지 타임스탬프: 도착 노드의 UTC epoch seconds
"""
from __future__ import annotations
import os, json, zipfile, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ALLOWED = re.compile(r"[^0-9T:\-\.Z\+\s]")

# def normalize_and_parse_ts(x):
#     if x is None:
#         return pd.NaT
#     s = str(x)

#     # 1) 숨은/제어 문자 제거
#     s = (s.replace("\ufeff","").replace("\u200b","")
#            .replace("\r","").replace("\n","").strip())
#     # 2) 유사 기호 정규화 (수학용 '−' → '-')
#     s = s.replace("−", "-").replace("―", "-").replace("–", "-")
#     # 3) 헤더/잡값 거르기
#     if s.lower() in {"ocel:timestamp", "timestamp", "null", "nan", "", "-", "[]"}:
#         return pd.NaT
#     # 4) 대괄호/따옴표 제거
#     if s.startswith("[") and s.endswith("]"):
#         s = s[1:-1]
#     s = s.strip(' "\'')

#     # 5) 문자열 내 ISO 패턴만 추출
#     m = re.search(
#         r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?)"
#         r"(Z|[+\-]\d{2}:?\d{2})?",
#         s
#     )
#     if m:
#         iso, tz = m.group(1), m.group(2) or "+00:00"
#         # +0000 → +00:00 보정
#         tz = re.sub(r"([+\-]\d{2})(\d{2})$", r"\1:\2", tz)
#         s = iso.replace("T", " ") + tz
#         return pd.to_datetime(s, errors="coerce", utc=True)

#     # 6) 전부 숫자면 epoch(sec/ms)로 해석
#     if s.isdigit():
#         v = int(s)
#         return pd.to_datetime(v, unit=("ms" if v > 10**11 else "s"),
#                               errors="coerce", utc=True)

#     # 7) 마지막 시도(잡문자 제거 후 일반 파싱)
#     s = ALLOWED.sub("", s).replace("T", " ")
#     return pd.to_datetime(s, errors="coerce", utc=True)

def normalize_and_parse_ts(x):
    if x is None:
        return pd.NaT
    s = str(x).replace("\ufeff","").replace("\u200b","").replace("\r","").replace("\n","").strip()
    s = s.replace("−","-").replace("–","-").replace("―","-").strip(' "\'')
    if s.lower() in {"ocel:timestamp","timestamp","null","nan","","-","[]"}:
        return pd.NaT

    # 1) ISO 토큰만 추출
    m = re.search(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?)(Z|[+\-]\d{2}:?\d{2})?", s)
    if m:
        iso, tz = m.group(1), (m.group(2) or "+00:00")
        tz = re.sub(r"([+\-]\d{2})(\d{2})$", r"\1:\2", tz)
        return pd.to_datetime(iso.replace("T"," ") + tz, errors="coerce", utc=True)

    # 2) 전부 숫자면 epoch(sec/ms)
    if s.isdigit():
        v = int(s)
        unit = "ms" if v > 10**11 else "s"
        return pd.to_datetime(v, unit=unit, errors="coerce", utc=True)

    # 3) 나머지: 잡문자 제거 후 일반 파싱
    s = ALLOWED.sub("", s).replace("T"," ")
    return pd.to_datetime(s, errors="coerce", utc=True)

def parse_ts_column_robust(df: pd.DataFrame, col: str) -> pd.Series:
    """벡터화 fast-path + row-wise fallback로 tz-aware(UTC) 시리즈 생성"""
    s = df[col].astype(str)

    # fast-path: 정규식으로 ISO 토큰 벡터 추출
    m = s.str.extract(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?)(Z|[+\-]\d{2}:?\d{2})?")
    iso = m[0].fillna("")
    tz  = m[1].fillna("+00:00").str.replace(r"([+\-]\d{2})(\d{2})$", r"\1:\2", regex=True)
    ts  = pd.to_datetime(iso.str.replace("T"," ", regex=False) + tz, errors="coerce", utc=True)

    # 실패분만 fallback
    mask = ts.isna()
    if mask.any():
        ts.loc[mask] = s[mask].apply(normalize_and_parse_ts)

    return ts  # dtype: datetime64[ns, UTC]

def detect_timestamp_column(df: pd.DataFrame) -> str:
    candidates = ["ocel:timestamp", "time:timestamp", "timestamp", "Start Timestamp", "completeTime", "end_timestamp"]
    for c in candidates:
        if c in df.columns:
            # df[c] = df[c].apply(normalize_and_parse_ts)
            # df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
            df[c] = parse_ts_column_robust(df, c)
            return c
    raise ValueError("No timestamp-like column found. Checked: " + ", ".join(candidates))

def detect_columns_2016(df: pd.DataFrame) -> Dict[str, str]:
    m = {}
    # event id
    if "ocel:eid" in df.columns: m["event_id"] = "ocel:eid"
    elif "event_id" in df.columns: m["event_id"] = "event_id"
    else:
        df["__eid__"] = np.arange(len(df))
        m["event_id"] = "__eid__"
    # activity
    for a in ["ocel:activity", "Activity", "concept:name"]:
        if a in df.columns:
            m["activity"] = a
            break
    else:
        raise ValueError("No activity column found (expected one of ocel:activity/Activity/concept:name).")
    # case
    if "SessionID" not in df.columns:
        raise ValueError("Expected 'SessionID' column for case grouping.")
    m["case"] = "SessionID"
    # feature2 (categorical)
    for f in ["HandlingChannelID", "VHOST", "AgeCategory", "Gender", "URL_FILE", "Office_U", "Office_W"]:
        if f in df.columns:
            m["feature2"] = f
            break
    else:
        m["feature2"] = None
    return m

def build_vocab(series: pd.Series) -> Tuple[List[str], Dict[str, int]]:
    vals = sorted(series.astype(str).fillna("UNKNOWN").unique().tolist())
    idx = {v:i for i, v in enumerate(vals)}
    return vals, idx

def onehot(i: int, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    if 0 <= i < dim: v[i] = 1.0
    return v

def make_features(sub: pd.DataFrame, cols: Dict[str, str], vocabs: Dict[str, Tuple[List[str], Dict[str,int]]]) -> np.ndarray:
    a2i = vocabs["activity"][1]; a_dim = len(vocabs["activity"][0])
    f2name = cols.get("feature2")
    feats = []
    if f2name:
        f2i = vocabs["feature2"][1]; f2_dim = len(vocabs["feature2"][0])
    else:
        f2i, f2_dim = None, 0
    for _, row in sub.iterrows():
        avec = onehot(a2i[str(row[cols["activity"]])], a_dim)
        if f2name:
            f2vec = onehot(f2i[str(row[f2name])], f2_dim)
            feats.append(np.concatenate([avec, f2vec], axis=0))
        else:
            feats.append(avec)
    return np.stack(feats, axis=0)

def write_case(sub: pd.DataFrame, cols: Dict[str, str], ts_col: str, out_case: Path, vocabs: Dict[str, Tuple[List[str], Dict[str,int]]]) -> Tuple[int,int]:
    if not pd.api.types.is_datetime64tz_dtype(sub[ts_col]):
        # 혹시 naive라면 UTC 로컬라이즈
        if pd.api.types.is_datetime64_dtype(sub[ts_col]):
            sub[ts_col] = sub[ts_col].dt.tz_localize("UTC")
        else:
            sub[ts_col] = parse_ts_column_robust(sub, ts_col)
    sub = sub.copy().sort_values(by=ts_col).reset_index(drop=True)
    sub["__node_id__"] = np.arange(len(sub))
    # nodes.csv
    ts_iso   = sub[ts_col].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    ts_epoch = (sub[ts_col].view("int64") // 10**9).astype("int64")
    out_case.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "node_id": sub["__node_id__"],
        "event_eid": sub[cols["event_id"]].astype(str),
        "activity": sub[cols["activity"]].astype(str),
        "feature2": (sub[cols["feature2"]].astype(str) if cols.get("feature2") else ""),
        "timestamp_iso": ts_iso,
        "timestamp_epoch": ts_epoch,
        "case_id": str(sub[cols["case"]].iloc[0]),
    }).to_csv(out_case/"nodes.csv", index=False)
    # edges.csv
    if len(sub) >= 2:
        edges = pd.DataFrame({
            "src": sub["__node_id__"][:-1].to_numpy(),
            "dst": sub["__node_id__"][1:].to_numpy(),
            "timestamp_epoch": (sub[ts_col][1:].view("int64") // 10**9).astype("int64").to_numpy(),
        })
    else:
        edges = pd.DataFrame(columns=["src","dst","timestamp_epoch"])
    edges.to_csv(out_case/"edges.csv", index=False)
    # node_features.npy
    np.save(out_case/"node_features.npy", make_features(sub, cols, vocabs))
    return len(sub), len(edges)

def zip_dir(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for f in files:
                full = Path(root)/f
                zf.write(str(full), arcname=str(full.relative_to(src_dir)))

def build_tgn_per_case_2016_session(input_csv: str, out_root: str="./tgn_input_2016_cases_SessionID", zip_output: Optional[str]="./tgn_input_2016_cases_SessionID.zip"):
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv, low_memory=False)
    orig_cases = df["SessionID"].astype(str).nunique()
    ts_col = detect_timestamp_column(df)

    nan_ratio = df[ts_col].isna().mean()
    cases_after = df.loc[df[ts_col].notna(), "SessionID"].astype(str).nunique()
    print(f"[TS parse] NaT ratio={nan_ratio:.2%} | sessions orig={orig_cases:,} -> with valid ts={cases_after:,}")
    cols = detect_columns_2016(df)
    # 필수 필드 결측 제거
    df = df.dropna(subset=[cols["case"], ts_col, cols["activity"]]).copy()
    df[cols["activity"]] = df[cols["activity"]].astype(str).fillna("UNKNOWN")
    if cols.get("feature2"):
        df[cols["feature2"]] = df[cols["feature2"]].astype(str).fillna("UNKNOWN")
    # vocabs
    vocabs = {"activity": build_vocab(df[cols["activity"]])}
    if cols.get("feature2"):
        vocabs["feature2"] = build_vocab(df[cols["feature2"]])
    feature_dim = len(vocabs["activity"][0]) + (len(vocabs["feature2"][0]) if "feature2" in vocabs else 0)
    # metadata
    meta = {
        "columns": {"timestamp": ts_col, "event_id": cols["event_id"], "activity": cols["activity"], "case": cols["case"], "feature2": cols.get("feature2")},
        "vocabs": {"activity": vocabs["activity"][0], "feature2": (vocabs["feature2"][0] if "feature2" in vocabs else None)},
        "feature_layout": {"activity_onehot": len(vocabs["activity"][0]), "feature2_onehot": (len(vocabs["feature2"][0]) if "feature2" in vocabs else 0)},
        "feature_dim": feature_dim,
        "node_semantics": "Nodes are events. Node features = [one-hot(activity) | one-hot(feature2, auto-chosen)].",
        "edge_semantics": "Directed edges connect consecutive events within a SessionID case (sorted by timestamp). Edge timestamp equals destination timestamp (UTC epoch seconds).",
        "case_basis": "SessionID"
    }
    with open(out_root/"metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # per-case
    case_ids = df[cols["case"]].astype(str).unique().tolist()
    rows = []
    for cid in tqdm(case_ids, desc="Building 2016 cases (SessionID)"):
        sub = df[df[cols["case"]].astype(str) == cid]
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", cid)
        out_case = out_root / f"case_{safe}"
        n_nodes, n_edges = write_case(sub, cols, ts_col, out_case, vocabs)
        rows.append({"case_id": cid, "num_events": n_nodes, "num_edges": n_edges, "case_folder": str(out_case)})
    pd.DataFrame(rows).sort_values(by="num_events", ascending=False).to_csv(out_root/"summary.csv", index=False)
    if zip_output: zip_dir(out_root, Path(zip_output))
    print(f"[DONE] 2016 per-case (SessionID) | Cases: {len(case_ids)} | feature_dim={feature_dim}")
    print(f"- out_root: {out_root.resolve()}")
    if zip_output: print(f"- zip: {Path(zip_output).resolve()}")

if __name__ == "__main__":
    INPUT_CSV = "bpi_challenge_2016.csv"
    OUT_DIR   = "tgn_input_2016_cases_SessionID_2"
    ZIP_PATH  = None
    build_tgn_per_case_2016_session(INPUT_CSV, OUT_DIR, ZIP_PATH)
