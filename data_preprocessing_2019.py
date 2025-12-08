"""
BPI Challenge 2019 → Temporal GNN 입력 생성 스크립트 (per-case by cID, with tqdm)
- 그래프 단위: cID 별 폴더
- 노드: 이벤트(event)
- 노드 피처: [one-hot(activity) | one-hot(feature2)], feature2는 기본 cDocType (없으면 자동 폴백)
- 엣지: 같은 case 내 시간 순서로 연속 이벤트 간 (src -> dst)
- 엣지 타임스탬프: 도착 노드의 UTC epoch seconds
"""
from __future__ import annotations
import os, json, zipfile, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def detect_ts(df: pd.DataFrame) -> str:
    candidates = ["ocel:timestamp", "time:timestamp", "timestamp", "Start Timestamp", "completeTime", "end_timestamp"]
    for c in candidates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
            return c
    raise ValueError(f"No timestamp-like column found among: {candidates}")

def detect_cols_2019_cid(df: pd.DataFrame) -> Dict[str, str]:
    """
    2019 샘플 구성 + cID 기준 case 분할
    필수: event_id, activity, case(최우선 cID)
    선택: feature2 (cDocType 우선, 없으면 cItemCat→cItemType→cGRbasedInvVerif→cGR→cCompany)
    """
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
    # case id priority: cID → (fallbacks)
    for c in ["cID", "case", "case:concept:name", "cPOID", "PO", "po"]:
        if c in df.columns:
            m["case"] = c
            break
    else:
        raise ValueError("No case column found. Expected one of cID/case/case:concept:name/cPOID/PO/po.")
    # second categorical feature
    for f in ["cDocType", "cItemCat", "cItemType", "cGRbasedInvVerif", "cGR", "cCompany"]:
        if f in df.columns:
            m["feature2"] = f
            break
    m["lifecycle"] = None
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
    act2idx = vocabs["activity"][1]; a_dim = len(vocabs["activity"][0])
    f2name = cols.get("feature2")
    if f2name:
        f2_2idx = vocabs["feature2"][1]; f2_dim = len(vocabs["feature2"][0])
    else:
        f2_2idx = None; f2_dim = 0
    feats = []
    for _, row in sub.iterrows():
        a_vec = onehot(act2idx[str(row[cols["activity"]])], a_dim)
        if f2name:
            f2_vec = onehot(f2_2idx[str(row[f2name])], f2_dim)
            feats.append(np.concatenate([a_vec, f2_vec], axis=0))
        else:
            feats.append(a_vec)
    return np.stack(feats, axis=0)

def write_case(case_df: pd.DataFrame, cols: Dict[str, str], ts_col: str, out_case: Path, vocabs: Dict[str, Tuple[List[str], Dict[str,int]]]) -> Tuple[int,int]:
    sub = case_df.copy().sort_values(by=ts_col).reset_index(drop=True)  # typo fix if needed: drop=True
    sub["__node_id__"] = np.arange(len(sub))
    ts_iso   = sub[ts_col].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    ts_epoch = (sub[ts_col].view("int64") // 10**9).astype("int64")
    # nodes.csv
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
    # edges.csv (연속 이벤트 간)
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

def build_tgn_per_case_2019_cid(input_csv: str, out_root: str="./tgn_input_2019_cases_cID", zip_output: Optional[str]="./tgn_input_2019_cases_cID.zip"):
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv, low_memory=False)
    ts_col = detect_ts(df)
    cols = detect_cols_2019_cid(df)
    # 필수 필드 결측 제거
    df = df.dropna(subset=[cols["case"], ts_col, cols["activity"]]).copy()
    df[cols["activity"]] = df[cols["activity"]].astype(str).fillna("UNKNOWN")
    if cols.get("feature2"):
        df[cols["feature2"]] = df[cols["feature2"]].astype(str).fillna("UNKNOWN")
    # vocab
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
        "node_semantics": "Nodes are events. Node features = [one-hot(activity) | one-hot(feature2=cDocType by default)].",
        "edge_semantics": "Directed edges connect consecutive events in the same case (sorted by timestamp). Edge timestamp equals destination timestamp (UTC epoch seconds)."
    }
    with open(out_root/"metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # per-case with tqdm
    case_ids = df[cols["case"]].astype(str).unique().tolist()
    rows = []
    for cid in tqdm(case_ids, desc="Building cases (cID)"):
        sub = df[df[cols["case"]].astype(str) == cid]
        safe_cid = re.sub(r"[^A-Za-z0-9_.-]", "_", cid)
        out_case = out_root / f"case_{safe_cid}"
        n_nodes, n_edges = write_case(sub, cols, ts_col, out_case, vocabs)
        rows.append({"case_id": cid, "num_events": n_nodes, "num_edges": n_edges, "case_folder": str(out_case)})
    pd.DataFrame(rows).sort_values(by="num_events", ascending=False).to_csv(out_root/"summary.csv", index=False)
    if zip_output: zip_dir(out_root, Path(zip_output))
    print(f"[DONE] 2019 per-case (cID) | Cases: {len(case_ids)} | feature_dim={feature_dim}")
    print(f"- out_root: {out_root.resolve()}")
    if zip_output: print(f"- zip: {Path(zip_output).resolve()}")

if __name__ == "__main__":
    INPUT_CSV = "bpi_challenge_2019.csv"      # 입력 CSV 경로
    OUT_DIR   = "tgn_input_2019_cases_cID"    # 출력 폴더
    ZIP_PATH  = None                          # ZIP 저장 원하면 경로 지정
    build_tgn_per_case_2019_cid(INPUT_CSV, OUT_DIR, ZIP_PATH)
