"""
BPI Challenge 2014 → Temporal GNN 입력 생성 (per-case by ocel:type:Interaction, with tqdm)
- 케이스 기준: ocel:type:Interaction (리스트형 → explode 후 각 Interaction ID별 폴더)
- 노드: 이벤트(event)
- 노드 피처: [one-hot(activity) | one-hot(Status)]  # Status가 없으면 Priority → Impact 순으로 폴백
- 엣지: 같은 case 내 시간순 연속 이벤트 (src -> dst)
- 엣지 타임스탬프: 도착 노드의 UTC epoch seconds
"""
from __future__ import annotations
import os, json, zipfile, re, ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def detect_timestamp_column(df: pd.DataFrame) -> str:
    candidates = ["ocel:timestamp", "time:timestamp", "timestamp", "Start Timestamp", "completeTime", "end_timestamp"]
    for c in candidates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
            return c
    raise ValueError("No timestamp-like column found. Checked: " + ", ".join(candidates))

def parse_list_column(series: pd.Series) -> pd.Series:
    def to_list(x):
        if isinstance(x, str):
            try:
                v = ast.literal_eval(x)
                return v if isinstance(v, list) else ([] if pd.isna(x) else [str(v)])
            except Exception:
                return []
        return []
    return series.apply(to_list)

def detect_columns_2014(df: pd.DataFrame) -> Dict[str, str]:
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
    # case list column (Interaction)
    if "ocel:type:Interaction" not in df.columns:
        raise ValueError("Expected 'ocel:type:Interaction' column for case grouping.")
    m["case_list"] = "ocel:type:Interaction"
    # second feature candidates
    for f in ["Status", "Priority", "Impact"]:
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
        "case_id": str(sub["__case_id__"].iloc[0]),
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
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf :
        for root, _, files in os.walk(src_dir):
            for f in files:
                full = Path(root)/f
                zf.write(str(full), arcname=str(full.relative_to(src_dir)))

def build_tgn_per_case_2014_interaction(input_csv: str, out_root: str="./tgn_input_2014_cases_Interaction", zip_output: Optional[str]="./tgn_input_2014_cases_Interaction.zip"):
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv, low_memory=False)
    ts_col = detect_timestamp_column(df)
    cols = detect_columns_2014(df)
    # Parse interaction list and explode
    inter_lists = parse_list_column(df[cols["case_list"]])
    df = df.assign(__case_list__=inter_lists)
    df = df[df["__case_list__"].map(len) > 0].copy()
    df = df.explode("__case_list__").rename(columns={"__case_list__": "__case_id__"})
    # Keep only rows with valid event info
    df = df.dropna(subset=[cols["activity"], ts_col]).copy()
    # Cast feature2 if exists
    if cols.get("feature2"):
        df[cols["feature2"]] = df[cols["feature2"]].astype(str).fillna("UNKNOWN")
    df[cols["activity"]] = df[cols["activity"]].astype(str).fillna("UNKNOWN")
    # Build vocabs
    vocabs = {"activity": build_vocab(df[cols["activity"]])}
    if cols.get("feature2"):
        vocabs["feature2"] = build_vocab(df[cols["feature2"]])
    feature_dim = len(vocabs["activity"][0]) + (len(vocabs["feature2"][0]) if "feature2" in vocabs else 0)
    # metadata
    meta = {
        "columns": {"timestamp": ts_col, "event_id": cols["event_id"], "activity": cols["activity"], "case_list": cols["case_list"], "case_id_exploded": "__case_id__", "feature2": cols.get("feature2")},
        "vocabs": {"activity": vocabs["activity"][0], "feature2": (vocabs["feature2"][0] if "feature2" in vocabs else None)},
        "feature_layout": {"activity_onehot": len(vocabs["activity"][0]), "feature2_onehot": (len(vocabs["feature2"][0]) if "feature2" in vocabs else 0)},
        "feature_dim": feature_dim,
        "node_semantics": "Nodes are events. Node features = [one-hot(activity) | one-hot(Status/else fallback)].",
        "edge_semantics": "Directed edges connect consecutive events within an Interaction case (sorted by timestamp). Edge timestamp equals destination timestamp (UTC epoch seconds).",
        "case_basis": "ocel:type:Interaction (list-exploded)"
    }
    with open(out_root/"metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # per-case
    case_ids = df["__case_id__"].astype(str).unique().tolist()
    rows = []
    for cid in tqdm(case_ids, desc="Building 2014 cases (Interaction)"):
        sub = df[df["__case_id__"].astype(str) == cid]
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", cid)
        out_case = out_root / f"case_{safe}"
        n_nodes, n_edges = write_case(sub, cols, ts_col, out_case, vocabs)
        rows.append({"case_id": cid, "num_events": n_nodes, "num_edges": n_edges, "case_folder": str(out_case)})
    pd.DataFrame(rows).sort_values(by="num_events", ascending=False).to_csv(out_root/"summary.csv", index=False)
    if zip_output: zip_dir(out_root, Path(zip_output))
    print(f"[DONE] 2014 per-case (Interaction) | Cases: {len(case_ids)} | feature_dim={feature_dim}")
    print(f"- out_root: {out_root.resolve()}")
    if zip_output: print(f"- zip: {Path(zip_output).resolve()}")

if __name__ == "__main__":
    INPUT_CSV = "bpi_challenge_2014.csv"
    OUT_DIR   = "tgn_input_2014_cases_Interaction"
    ZIP_PATH  = None
    build_tgn_per_case_2014_interaction(INPUT_CSV, OUT_DIR, ZIP_PATH)
