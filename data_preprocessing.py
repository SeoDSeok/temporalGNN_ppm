from __future__ import annotations
import os, json, zipfile
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# ---------------------------
# Helpers
# ---------------------------

APP_COLS  = ["Application Type", "ApplicationType", "AppType"]
GOAL_COLS = ["LoanGoal", "Loan Goal", "Purpose"]
CS_COLS   = ["CreditScore", "creditScore", "Credit Score", "credit"]

def detect_timestamp_column(df: pd.DataFrame) -> str:
    candidates = ["ocel:timestamp", "time:timestamp", "timestamp",
                  "Start Timestamp", "completeTime", "end_timestamp"]
    for c in candidates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
            return c
    raise ValueError("No timestamp-like column found. Checked: " + ", ".join(candidates))

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    mapping = {}
    # event id
    if "ocel:eid" in df.columns: mapping["event_id"] = "ocel:eid"
    elif "event_id" in df.columns: mapping["event_id"] = "event_id"
    else:
        df["__tmp_eid__"] = np.arange(len(df))
        mapping["event_id"] = "__tmp_eid__"
    # activity
    for c in ["ocel:activity", "Activity", "concept:name"]:
        if c in df.columns:
            mapping["activity"] = c
            break
    else:
        raise ValueError("No activity column found (expected one of ocel:activity/Activity/concept:name).")
    # lifecycle
    if "lifecycle" in df.columns: mapping["lifecycle"] = "lifecycle"
    elif "lifecycle:transition" in df.columns: mapping["lifecycle"] = "lifecycle:transition"
    else:
        df["__lifecycle__"] = "UNKNOWN"; mapping["lifecycle"] = "__lifecycle__"
    # action
    if "Action" in df.columns: mapping["action"] = "Action"
    elif "action" in df.columns: mapping["action"] = "action"
    else:
        df["__action__"] = "UNKNOWN"; mapping["action"] = "__action__"
    # case
    if "case" in df.columns: mapping["case"] = "case"
    elif "case:concept:name" in df.columns: mapping["case"] = "case:concept:name"
    else:
        raise ValueError("No case id column found (expected 'case' or 'case:concept:name').")
    # optional (detected later too)
    return mapping

def build_vocab(series: pd.Series) -> Tuple[List[str], Dict[str, int]]:
    vals = sorted(series.astype(str).fillna("UNKNOWN").unique().tolist())
    idx = {v: i for i, v in enumerate(vals)}
    return vals, idx

def onehot(index: int, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    if 0 <= index < dim:
        v[index] = 1.0
    return v

# ---------------------------
# Core
# ---------------------------

def build_feature_row(
    action_val: str, lifecycle_val: str,
    action2idx: Dict[str, int], lifecycle2idx: Dict[str, int],
    a_dim: int, l_dim: int,
    app_val: str | None, goal_val: str | None, credit_val: float | None,
    app2idx: Dict[str, int] | None, goal2idx: Dict[str, int] | None,
    app_dim: int, goal_dim: int
) -> np.ndarray:
    parts = []
    # Action/Lifecycle
    parts.append(onehot(action2idx[str(action_val)], a_dim))
    parts.append(onehot(lifecycle2idx[str(lifecycle_val)], l_dim))
    # AppType
    if app2idx is not None and app_dim > 0 and app_val is not None:
        parts.append(onehot(app2idx[str(app_val)], app_dim))
    # LoanGoal
    if goal2idx is not None and goal_dim > 0 and goal_val is not None:
        parts.append(onehot(goal2idx[str(goal_val)], goal_dim))
    # CreditScore (already normalized [0,1])
    if credit_val is not None:
        parts.append(np.array([float(credit_val)], dtype=np.float32))
    return np.concatenate(parts, axis=0)

def write_case_graph(
    sub: pd.DataFrame, cols: Dict[str, str], ts_col: str, out_case_dir: Path,
    action2idx: Dict[str, int], lifecycle2idx: Dict[str, int], a_dim: int, l_dim: int,
    app_col: str | None, goal_col: str | None, cs_col: str | None,
    app2idx: Dict[str, int] | None, goal2idx: Dict[str, int] | None,
    app_dim: int, goal_dim: int, cs_min: float | None, cs_max: float | None
) -> Tuple[int, int]:
    sub = sub.copy().sort_values(by=ts_col).reset_index(drop=True)
    sub["__node_id__"] = range(len(sub))

    # normalize credit score to [0,1] (clip to robust range)
    if cs_col is not None and cs_min is not None and cs_max is not None:
        cs = pd.to_numeric(sub[cs_col], errors="coerce")
        mid = (cs_min + cs_max) / 2.0
        cs = cs.fillna(mid)
        cs = cs.clip(lower=cs_min, upper=cs_max)
        cs_norm = (cs - cs_min) / (cs_max - cs_min + 1e-8)
    else:
        cs_norm = pd.Series([np.nan] * len(sub), index=sub.index)

    # nodes.csv (옵션 컬럼도 같이 저장해두면 디버깅 편함)
    nodes = pd.DataFrame({
        "node_id": sub["__node_id__"],
        "event_eid": sub[cols["event_id"]].astype(str),
        "activity": sub[cols["activity"]].astype(str),
        "lifecycle": sub[cols["lifecycle"]].astype(str),
        "action": sub[cols["action"]].astype(str),
        "timestamp_iso": sub[ts_col].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "timestamp_epoch": (sub[ts_col].view("int64") // 10**9).astype("int64"),
        "case_id": sub[cols["case"]].iloc[0],
    })
    if app_col:  nodes["application_type"] = sub[app_col].astype(str)
    if goal_col: nodes["loan_goal"]       = sub[goal_col].astype(str)
    if cs_col:   nodes["credit_norm01"]   = cs_norm.astype(np.float32)

    out_case_dir.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(out_case_dir / "nodes.csv", index=False)

    # edges.csv
    if len(sub) >= 2:
        edges = pd.DataFrame({
            "src": sub["__node_id__"][:-1].to_numpy(),
            "dst": sub["__node_id__"][1:].to_numpy(),
            "timestamp_epoch": (sub[ts_col][1:].view("int64") // 10**9).astype("int64").to_numpy(),
        })
    else:
        edges = pd.DataFrame(columns=["src", "dst", "timestamp_epoch"])
    edges.to_csv(out_case_dir / "edges.csv", index=False)

    # node_features.npy
    feats = np.stack([
        build_feature_row(
            a, l, action2idx, lifecycle2idx, a_dim, l_dim,
            (sub[app_col].iat[i] if app_col else None),
            (sub[goal_col].iat[i] if goal_col else None),
            (cs_norm.iat[i] if cs_col else None),
            app2idx, goal2idx, app_dim, goal_dim
        )
        for i, (a, l) in enumerate(zip(sub[cols["action"]], sub[cols["lifecycle"]]))
    ], axis=0)
    np.save(out_case_dir / "node_features.npy", feats)

    return len(nodes), len(edges)

def zip_dir(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for f in files:
                full = Path(root) / f
                arc = full.relative_to(src_dir)
                zf.write(str(full), arcname=str(arc))

# ---------------------------
# Entry
# ---------------------------

def build_tgn_per_case(
    input_csv: str,
    out_root: str = "./tgn_input",
    zip_output: str | None = "./tgn_input_bpi2017_per_case.zip"
):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # CSV
    df = pd.read_csv(input_csv, low_memory=False)
    ts_col = detect_timestamp_column(df)
    cols = detect_columns(df)

    # optional column names
    app_col  = _pick_col(df, APP_COLS)
    goal_col = _pick_col(df, GOAL_COLS)
    cs_col   = _pick_col(df, CS_COLS)

    # 필수 최소 정리
    df = df.dropna(subset=[cols["case"], ts_col, cols["activity"]]).copy()
    df[cols["lifecycle"]] = df[cols["lifecycle"]].fillna("UNKNOWN").astype(str)
    df[cols["action"]]    = df[cols["action"]].fillna("UNKNOWN").astype(str)
    if app_col:  df[app_col]  = df[app_col].fillna("UNKNOWN").astype(str)
    if goal_col: df[goal_col] = df[goal_col].fillna("UNKNOWN").astype(str)

    # vocab들
    action_vocab, action2idx     = build_vocab(df[cols["action"]])
    lifecycle_vocab, lifecycle2idx = build_vocab(df[cols["lifecycle"]])
    a_dim, l_dim = len(action_vocab), len(lifecycle_vocab)

    if app_col:
        app_vocab, app2idx = build_vocab(df[app_col])
        app_dim = len(app_vocab)
    else:
        app_vocab, app2idx, app_dim = [], None, 0

    if goal_col:
        goal_vocab, goal2idx = build_vocab(df[goal_col])
        goal_dim = len(goal_vocab)
    else:
        goal_vocab, goal2idx, goal_dim = [], None, 0

    # CreditScore robust min–max (1~99퍼센타일)
    cs_min = cs_max = None
    if cs_col:
        cs_all = pd.to_numeric(df[cs_col], errors="coerce").dropna().to_numpy()
        if cs_all.size > 0:
            low, high = np.percentile(cs_all, [1, 99])
            cs_min, cs_max = float(low), float(high)

    # 메타데이터
    feature_dim = a_dim + l_dim + app_dim + goal_dim + (1 if cs_col else 0)
    meta = {
        "columns": {
            "timestamp": ts_col,
            "event_id": cols["event_id"],
            "activity": cols["activity"],
            "lifecycle": cols["lifecycle"],
            "action": cols["action"],
            "case": cols["case"],
            "application_type": app_col,
            "loan_goal": goal_col,
            "credit_score": cs_col
        },
        "vocabs": {
            "action": action_vocab,
            "lifecycle": lifecycle_vocab,
            "application_type": app_vocab,
            "loan_goal": goal_vocab
        },
        "scalers": {
            "credit_min": cs_min,
            "credit_max": cs_max
        },
        "feature_layout": {
            "action_onehot": a_dim,
            "lifecycle_onehot": l_dim,
            "application_type_onehot": app_dim,
            "loan_goal_onehot": goal_dim,
            "credit_norm01": 1 if cs_col else 0
        },
        "feature_dim": feature_dim,
        "edge_semantics": "Directed edges connect consecutive events in a case.",
        "node_semantics": "Features = [Action 1H | Lifecycle 1H | AppType 1H? | LoanGoal 1H? | CreditScore_norm?]"
    }
    with open(out_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # case loop
    case_ids = df[cols["case"]].astype(str).unique().tolist()
    summary_rows = []
    for cid in case_ids:
        sub = df[df[cols["case"]].astype(str) == cid]
        safe_cid = cid.replace("/", "_").replace("\\", "_")
        out_case_dir = out_root / f"case_{safe_cid}"
        n_nodes, n_edges = write_case_graph(
            sub, cols, ts_col, out_case_dir,
            action2idx, lifecycle2idx, a_dim, l_dim,
            app_col, goal_col, cs_col,
            app2idx, goal2idx, app_dim, goal_dim, cs_min, cs_max
        )
        summary_rows.append({"case_id": cid, "num_events": n_nodes, "num_edges": n_edges,
                             "case_folder": str(out_case_dir)})

    pd.DataFrame(summary_rows).sort_values("num_events", ascending=False).to_csv(out_root / "summary.csv", index=False)

    if zip_output:
        zip_dir(out_root, Path(zip_output))

    print(f"[DONE] Cases: {len(case_ids)} | Feature dim: {feature_dim}")
    print(f"- Output dir: {out_root.resolve()}")
    if zip_output: print(f"- ZIP: {Path(zip_output).resolve()}")

if __name__ == "__main__":
    INPUT_CSV = "bpi_challenge_data.csv"
    OUT_DIR = "tgn_input3"
    ZIP_PATH = None
    build_tgn_per_case(INPUT_CSV, OUT_DIR, ZIP_PATH)
