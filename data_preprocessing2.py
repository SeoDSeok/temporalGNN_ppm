# make_tgn_input_activity_final_per_case_min.py
import re
from pathlib import Path
import numpy as np
import pandas as pd

def _to_epoch(v):
    # 이미 정수 epoch면 그대로, 아니면 파싱
    if pd.api.types.is_integer(v) or isinstance(v, (int, np.integer)):
        return int(v)
    return int(pd.to_datetime(v, utc=True, errors="coerce").value // 10**9)

def _safe(s: str) -> str:
    s = str(s)
    return re.sub(r"[^\w\-\.]+", "_", s)[:120]

def make_tgn_input_per_case(
    csv_path: str,
    out_dir: str = "tgn_input_af_per_case",
    case_col: str = "case_id",
    ts_col: str = "time:timestamp",
    act_col: str = "activity_final",
    feature_dim: int = 32,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)
    # 필수 컬럼 체크
    for c in (case_col, ts_col, act_col):
        if c not in df.columns:
            raise ValueError(f"필수 컬럼 '{c}' 가 없습니다. columns={df.columns.tolist()}")

    # epoch 정리 + 정렬
    df = df.copy()
    df["__epoch__"] = df[ts_col].apply(_to_epoch)
    df = df.dropna(subset=["__epoch__", case_col, act_col]).reset_index(drop=True)
    df["__epoch__"] = df["__epoch__"].astype("int64")
    df = df.sort_values([case_col, "__epoch__"], kind="mergesort")

    # 케이스별 저장
    summary = []
    for cid, g in df.groupby(case_col, sort=False):
        g = g.sort_values("__epoch__", kind="mergesort").reset_index(drop=True)
        g["node_id"] = np.arange(len(g), dtype=np.int64)

        cdir = out / f"case_{_safe(cid)}"
        cdir.mkdir(parents=True, exist_ok=True)

        # nodes.csv
        nodes = pd.DataFrame({
            "id": g["node_id"],
            "activity": g[act_col].astype(str),
            "timestamp_epoch": g["__epoch__"].astype("int64")
        })
        nodes.to_csv(cdir / "nodes.csv", index=False)

        # edges.csv (연속 이벤트 & t_v > t_u 만)
        if len(g) >= 2:
            src = g["node_id"].values[:-1]
            dst = g["node_id"].values[1:]
            t_u = g["__epoch__"].values[:-1]
            t_v = g["__epoch__"].values[1:]
            mask = t_v > t_u
            edges = pd.DataFrame({"src": src[mask], "dst": dst[mask]})
        else:
            edges = pd.DataFrame(columns=["src", "dst"])
        edges.to_csv(cdir / "edges.csv", index=False)

        # node_features.npy (zeros)
        feats = np.zeros((len(nodes), feature_dim), dtype=np.float32)
        np.save(cdir / "node_features.npy", feats)

        summary.append({
            "case_id": cid,
            "num_nodes": int(len(nodes)),
            "num_edges": int(len(edges)),
            "folder": str(cdir)
        })

    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    print(f"[DONE] cases={len(summary)} | feature_dim={feature_dim} | out={out.resolve()}")

# --------- 예시 실행 (직접 값 넣어서 호출) ---------
if __name__ == "__main__":
    make_tgn_input_per_case(
        csv_path="bpi2017_with_final_activity.csv",
        out_dir="tgn_input_af_per_case",
        case_col="case",
        ts_col="ocel:timestamp",
        act_col="activity_final",
        feature_dim=32,
    )
