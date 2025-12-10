# tgn_multitask.py
# TGN-style Multi-task:
#   - next activity (classification)
#   - next Δt (minutes, normalized in [0,1] for training; report in minutes)
#   - remaining time to case end (days, normalized in [0,1] for training; report in days)

import os, json, glob, random, time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------- Speed/Determinism -----------------
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def count_params(model):
    m = model.module if isinstance(model, nn.DataParallel) else model
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ----------------- Small timing helper -----------------
class EpochTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self):
        self.fwd = 0.0; self.bwd = 0.0; self.opt = 0.0; self.io = 0.0
        self.steps = 0
    def start_epoch(self): self.reset(); self._t0 = time.perf_counter()
    def end_epoch(self): _cuda_sync(); return time.perf_counter() - self._t0
    def add_io(self, s): self.io += s
    def step_done(self): self.steps += 1
    def _evt(self):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        return e0, e1
    def summary(self, wall_s):
        if self.steps == 0: self.steps = 1
        print(f"[TIMING] wall={wall_s:.2f}s | per-step: "
              f"IO={self.io/self.steps:.4f}s, FWD={self.fwd/self.steps:.4f}s, "
              f"BWD={self.bwd/self.steps:.4f}s, OPT={self.opt/self.steps:.4f}s")

@torch.no_grad()
def benchmark_inference(model, loader, device, max_batches=None, desc="Inference"):
    model.eval()
    n_ex = 0
    _cuda_sync()
    t0 = time.perf_counter()
    for i, batch in enumerate(tqdm(loader, desc=f"[Bench] {desc}", leave=False)):
        if max_batches is not None and i >= max_batches:
            break
        x_u, x_ctx, t_ctx, y_dt, y_cls, act_u, act_ctx, y_rem = batch
        x_u   = x_u.to(device, non_blocking=True)
        x_ctx = x_ctx.to(device, non_blocking=True)
        t_ctx = t_ctx.to(device, non_blocking=True)
        act_u   = act_u.to(device, non_blocking=True)
        act_ctx = act_ctx.to(device, non_blocking=True)
        # forward only
        logits, y_hat_dt, y_hat_rem = model(x_u, x_ctx, t_ctx, act_u.long(), act_ctx.long())
        n_ex += x_u.size(0)
    _cuda_sync()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    return {
        "examples": n_ex,
        "time_s": elapsed,
        "throughput_ex_s": n_ex / max(elapsed, 1e-9),
        "latency_ms_per_ex": (elapsed / max(n_ex, 1)) * 1000.0
    }

# ----------------- Dataset -----------------
class TemporalGraphDataset(Dataset):
    """
    각 케이스의 연속 엣지 (u->v)를 하나의 샘플로 사용.
    반환 텐서: (x_u, x_ctx, t_ctx, y_dt_norm, y_cls, act_u, act_ctx, y_rem_norm)
      - x_u:        (D,)
      - x_ctx:      (L,D)
      - t_ctx:      (L,)   (u 기준 과거 이벤트들과의 시차 [sec], raw)
      - y_dt_norm:  (1,)   (정규화된 Δt[min] in [0,1])
      - y_cls:      ()     (다음 activity id)
      - act_u:      ()     (현재 u의 activity idx +1; 0=pad)
      - act_ctx:    (L,)   (컨텍스트 activity idx +1; 0=pad)
      - y_rem_norm: (1,)   (정규화된 remaining days in [0,1])
    """
    def __init__(self, root_dir: str, L: int = 20, split: str = "train",
                 split_ratio=(0.8, 0.1, 0.1), seed=42):
        super().__init__()
        self.root = Path(root_dir)
        self.L = L
        self.meta = json.load(open(self.root / "metadata.json", "r", encoding="utf-8"))

        # 1) 모든 케이스 나열 + split
        all_case_dirs = sorted([Path(p) for p in glob.glob(str(self.root / "case_*")) if Path(p).is_dir()])
        split_file = self.root / "case_split.json"
        if split_file.exists():
            sp = json.load(open(split_file, "r", encoding="utf-8"))
        else:
            names = [d.name for d in all_case_dirs]
            rng = random.Random(seed); rng.shuffle(names)
            n = len(names); n_train = int(n * split_ratio[0]); n_val = int(n * split_ratio[1])
            sp = {"train": names[:n_train], "val": names[n_train:n_train+n_val], "test": names[n_train+n_val:]}
            with open(split_file, "w", encoding="utf-8") as f:
                json.dump(sp, f, indent=2, ensure_ascii=False)
        case_dirs = [d for d in all_case_dirs if d.name in set(sp[split])]

        # 2) activity vocab (전 케이스 기준)
        self.activity2idx: Dict[str, int] = {}
        self.idx2activity: List[str] = []
        for cdir in all_case_dirs:
            nodes = pd.read_csv(cdir / "nodes.csv")
            for a in nodes["activity"].astype(str).values.tolist():
                if a not in self.activity2idx:
                    self.activity2idx[a] = len(self.idx2activity)
                    self.idx2activity.append(a)

        # 3) Δt(분), 잔여일(일) 전역 통계 (min-max norm)
        all_dt_min = []
        all_rem_days = []
        for cdir in all_case_dirs:
            nodes = pd.read_csv(cdir / "nodes.csv")
            edges = pd.read_csv(cdir / "edges.csv")
            if len(nodes) == 0: continue
            ts = nodes["timestamp_epoch"].astype(int).to_numpy()
            case_end = int(ts[-1])

            for i in range(len(edges)):
                u = int(edges.loc[i, "src"]); v = int(edges.loc[i, "dst"])
                t_u, t_v = int(ts[u]), int(ts[v])
                if t_v > t_u:
                    all_dt_min.append((t_v - t_u) / 60.0)                  # minutes
                if case_end > t_u:
                    all_rem_days.append((case_end - t_u) / 86400.0)        # days

        if not all_dt_min:
            raise RuntimeError("No valid (t_v > t_u) edges to compute Δt stats.")
        if not all_rem_days:
            raise RuntimeError("No valid remaining-time samples to compute remaining-time stats.")

        self.dt_min = float(np.min(all_dt_min))
        self.dt_max = float(np.max(all_dt_min))
        self.dt_denom = (self.dt_max - self.dt_min) + 1e-8

        self.rem_min = float(np.min(all_rem_days))
        self.rem_max = float(np.max(all_rem_days))
        self.rem_denom = (self.rem_max - self.rem_min) + 1e-8

        print(f"[Δt(min)]  min={self.dt_min:.4f} | max={self.dt_max:.4f}")
        print(f"[Rem(days)] min={self.rem_min:.4f} | max={self.rem_max:.4f}")

        # 4) 샘플 구축(현재 split)
        samples = []
        for cdir in case_dirs:
            nodes = pd.read_csv(cdir / "nodes.csv")
            edges = pd.read_csv(cdir / "edges.csv")
            if len(nodes) == 0: continue

            ts = nodes["timestamp_epoch"].astype(int).to_numpy()
            acts = nodes["activity"].astype(str).to_numpy()
            case_end = int(ts[-1])

            for i in range(len(edges)):
                u = int(edges.loc[i, "src"]); v = int(edges.loc[i, "dst"])
                t_u, t_v = int(ts[u]), int(ts[v])
                if t_v <= t_u:  # skip invalid
                    continue

                start = max(0, u - self.L)
                ctx_ids = list(range(start, u))
                ctx_ids = ([-1] * (self.L - len(ctx_ids))) + ctx_ids

                # 레이블: Δt(min) 정규화
                raw_dt_min = (t_v - t_u) / 60.0
                y_dt_norm = (raw_dt_min - self.dt_min) / self.dt_denom

                # 레이블: 남은 시간(일) 정규화
                raw_rem_days = (case_end - t_u) / 86400.0
                y_rem_norm = (raw_rem_days - self.rem_min) / self.rem_denom

                samples.append({
                    "case_dir": str(cdir),
                    "u": u, "v": v,
                    "context_ids": ctx_ids,
                    "t_u": t_u,
                    "y_activity": self.activity2idx[acts[v]],
                    "y_dt_norm": float(y_dt_norm),
                    "y_rem_norm": float(y_rem_norm),
                })

        self.samples = samples
        self.feature_dim = int(self.meta["feature_dim"])
        self.num_classes = len(self.idx2activity)

        # 5) 캐시
        self._cache_case = None
        self._cache_feats = None
        self._cache_ts = None
        self._cache_act_idx = None

    def __len__(self): return len(self.samples)

    def _ensure_case(self, case_dir: str):
        if self._cache_case == case_dir:
            return
        feats = np.load(Path(case_dir) / "node_features.npy")
        nodes = pd.read_csv(Path(case_dir) / "nodes.csv")
        self._cache_case = case_dir
        self._cache_feats = feats
        self._cache_ts = nodes["timestamp_epoch"].astype(int).to_numpy()
        self._cache_act_idx = np.array(
            [self.activity2idx[a] for a in nodes["activity"].astype(str)], dtype=np.int64
        )

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        self._ensure_case(s["case_dir"])

        u = s["u"]; t_u = s["t_u"]; ctx_ids = s["context_ids"]

        x_u = self._cache_feats[u]
        x_ctx = np.stack([
            self._cache_feats[c] if c >= 0 else np.zeros_like(self._cache_feats[0])
        for c in ctx_ids], axis=0)
        t_ctx = np.array(
            [(t_u - self._cache_ts[c]) if c >= 0 else 0 for c in ctx_ids],
            dtype=np.float32
        )  # shape (L,)

        act_u = self._cache_act_idx[u] + 1
        act_ctx = np.array([(self._cache_act_idx[c] + 1) if c >= 0 else 0 for c in ctx_ids], dtype=np.int64)

        y_cls = s["y_activity"]
        y_dt_norm = s["y_dt_norm"]
        y_rem_norm = s["y_rem_norm"]

        return (
            torch.from_numpy(x_u).float(),            # (D,)
            torch.from_numpy(x_ctx).float(),          # (L,D)
            torch.from_numpy(t_ctx).float(),          # (L,)
            torch.tensor([y_dt_norm]).float(),        # (1,)
            torch.tensor(y_cls).long(),               # ()
            torch.tensor(act_u).long(),               # ()
            torch.from_numpy(act_ctx).long(),         # (L,)
            torch.tensor([y_rem_norm]).float(),       # (1,)
        )

# ----------------- Time Encoding -----------------
class TimeEncoding(nn.Module):
    def __init__(self, out_dim: int, w_init: float = 1.0):
        super().__init__()
        self.freq  = nn.Parameter(torch.randn(out_dim) * w_init)
        self.phase = nn.Parameter(torch.zeros(out_dim))
    def forward(self, delta_t):  # (B,L) in seconds
        z = torch.log1p(delta_t.clamp_min(0.0))
        z = z.unsqueeze(-1) * self.freq + self.phase  # (B,L,T)
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)  # (B,L,2T)

# ----------------- Model (TGN-style, 3 heads) -----------------
class TGNPredictor(nn.Module):
    """
    메시지 φ → GRU 메모리 업데이트 → 리드아웃 g
    Heads:
      - cls_head:      next activity
      - reg_next_head: next Δt (normalized)
      - reg_rem_head:  remaining time to case end (normalized)
    """
    def __init__(self,
                 in_dim: int,
                 num_classes: int,
                 act_emb_dim: int = 32,
                 time_dim: int = 8,
                 hidden: int = 256,
                 msg_hidden: int = 256,
                 mlp_hidden: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.hidden = hidden

        self.act_emb = nn.Embedding(num_classes + 1, act_emb_dim, padding_idx=0)
        self.time_enc = TimeEncoding(time_dim)

        msg_in = in_dim + act_emb_dim + 2 * time_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_in, msg_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(msg_hidden, hidden)
        )
        self.updater = nn.GRU(input_size=hidden, hidden_size=hidden, batch_first=True)

        read_in = in_dim + act_emb_dim + hidden
        self.readout = nn.Sequential(
            nn.Linear(read_in, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.cls_head      = nn.Linear(mlp_hidden, num_classes)
        self.reg_next_head = nn.Linear(mlp_hidden, 1)
        self.reg_rem_head  = nn.Linear(mlp_hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x_u, x_ctx, t_ctx, act_u_idx, act_ctx_idx):
        B, L, D = x_ctx.shape
        mask = (act_ctx_idx > 0).float()           # (B,L)
        e_u   = self.act_emb(act_u_idx)            # (B,E)
        e_ctx = self.act_emb(act_ctx_idx)          # (B,L,E)
        te    = self.time_enc(t_ctx)               # (B,L,2T)

        msg_in = torch.cat([x_ctx, e_ctx, te], dim=-1)  # (B,L,D+E+2T)
        m = self.msg_mlp(msg_in)                        # (B,L,H)
        m = m * mask.unsqueeze(-1)                      # pad 0

        lengths = mask.sum(dim=1).clamp_min(1).long()
        h0 = torch.zeros(1, B, self.hidden, device=x_u.device)
        m_packed = nn.utils.rnn.pack_padded_sequence(m, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_last = self.updater(m_packed, h0)
        mem_u = h_last.squeeze(0)                       # (B,H)

        z_in = torch.cat([x_u, e_u, mem_u], dim=-1)     # (B,D+E+H)
        h = self.readout(z_in)
        logits  = self.cls_head(h)
        dt_hat  = torch.sigmoid(self.reg_next_head(h))  # normalized [0,1]
        rem_hat = torch.sigmoid(self.reg_rem_head(h))   # normalized [0,1]
        return logits, dt_hat, rem_hat

# ----------------- Train/Eval -----------------
def collate(batch):
    x_u, x_ctx, t_ctx, y_dt, y_cls, act_u, act_ctx, y_rem = zip(*batch)
    return (torch.stack(x_u, 0),
            torch.stack(x_ctx, 0),
            torch.stack(t_ctx, 0),
            torch.stack(y_dt, 0),
            torch.stack(y_cls, 0),
            torch.stack(act_u, 0),
            torch.stack(act_ctx, 0),
            torch.stack(y_rem, 0))

def _rmse_from_values(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(arr))))

def train_one_epoch_timed(model, loader, opt, device, ce_crit, reg_crit,
                          lambda_dt=1.0, lambda_rem=1.0, clip=1.0):
    model.train()
    sum_total = sum_ce = sum_dt = sum_rem = 0.0
    n = 0

    timer = EpochTimer(device)
    opt.zero_grad(set_to_none=True)
    timer.start_epoch()

    for batch in loader:
        # --- I/O ---
        ti0 = time.perf_counter()
        x_u, x_ctx, t_ctx, y_dt, y_cls, act_u, act_ctx, y_rem = batch
        x_u   = x_u.to(device, non_blocking=True)
        x_ctx = x_ctx.to(device, non_blocking=True)
        t_ctx = t_ctx.to(device, non_blocking=True)
        y_dt  = y_dt.to(device, non_blocking=True)
        y_cls = y_cls.to(device, non_blocking=True)
        y_rem = y_rem.to(device, non_blocking=True)
        act_u   = act_u.to(device, non_blocking=True)
        act_ctx = act_ctx.to(device, non_blocking=True)
        _cuda_sync(); timer.add_io(time.perf_counter() - ti0)

        # --- FORWARD ---
        if device.type == "cuda":
            e0, e1 = timer._evt(); e0.record()
            logits, y_hat_dt, y_hat_rem = model(x_u, x_ctx, t_ctx, act_u.long(), act_ctx.long())
            loss_ce  = ce_crit(logits, y_cls)
            loss_dt  = reg_crit(y_hat_dt,  y_dt)
            loss_rem = reg_crit(y_hat_rem, y_rem)
            loss = loss_ce + lambda_dt*loss_dt + lambda_rem*loss_rem
            e1.record(); _cuda_sync()
            timer.fwd += e0.elapsed_time(e1) / 1000.0
        else:
            t0 = time.perf_counter()
            logits, y_hat_dt, y_hat_rem = model(x_u, x_ctx, t_ctx, act_u.long(), act_ctx.long())
            loss_ce  = ce_crit(logits, y_cls)
            loss_dt  = reg_crit(y_hat_dt,  y_dt)
            loss_rem = reg_crit(y_hat_rem, y_rem)
            loss = loss_ce + lambda_dt*loss_dt + lambda_rem*loss_rem
            timer.fwd += time.perf_counter() - t0

        # --- BACKWARD ---
        if device.type == "cuda":
            e0, e1 = timer._evt(); e0.record()
            loss.backward()
            e1.record(); _cuda_sync()
            timer.bwd += e0.elapsed_time(e1) / 1000.0
        else:
            t0 = time.perf_counter(); loss.backward(); timer.bwd += time.perf_counter() - t0

        # --- OPTIM ---
        if device.type == "cuda":
            e0, e1 = timer._evt(); e0.record()
            if clip: torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); opt.zero_grad(set_to_none=True)
            e1.record(); _cuda_sync()
            timer.opt += e0.elapsed_time(e1) / 1000.0
        else:
            t0 = time.perf_counter()
            if clip: torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); opt.zero_grad(set_to_none=True)
            timer.opt += time.perf_counter() - t0

        # --- metrics ---
        bs = x_u.size(0)
        sum_total += loss.item() * bs
        sum_ce    += loss_ce.item() * bs
        sum_dt    += loss_dt.item() * bs
        sum_rem   += loss_rem.item() * bs
        n += bs
        timer.step_done()

    wall = timer.end_epoch()
    timer.summary(wall)
    return {
        "loss": sum_total / n, "loss_ce": sum_ce / n,
        "loss_dt": sum_dt / n, "loss_rem": sum_rem / n
    }

@torch.no_grad()
def evaluate(model, loader, device, ce_crit, reg_crit,
             lambda_dt=1.0, lambda_rem=1.0, desc="Evaluating"):
    model.eval()
    n = 0
    sum_total = sum_ce = sum_dt = sum_rem = 0.0
    correct1 = correct3 = 0
    all_pred, all_true = [], []

    ds = loader.dataset
    dt_min, dt_denom = ds.dt_min, ds.dt_denom
    rem_min, rem_denom = ds.rem_min, ds.rem_denom

    dt_abs_err_list = []    # minutes
    rem_abs_err_list = []   # days
    dt_val_list = []        # minutes (for RMSE)
    dt_true_list = []
    rem_val_list = []       # days (for RMSE)
    rem_true_list = []

    for x_u, x_ctx, t_ctx, y_dt, y_cls, act_u, act_ctx, y_rem in tqdm(loader, desc=desc, leave=False):
        x_u, x_ctx, t_ctx = x_u.to(device), x_ctx.to(device), t_ctx.to(device)
        y_dt, y_cls, y_rem = y_dt.to(device), y_cls.to(device), y_rem.to(device)
        act_u, act_ctx = act_u.to(device), act_ctx.to(device)

        logits, y_hat_dt, y_hat_rem = model(x_u, x_ctx, t_ctx, act_u, act_ctx)

        loss_ce  = ce_crit(logits, y_cls)
        loss_dt  = reg_crit(y_hat_dt,  y_dt)
        loss_rem = reg_crit(y_hat_rem, y_rem)
        loss = loss_ce + lambda_dt * loss_dt + lambda_rem * loss_rem

        bs = x_u.size(0)
        sum_total += loss.item() * bs
        sum_ce    += loss_ce.item() * bs
        sum_dt    += loss_dt.item() * bs
        sum_rem   += loss_rem.item() * bs

        pred1 = logits.argmax(-1)
        top3  = logits.topk(k=min(3, logits.size(-1)), dim=-1).indices
        correct1 += (pred1 == y_cls).sum().item()
        correct3 += top3.eq(y_cls.unsqueeze(-1)).any(dim=-1).sum().item()
        all_pred += pred1.detach().cpu().tolist()
        all_true += y_cls.detach().cpu().tolist()
        n += bs

        # ---- 역정규화 (report in minutes / days) ----
        y_dt_pred = (y_hat_dt.detach().cpu().numpy().ravel() * dt_denom + dt_min)     # minutes
        y_dt_true = (y_dt.detach().cpu().numpy().ravel()      * dt_denom + dt_min)     # minutes
        y_rem_pred= (y_hat_rem.detach().cpu().numpy().ravel() * rem_denom + rem_min)   # days
        y_rem_true= (y_rem.detach().cpu().numpy().ravel()     * rem_denom + rem_min)   # days

        dt_abs_err_list.append(np.abs(y_dt_pred - y_dt_true))
        rem_abs_err_list.append(np.abs(y_rem_pred - y_rem_true))
        dt_val_list.append(y_dt_pred); dt_true_list.append(y_dt_true)
        rem_val_list.append(y_rem_pred); rem_true_list.append(y_rem_true)

    from sklearn.metrics import f1_score
    acc = correct1 / max(n,1); top3 = correct3 / max(n,1)
    f1m = f1_score(all_true, all_pred, average="macro") if n else float("nan")

    dt_mae_min  = float(np.concatenate(dt_abs_err_list).mean()) if dt_abs_err_list else float("nan")
    dt_rmse_min = _rmse_from_values(np.concatenate(dt_val_list) - np.concatenate(dt_true_list)) if dt_val_list else float("nan")
    rem_mae_d   = float(np.concatenate(rem_abs_err_list).mean()) if rem_abs_err_list else float("nan")
    rem_rmse_d  = _rmse_from_values(np.concatenate(rem_val_list) - np.concatenate(rem_true_list)) if rem_val_list else float("nan")

    return {
        "loss": sum_total / max(n,1), "loss_ce": sum_ce / max(n,1),
        "loss_dt": sum_dt / max(n,1), "loss_rem": sum_rem / max(n,1),
        "acc": acc, "top3": top3, "f1_macro": f1m,
        "mae_dt_minutes": dt_mae_min, "rmse_dt_minutes": dt_rmse_min,
        "mae_rem_days": rem_mae_d, "rmse_rem_days": rem_rmse_d
    }

# ----------------- main -----------------
def main():
    set_seed(42)

    # === 경로/하이퍼파라미터 ===
    # root = "./tgn_input"
    root = "./tgn_input_2019_cases_cID"
    # root = "./tgn_input_2016_cases_SessionID_2"
    # root = "./tgn_input_2014_cases_Interaction"
    # root = "./tgn_input_af_per_case"

    L = 20
    bs = 256
    epochs = 50
    lr = 1e-3
    weight_decay = 0.0
    dropout = 0.2
    hidden = 256
    msg_hidden = 256
    mlp_hidden = 256
    lambda_dt = 1.0
    lambda_rem = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Dataset / Loader ===
    train_ds = TemporalGraphDataset(root, L=L, split="train")
    val_ds   = TemporalGraphDataset(root, L=L, split="val")
    test_ds  = TemporalGraphDataset(root, L=L, split="test")

    def make_ld(ds, bs, shuffle):
        nw = max(0, os.cpu_count()//2)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate,
                          num_workers=nw, pin_memory=True,
                          persistent_workers=(nw > 0), prefetch_factor=2 if nw > 0 else None)

    train_ld = make_ld(train_ds, bs, True)
    val_ld   = make_ld(val_ds,   bs, False)
    test_ld  = make_ld(test_ds,  bs, False)

    # === Model ===
    model = TGNPredictor(
        in_dim=train_ds.feature_dim,
        num_classes=train_ds.num_classes,
        act_emb_dim=32, time_dim=8,
        hidden=hidden, msg_hidden=msg_hidden, mlp_hidden=mlp_hidden, dropout=dropout
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce_crit  = nn.CrossEntropyLoss()
    reg_crit = nn.L1Loss()

    BEST_PATH, LAST_PATH = "tgn_multi_best.pt", "tgn_multi_last.pt"
    best_acc = 0.0

    n_params = count_params(model)
    print(f"[MODEL] trainable params = {n_params:,}")

    total_train_sec = 0.0
    for ep in tqdm(range(1, epochs+1), desc="Epochs"):
        _cuda_sync(); t_ep0 = time.perf_counter()
        tr = train_one_epoch_timed(model, train_ld, opt, device, ce_crit, reg_crit,
                                   lambda_dt=lambda_dt, lambda_rem=lambda_rem, clip=1.0)
        _cuda_sync(); t_ep = time.perf_counter() - t_ep0
        total_train_sec += t_ep

        va = evaluate(model, val_ld, device, ce_crit, reg_crit,
                      lambda_dt=lambda_dt, lambda_rem=lambda_rem, desc="Val")
        print(f"[TIME] epoch {ep:03d}: train {t_ep:.2f}s | cum {total_train_sec:.2f}s\n")
        print(f"[EP{ep:03d}] "
              f"train_total={tr['loss']:.4f} (CE={tr['loss_ce']:.4f}, dt={tr['loss_dt']:.4f}, rem={tr['loss_rem']:.4f}) | "
              f"val_total={va['loss']:.4f} (CE={va['loss_ce']:.4f}, dt={va['loss_dt']:.4f}, rem={va['loss_rem']:.4f}) | "
              f"val_acc={va['acc']:.4f} | val_top3={va['top3']:.4f} | f1M={va['f1_macro']:.4f} | "
              f"MAE(dt_min)={va['mae_dt_minutes']:.2f} RMSE(dt_min)={va['rmse_dt_minutes']:.2f} | "
              f"MAE(rem_days)={va['mae_rem_days']:.2f} RMSE(rem_days)={va['rmse_rem_days']:.2f}")

        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        meta = {"activity_vocab": train_ds.idx2activity, "feature_dim": train_ds.feature_dim, "L": L}
        if va["acc"] > best_acc:
            best_acc = va["acc"]
            torch.save({"model": state, "meta": meta}, BEST_PATH)
        torch.save({"model": state, "meta": meta}, LAST_PATH)

    # === Test ===
    load_path = BEST_PATH if os.path.exists(BEST_PATH) else LAST_PATH
    ckpt = torch.load(load_path, map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])

    te = evaluate(model, test_ld, device, ce_crit, reg_crit,
                  lambda_dt=lambda_dt, lambda_rem=lambda_rem, desc="Test")
    print(f"[TEST] total={te['loss']:.4f} (CE={te['loss_ce']:.4f}, dt={te['loss_dt']:.4f}, rem={te['loss_rem']:.4f}) | "
          f"acc={te['acc']:.4f} | top3={te['top3']:.4f} | f1M={te['f1_macro']:.4f} | "
          f"MAE(dt_min)={te['mae_dt_minutes']:.2f} RMSE(dt_min)={te['rmse_dt_minutes']:.2f} | "
          f"MAE(rem_days)={te['mae_rem_days']:.2f} RMSE(rem_days)={te['rmse_rem_days']:.2f}")

    print("-" * 80)
    print("\n[Benchmark] Inference speed on TEST loader (forward-only)")
    bench = benchmark_inference(model, test_ld, device, max_batches=None, desc="Inference")
    print(f" - examples: {bench['examples']}")
    print(f" - time: {bench['time_s']:.3f}s")
    print(f" - throughput: {bench['throughput_ex_s']:.2f} ex/s")
    print(f" - latency: {bench['latency_ms_per_ex']:.3f} ms/example")

if __name__ == "__main__":
    main()
