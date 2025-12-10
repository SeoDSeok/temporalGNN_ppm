# tgat_ppm_3task_v2.py
# TGAT multi-task: next-activity (cls) + next-Δt(min) (reg, normalized) + remaining-time(days) (reg, normalized)
# - Epoch timer with IO/FWD/BWD/OPT breakdown (CUDA events)
# - AMP optional (fp16/bf16) for speedup
# - Correct RMSE computation (on values, not squared errors of squared errors)
# - Safer casting for embedding indices on every forward path
# - Inference benchmark (forward-only throughput/latency)

import os, json, math, random, glob, time
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------- Speed/Determinism ----------------
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

# ---------------- Timing helpers ----------------
class EpochTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.reset()
    def reset(self):
        self.fwd = 0.0; self.bwd = 0.0; self.opt = 0.0; self.io = 0.0; self.steps = 0
    def _evt(self):
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    def start(self): self.reset(); self._t0 = time.perf_counter()
    def end(self): _cuda_sync(); return time.perf_counter() - self._t0
    def add_io(self, s): self.io += s
    def step_done(self): self.steps += 1
    def summary(self, wall):
        s = max(self.steps, 1)
        print(f"[TIMING] wall={wall:.2f}s | per-step: IO={self.io/s:.4f}s, FWD={self.fwd/s:.4f}s, "
              f"BWD={self.bwd/s:.4f}s, OPT={self.opt/s:.4f}s")

@torch.no_grad()
def benchmark_inference(model, loader, device, max_batches=None, desc="Inference"):
    """순수 forward만 측정 (loss/metric 제외)."""
    model.eval()
    n_ex, t_forward = 0, 0.0
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    for b_idx, batch in enumerate(tqdm(loader, desc=f"[Bench] {desc}", leave=False)):
        if max_batches is not None and b_idx >= max_batches:
            break
        # TGAT 입력 공통 8-튜플
        x_u, x_ctx, t_ctx, _, _, act_u, act_ctx, _ = batch
        x_u   = x_u.to(device, non_blocking=True)
        x_ctx = x_ctx.to(device, non_blocking=True)
        t_ctx = t_ctx.to(device, non_blocking=True)
        act_u = act_u.to(device, non_blocking=True).long()
        act_ctx = act_ctx.to(device, non_blocking=True).long()

        _cuda_sync()
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type=="cuda")):
            _ = model(x_u, x_ctx, t_ctx, act_u, act_ctx)
        _cuda_sync()
        t_forward += time.perf_counter() - t0
        n_ex += x_u.size(0)

    return {
        "examples": n_ex,
        "time_s": t_forward,
        "throughput_ex_s": (n_ex / t_forward) if t_forward > 0 else 0.0,
        "latency_ms_per_ex": (t_forward / max(n_ex,1)) * 1000.0
    }

# ================= Dataset =================
class TemporalGraphDataset(Dataset):
    """
    각 케이스의 연속 엣지 (u->v)를 하나의 샘플로 사용.
    반환: (x_u, x_ctx, t_ctx, y_dt_norm, y_cls, act_u, act_ctx, y_rem_norm)
      - y_dt_norm: Δt(min) min-max 정규화
      - y_rem_norm: remaining days min-max 정규화
    """
    def __init__(self, root_dir: str, L: int = 20, split: str = "train",
                 split_ratio=(0.8, 0.1, 0.1), seed=42):
        self.root = Path(root_dir)
        self.L = L
        self.meta = json.load(open(self.root / "metadata.json", "r", encoding="utf-8"))

        # --- 케이스 split ---
        all_case_dirs = sorted([Path(p) for p in glob.glob(str(self.root / "case_*")) if Path(p).is_dir()])
        split_file = self.root / "case_split.json"
        if split_file.exists():
            sp = json.load(open(split_file, "r", encoding="utf-8"))
        else:
            names = [d.name for d in all_case_dirs]
            rng = random.Random(seed); rng.shuffle(names)
            n = len(names); n_tr = int(n*split_ratio[0]); n_va = int(n*split_ratio[1])
            sp = {"train": names[:n_tr], "val": names[n_tr:n_tr+n_va], "test": names[n_tr+n_va:]}
            with open(split_file, "w", encoding="utf-8") as f:
                json.dump(sp, f, indent=2)
        case_dirs_split = [d for d in all_case_dirs if d.name in set(sp[split])]

        # --- activity vocab ---
        self.activity2idx: Dict[str, int] = {}
        self.idx2activity: List[str] = []
        for cdir in all_case_dirs:
            nodes = pd.read_csv(cdir / "nodes.csv")
            for a in nodes["activity"].astype(str).values.tolist():
                if a not in self.activity2idx:
                    self.activity2idx[a] = len(self.idx2activity)
                    self.idx2activity.append(a)

        # --- 전역 통계 (Δt[min], remaining[day]) ---
        all_deltas_min, all_remaining_days = [], []
        for cdir in tqdm(all_case_dirs, desc="Scanning stats"):
            nodes = pd.read_csv(cdir / "nodes.csv")
            edges = pd.read_csv(cdir / "edges.csv")
            if len(nodes) == 0: continue
            ts = nodes["timestamp_epoch"].astype(int).to_numpy()
            t_end = int(ts[-1])
            for i in range(len(edges)):
                u = int(edges.loc[i, "src"]); v = int(edges.loc[i, "dst"])
                t_u, t_v = int(ts[u]), int(ts[v])
                if t_v > t_u:
                    all_deltas_min.append((t_v - t_u) / 60.0)
                all_remaining_days.append(max(t_end - t_u, 0) / 86400.0)

        if not all_deltas_min:
            raise RuntimeError("No valid (t_v > t_u) edges found.")
        self.dt_min  = float(np.min(all_deltas_min))
        self.dt_max  = float(np.max(all_deltas_min))
        self.dt_denom = (self.dt_max - self.dt_min) + 1e-8

        self.rem_min = float(np.min(all_remaining_days)) if all_remaining_days else 0.0
        self.rem_max = float(np.max(all_remaining_days)) if all_remaining_days else 1.0
        self.rem_denom = (self.rem_max - self.rem_min) + 1e-8

        print(f"[Δt stats] min={self.dt_min:.4f} min | max={self.dt_max:.4f} min")
        print(f"[Remaining days stats] min={self.rem_min:.4f} | max={self.rem_max:.4f}")

        # --- 샘플 구축 (split 대상만) ---
        samples = []
        for cdir in tqdm(case_dirs_split, desc=f"Building samples({split})"):
            nodes = pd.read_csv(cdir / "nodes.csv")
            edges = pd.read_csv(cdir / "edges.csv")
            if len(nodes) == 0: continue
            ts = nodes["timestamp_epoch"].astype(int).to_numpy()
            acts = nodes["activity"].astype(str).to_numpy()
            t_end = int(ts[-1])

            for i in range(len(edges)):
                u = int(edges.loc[i, "src"]); v = int(edges.loc[i, "dst"])
                t_u, t_v = int(ts[u]), int(ts[v])
                if t_v <= t_u: continue

                start = max(0, u - self.L)
                context_ids = list(range(start, u))
                context_ids = ([-1] * (self.L - len(context_ids))) + context_ids

                raw_dt_min = (t_v - t_u) / 60.0
                dt_norm = (raw_dt_min - self.dt_min) / self.dt_denom

                rem_days = max(t_end - t_u, 0) / 86400.0
                rem_norm = (rem_days - self.rem_min) / self.rem_denom

                samples.append({
                    "case_dir": str(cdir),
                    "u": u, "v": v,
                    "context_ids": context_ids,
                    "t_u": t_u,
                    "y_activity": self.activity2idx[acts[v]],
                    "y_dt_norm": float(dt_norm),
                    "y_rem_norm": float(rem_norm),
                })

        self.samples = samples
        self.feature_dim = int(self.meta["feature_dim"])
        self.num_classes = len(self.idx2activity)

        # cache
        self._cache_case = None
        self._cache_feats = None
        self._cache_ts = None
        self._cache_act_idx = None

    def __len__(self): return len(self.samples)

    def _ensure_case(self, case_dir: str):
        if self._cache_case == case_dir: return
        feats = np.load(Path(case_dir) / "node_features.npy")
        nodes = pd.read_csv(Path(case_dir) / "nodes.csv")
        self._cache_case = case_dir
        self._cache_feats = feats
        self._cache_ts = nodes["timestamp_epoch"].astype(int).to_numpy()
        self._cache_act_idx = np.array([self.activity2idx[a] for a in nodes["activity"].astype(str)], dtype=np.int64)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        self._ensure_case(s["case_dir"])
        u = s["u"]; t_u = s["t_u"]; ctx_ids = s["context_ids"]

        x_u = self._cache_feats[u]
        x_ctx = np.stack([
            self._cache_feats[c] if c >= 0 else np.zeros_like(self._cache_feats[0])
            for c in ctx_ids
        ], 0)
        t_ctx = np.array([(t_u - self._cache_ts[c]) if c >= 0 else 0 for c in ctx_ids], dtype=np.float32)

        act_u = self._cache_act_idx[u] + 1
        act_ctx = np.array([(self._cache_act_idx[c] + 1) if c >= 0 else 0 for c in ctx_ids], dtype=np.int64)

        return (
            torch.from_numpy(x_u).float(),             # (D,)
            torch.from_numpy(x_ctx).float(),           # (L,D)
            torch.from_numpy(t_ctx).float(),           # (L,)
            torch.tensor([s["y_dt_norm"]]).float(),    # (1,)
            torch.tensor(s["y_activity"]).long(),      # ()
            torch.tensor(act_u).long(),                # ()
            torch.from_numpy(act_ctx).long(),          # (L,)
            torch.tensor([s["y_rem_norm"]]).float(),   # (1,)
        )

# ================= Time Encoding / TGAT =================
class TimeEncoding(nn.Module):
    def __init__(self, out_dim: int, w_init: float = 1.0):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(out_dim) * w_init)
        self.phase = nn.Parameter(torch.zeros(out_dim))
    def forward(self, delta_t: torch.Tensor):  # (B,L)
        z = torch.log1p(delta_t.clamp_min(0.0))
        z = z.unsqueeze(-1) * self.freq + self.phase
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)  # (B,L,2T)

class TemporalAttention(nn.Module):
    """Multi-head temporal attention (scaled dot-product)."""
    def __init__(self, in_dim: int, time_dim: int, hidden: int, num_heads: int=4, dropout: float=0.2):
        super().__init__()
        assert hidden % num_heads == 0, "hidden must be divisible by heads"
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.q = nn.Linear(in_dim, hidden, bias=False)
        self.k = nn.Linear(in_dim + 2*time_dim, hidden, bias=False)
        self.v = nn.Linear(in_dim + 2*time_dim, hidden, bias=False)
        self.drop  = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    def forward(self, x_u, x_ctx, te_ctx):
        B, L, _ = x_ctx.size()
        q  = self.q(x_u).view(B, self.num_heads, 1, self.head_dim)
        kv = torch.cat([x_ctx, te_ctx], dim=-1)
        k  = self.k(kv).view(B, self.num_heads, L, self.head_dim)
        v  = self.v(kv).view(B, self.num_heads, L, self.head_dim)
        att = (q * k).sum(-1) / self.scale
        att = torch.softmax(att, dim=-1)
        att = self.drop(att)
        out = (att.unsqueeze(-1) * v).sum(2).reshape(B, -1)
        return out, att

class TGATBlock(nn.Module):
    def __init__(self, in_dim: int, time_dim: int, hidden: int, num_heads: int=4, dropout: float=0.2):
        super().__init__()
        self.time_enc = TimeEncoding(time_dim)
        self.attn     = TemporalAttention(in_dim, time_dim, hidden, num_heads=num_heads, dropout=dropout)
        self.res      = nn.Linear(in_dim, hidden)
        self.ff       = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden)
        )
        self.n1 = nn.LayerNorm(hidden)
        self.n2 = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
    def forward(self, x_u, x_ctx, t_ctx):
        te  = self.time_enc(t_ctx)
        agg, _ = self.attn(x_u, x_ctx, te)
        z = self.n1(agg + self.res(x_u))
        z = self.n2(self.ff(self.drop(z)) + z)
        return z

class TGATPredictorMT(nn.Module):
    """
    Heads:
      - cls_head: next activity
      - dt_head : next-Δt (normalized [0,1])
      - rem_head: remaining-to-end days (normalized [0,1])
    """
    def __init__(self, in_dim: int, num_classes: int,
                 act_emb_dim: int = 32, time_dim: int = 8,
                 hidden: int = 255, num_layers: int = 3, num_heads: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        if hidden % num_heads != 0:
            nh = (hidden // num_heads) * num_heads or num_heads
            print(f"[WARN] hidden {hidden} adjusted to {nh} for heads={num_heads}")
            hidden = nh
        self.num_layers = num_layers
        self.hidden = hidden
        self.act_emb = nn.Embedding(num_classes + 1, act_emb_dim, padding_idx=0)

        enc_in0 = in_dim + act_emb_dim
        self.enc0   = TGATBlock(enc_in0, time_dim, hidden, num_heads=num_heads, dropout=dropout)
        self.ctx_p0 = nn.Linear(enc_in0, hidden)
        self.encs   = nn.ModuleList([
            TGATBlock(hidden, time_dim, hidden, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers - 1)
        ])
        self.ctx_ph = nn.Linear(hidden, hidden)

        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.cls_head = nn.Linear(hidden, num_classes)
        self.dt_head  = nn.Linear(hidden, 1)
        self.rem_head = nn.Linear(hidden, 1)

    def forward(self, x_u, x_ctx, t_ctx, act_u_idx, act_ctx_idx):
        # 안전 캐스팅 (DataParallel/loader 혼선 대비)
        act_u_idx = act_u_idx.long()
        act_ctx_idx = act_ctx_idx.long()

        B, L, _ = x_ctx.size()
        e_u   = self.act_emb(act_u_idx)           # (B,E)
        e_ctx = self.act_emb(act_ctx_idx)         # (B,L,E)
        x_u0  = torch.cat([x_u, e_u], dim=-1)
        x_c0  = torch.cat([x_ctx, e_ctx], dim=-1)

        z = self.enc0(x_u0, x_c0, t_ctx)
        ctx = self.ctx_p0(x_c0)
        z = self.encs[0](z, ctx, t_ctx) if len(self.encs) > 0 else z
        for enc in self.encs[1:]:
            ctx = self.ctx_ph(z).unsqueeze(1).expand(-1, L, -1)
            z = enc(z, ctx, t_ctx)

        h = self.mlp(z)
        logits = self.cls_head(h)
        dt_hat = torch.sigmoid(self.dt_head(h))
        rem_hat = torch.sigmoid(self.rem_head(h))
        return logits, dt_hat, rem_hat

# ================= Collate =================
def collate(batch):
    x_u, x_ctx, t_ctx, y_dt_norm, y_cls, act_u, act_ctx, y_rem_norm = zip(*batch)
    return (torch.stack(x_u,0), torch.stack(x_ctx,0), torch.stack(t_ctx,0),
            torch.stack(y_dt_norm,0), torch.stack(y_cls,0),
            torch.stack(act_u,0), torch.stack(act_ctx,0),
            torch.stack(y_rem_norm,0))

# ================= Train/Eval =================
def _rmse(values_diff: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values_diff))))

def train_one_epoch(model, loader, opt, device, ce_crit, reg_crit,
                    lambda_dt=1.0, lambda_rem=1.0, clip=1.0, use_amp=True):
    model.train()
    sum_total = sum_ce = sum_dt = sum_rem = 0.0
    n = 0
    timer = EpochTimer(device); timer.start()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type=="cuda"))
    amp_dtype = torch.bfloat16 if (device.type=="cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    opt.zero_grad(set_to_none=True)
    for batch in tqdm(loader, desc="Training", leave=False):
        # I/O
        ti0 = time.perf_counter()
        x_u, x_ctx, t_ctx, y_dt, y_cls, act_u, act_ctx, y_rem = batch
        x_u   = x_u.to(device, non_blocking=True)
        x_ctx = x_ctx.to(device, non_blocking=True)
        t_ctx = t_ctx.to(device, non_blocking=True)
        y_cls = y_cls.to(device, non_blocking=True)
        y_dt  = y_dt.to(device, non_blocking=True)
        y_rem = y_rem.to(device, non_blocking=True)
        act_u   = act_u.to(device, non_blocking=True)
        act_ctx = act_ctx.to(device, non_blocking=True)
        _cuda_sync(); timer.add_io(time.perf_counter() - ti0)

        # FORWARD
        if device.type == "cuda":
            e0, e1 = timer._evt(); e0.record()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(use_amp and device.type=="cuda")):
            logits, dt_hat, rem_hat = model(x_u, x_ctx, t_ctx, act_u.long(), act_ctx.long())
            loss_ce  = ce_crit(logits, y_cls)
            loss_dt  = reg_crit(dt_hat, y_dt)
            loss_rem = reg_crit(rem_hat, y_rem)
            loss = loss_ce + lambda_dt*loss_dt + lambda_rem*loss_rem
        if device.type == "cuda":
            e1.record(); _cuda_sync(); timer.fwd += e0.elapsed_time(e1) / 1000.0

        # BACKWARD
        if device.type == "cuda":
            e0, e1 = timer._evt(); e0.record()
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if device.type == "cuda":
            e1.record(); _cuda_sync(); timer.bwd += e0.elapsed_time(e1) / 1000.0

        # OPT
        if device.type == "cuda":
            e0, e1 = timer._evt(); e0.record()
        if clip: nn.utils.clip_grad_norm_(model.parameters(), clip)
        if scaler.is_enabled():
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        else:
            opt.step(); opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            e1.record(); _cuda_sync(); timer.opt += e0.elapsed_time(e1) / 1000.0

        # metrics
        bs = x_u.size(0)
        sum_total += loss.item() * bs
        sum_ce    += loss_ce.item() * bs
        sum_dt    += loss_dt.item() * bs
        sum_rem   += loss_rem.item() * bs
        n += bs
        timer.step_done()

    wall = timer.end()
    timer.summary(wall)
    return {
        "loss":    sum_total / max(n,1),
        "loss_ce": sum_ce / max(n,1),
        "loss_dt": sum_dt / max(n,1),
        "loss_rem":sum_rem / max(n,1),
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

    dt_abs_err_min = []   # minutes
    rem_abs_err_d  = []   # days
    dt_val_list, dt_true_list = [], []
    rem_val_list, rem_true_list = [], []

    for x_u, x_ctx, t_ctx, y_dt, y_cls, act_u, act_ctx, y_rem in tqdm(loader, desc=desc, leave=False):
        x_u, x_ctx, t_ctx = x_u.to(device), x_ctx.to(device), t_ctx.to(device)
        y_cls = y_cls.to(device); y_dt = y_dt.to(device); y_rem = y_rem.to(device)
        act_u, act_ctx = act_u.to(device), act_ctx.to(device)

        logits, dt_hat, rem_hat = model(x_u, x_ctx, t_ctx, act_u.long(), act_ctx.long())

        loss_ce  = ce_crit(logits, y_cls)
        loss_dt  = reg_crit(dt_hat, y_dt)
        loss_rem = reg_crit(rem_hat, y_rem)
        loss = loss_ce + lambda_dt*loss_dt + lambda_rem*loss_rem

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

        # 역정규화 후 실제 단위로 MAE/RMSE
        y_dt_pred  = (dt_hat.detach().cpu().numpy().ravel()  * dt_denom  + dt_min)   # minutes
        y_dt_true  = (y_dt.detach().cpu().numpy().ravel()    * dt_denom  + dt_min)
        y_rem_pred = (rem_hat.detach().cpu().numpy().ravel() * rem_denom + rem_min)  # days
        y_rem_true = (y_rem.detach().cpu().numpy().ravel()   * rem_denom + rem_min)

        dt_abs_err_min.append(np.abs(y_dt_pred - y_dt_true))
        rem_abs_err_d.append(np.abs(y_rem_pred - y_rem_true))
        dt_val_list.append(y_dt_pred); dt_true_list.append(y_dt_true)
        rem_val_list.append(y_rem_pred); rem_true_list.append(y_rem_true)

    from sklearn.metrics import f1_score
    acc = correct1 / max(n,1); top3 = correct3 / max(n,1)
    f1m = f1_score(all_true, all_pred, average="macro") if n else float("nan")

    dt_vals  = np.concatenate(dt_val_list)  if dt_val_list  else np.array([])
    dt_trues = np.concatenate(dt_true_list) if dt_true_list else np.array([])
    rem_vals  = np.concatenate(rem_val_list)  if rem_val_list  else np.array([])
    rem_trues = np.concatenate(rem_true_list) if rem_true_list else np.array([])

    dt_mae_min  = float(np.concatenate(dt_abs_err_min).mean()) if dt_abs_err_min else float("nan")
    dt_rmse_min = _rmse(dt_vals - dt_trues) if dt_vals.size else float("nan")
    rem_mae_d   = float(np.concatenate(rem_abs_err_d).mean()) if rem_abs_err_d else float("nan")
    rem_rmse_d  = _rmse(rem_vals - rem_trues) if rem_vals.size else float("nan")

    return {
        "loss":     (sum_total / max(n,1)),
        "loss_ce":  (sum_ce / max(n,1)),
        "loss_dt":  (sum_dt / max(n,1)),
        "loss_rem": (sum_rem / max(n,1)),
        "acc":      acc, "top3": top3, "f1_macro": f1m,
        "mae_dt_minutes": dt_mae_min, "rmse_dt_minutes": dt_rmse_min,
        "mae_rem_days": rem_mae_d, "rmse_rem_days": rem_rmse_d,
    }

# ================= main =================
def main():
    set_seed(42)
    # root = "./tgn_input_2016_cases_SessionID_2"
    root = "./tgn_input_2019_cases_cID"
    # root = "./tgn_input_2014_cases_Interaction"
    # root = "./tgn_input_af_per_case"

    L = 20
    bs = 256
    epochs = 50
    lr = 1e-3
    dropout = 0.2
    hidden = 255
    num_layers = 3
    num_heads = 3
    weight_decay = 0.0
    lambda_dt = 1.0
    lambda_rem = 1.0
    use_amp = True  # AMP 사용 여부

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset / Loader ---
    train_ds = TemporalGraphDataset(root, L=L, split="train")
    val_ds   = TemporalGraphDataset(root, L=L, split="val")
    test_ds  = TemporalGraphDataset(root, L=L, split="test")

    def make_ld(ds, batch, shuffle):
        nw = max(0, os.cpu_count()//2)
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, collate_fn=collate,
                          num_workers=nw, pin_memory=True,
                          persistent_workers=(nw>0), prefetch_factor=2 if nw>0 else None)

    train_ld = make_ld(train_ds, bs, True)
    val_ld   = make_ld(val_ds,   bs, False)
    test_ld  = make_ld(test_ds,  bs, False)

    # --- Model ---
    model = TGATPredictorMT(
        in_dim=train_ds.feature_dim,
        num_classes=train_ds.num_classes,
        act_emb_dim=32, time_dim=8,
        hidden=hidden, num_layers=num_layers, num_heads=num_heads, dropout=dropout
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # --- Optimizer / Loss ---
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce_crit  = nn.CrossEntropyLoss()
    reg_crit = nn.L1Loss()

    # --- Train ---
    n_params = count_params(model)
    print(f"[MODEL] trainable params = {n_params:,}")
    total_train_sec = 0.0
    best = 0.0
    BEST_PATH, LAST_PATH = "tgat_mt_best.pt", "tgat_mt_last.pt"

    for ep in tqdm(range(1, epochs+1), desc="Epochs"):
        _cuda_sync(); t_ep0 = time.perf_counter()
        tr = train_one_epoch(model, train_ld, opt, device, ce_crit, reg_crit,
                             lambda_dt=lambda_dt, lambda_rem=lambda_rem, clip=1.0, use_amp=use_amp)
        _cuda_sync(); t_ep = time.perf_counter() - t_ep0
        total_train_sec += t_ep

        va = evaluate(model, val_ld, device, ce_crit, reg_crit,
                      lambda_dt=lambda_dt, lambda_rem=lambda_rem, desc="Val")

        print(f"[TIME] epoch {ep:03d}: train {t_ep:.2f}s | cum {total_train_sec:.2f}s\n")
        print(f"[EP{ep:03d}] "
              f"train_total={tr['loss']:.4f} (CE={tr['loss_ce']:.4f}, dt={tr['loss_dt']:.4f}, rem={tr['loss_rem']:.4f}) | "
              f"val_total={va['loss']:.4f} (CE={va['loss_ce']:.4f}, dt={va['loss_dt']:.4f}, rem={va['loss_rem']:.4f}) | "
              f"val_acc={va['acc']:.4f} | val_top3={va['top3']:.4f} | f1M={va['f1_macro']:.4f} | "
              f"MAE(Δt_min)={va['mae_dt_minutes']:.2f} RMSE(Δt_min)={va['rmse_dt_minutes']:.2f} | "
              f"MAE(RT_days)={va['mae_rem_days']:.2f} RMSE(RT_days)={va['rmse_rem_days']:.2f}")

        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        meta = {"activity_vocab": train_ds.idx2activity, "feature_dim": train_ds.feature_dim, "L": L}
        if va["acc"] > best:
            best = va["acc"]
            torch.save({"model": state, "meta": meta}, BEST_PATH)
        torch.save({"model": state, "meta": meta}, LAST_PATH)

    # --- Test ---
    load_path = BEST_PATH if os.path.exists(BEST_PATH) else LAST_PATH
    ckpt = torch.load(load_path, map_location=device)
    if isinstance(model, nn.DataParallel): model.module.load_state_dict(ckpt["model"])
    else: model.load_state_dict(ckpt["model"])

    te = evaluate(model, test_ld, device, ce_crit, reg_crit,
                  lambda_dt=lambda_dt, lambda_rem=lambda_rem, desc="Test")
    print(f"[TEST] total={te['loss']:.4f} (CE={te['loss_ce']:.4f}, dt={te['loss_dt']:.4f}, rem={te['loss_rem']:.4f}) | "
          f"acc={te['acc']:.4f} | top3={te['top3']:.4f} | f1M={te['f1_macro']:.4f} | "
          f"MAE(Δt_min)={te['mae_dt_minutes']:.2f} RMSE(Δt_min)={te['rmse_dt_minutes']:.2f} | "
          f"MAE(RT_days)={te['mae_rem_days']:.2f} RMSE(RT_days)={te['rmse_rem_days']:.2f}")

    print("-"*80)
    print("\n[Benchmark] Inference speed on TEST loader (forward-only)")
    bench = benchmark_inference(model, test_ld, device, max_batches=None, desc="Inference")
    print(f" - examples: {bench['examples']}")
    print(f" - time: {bench['time_s']:.3f}s")
    print(f" - throughput: {bench['throughput_ex_s']:.2f} ex/s")
    print(f" - latency: {bench['latency_ms_per_ex']:.3f} ms/example")

if __name__ == "__main__":
    main()
