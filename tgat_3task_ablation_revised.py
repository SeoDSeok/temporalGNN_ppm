# TGAT 3-task (next-activity, next-Δt, remaining time) + Ablation Suite

import os, json, glob, random, time, math
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------- utils -----------------
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ----------------- Dataset (3-task: cls + Δt + remaining) -----------------
class TemporalGraphDataset(Dataset):
    """
    각 케이스의 연속 엣지 (u->v)를 하나의 샘플로 사용.
    반환: (x_u, x_ctx, t_ctx, y_dt_norm, y_cls, act_u, act_ctx, y_rem_norm)
      - x_u:        (D,)
      - x_ctx:      (L,D)
      - t_ctx:      (L,)   (현재 u와 과거 context 간 경과시간 [sec], raw)
      - y_dt_norm:  (1,)   다음 이벤트까지 Δt(분) 정규화 [0,1]
      - y_cls:      ()     다음 activity id
      - act_u:      ()     현재 액티비티 id+1 (pad=0)
      - act_ctx:    (L,)   컨텍스트 액티비티 id+1 (pad=0)
      - y_rem_norm: (1,)   케이스 종료까지 남은 일수 정규화 [0,1]
    """
    def __init__(self, root_dir: str, L: int = 20, split: str = "train",
                 split_ratio=(0.8, 0.1, 0.1), seed=42):
        super().__init__()
        self.root = Path(root_dir)
        self.L = L
        self.meta = json.load(open(self.root / "metadata.json", "r", encoding="utf-8"))

        # 1) 모든 케이스 나열 + split
        all_case_dirs = sorted(
            [Path(p) for p in glob.glob(str(self.root / "case_*")) if Path(p).is_dir()]
        )
        split_file = self.root / "case_split.json"
        if split_file.exists():
            sp = json.load(open(split_file, "r", encoding="utf-8"))
        else:
            names = [d.name for d in all_case_dirs]
            rng = random.Random(seed); rng.shuffle(names)
            n = len(names); n_train = int(n * split_ratio[0]); n_val = int(n * split_ratio[1])
            sp = {
                "train": names[:n_train],
                "val":   names[n_train:n_train+n_val],
                "test":  names[n_train+n_val:],
            }
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

        # 3) Δt(분), 잔여일(일) 전역 통계
        all_dt_min = []
        all_rem_days = []
        for cdir in all_case_dirs:
            nodes = pd.read_csv(cdir / "nodes.csv")
            edges = pd.read_csv(cdir / "edges.csv")
            if len(nodes) == 0:
                continue
            ts = nodes["timestamp_epoch"].astype(int).to_numpy()
            case_end = int(ts[-1])

            for i in range(len(edges)):
                u = int(edges.loc[i, "src"]); v = int(edges.loc[i, "dst"])
                t_u, t_v = int(ts[u]), int(ts[v])
                if t_v > t_u:
                    all_dt_min.append((t_v - t_u) / 60.0)
                if case_end > t_u:
                    all_rem_days.append((case_end - t_u) / 86400.0)  # seconds -> days

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
            if len(nodes) == 0:
                continue

            feats = np.load(cdir / "node_features.npy")
            ts = nodes["timestamp_epoch"].astype(int).to_numpy()
            acts = nodes["activity"].astype(str).to_numpy()
            case_end = int(ts[-1])

            for i in range(len(edges)):
                u = int(edges.loc[i, "src"]); v = int(edges.loc[i, "dst"])
                t_u, t_v = int(ts[u]), int(ts[v])
                if t_v <= t_u:
                    continue

                start = max(0, u - self.L)
                ctx_ids = list(range(start, u))
                ctx_ids = ([-1] * (self.L - len(ctx_ids))) + ctx_ids

                # Δt(min) 정규화
                raw_dt_min = (t_v - t_u) / 60.0
                y_dt_norm = (raw_dt_min - self.dt_min) / self.dt_denom

                # 남은 시간(일) 정규화
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

    def __len__(self):
        return len(self.samples)

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
            for c in ctx_ids
        ], axis=0)
        t_ctx = np.array(
            [(t_u - self._cache_ts[c]) if c >= 0 else 0 for c in ctx_ids],
            dtype=np.float32
        )

        act_u = self._cache_act_idx[u] + 1
        act_ctx = np.array(
            [(self._cache_act_idx[c] + 1) if c >= 0 else 0 for c in ctx_ids],
            dtype=np.int64
        )

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


# ----------------- Temporal Attention (ablation-aware) -----------------
class TemporalAttentionAblation(nn.Module):
    """Multi-head temporal attention with optional time encoding."""
    def __init__(self, in_dim: int, time_dim: int, hidden: int,
                 num_heads: int = 4, dropout: float = 0.2,
                 use_time: bool = True):
        super().__init__()
        assert hidden % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_time = use_time

        # query: x_u (in_dim)
        self.q_proj = nn.Linear(in_dim, hidden, bias=False)

        kv_in_dim = in_dim + (2 * time_dim if use_time else 0)
        self.k_proj = nn.Linear(kv_in_dim, hidden, bias=False)
        self.v_proj = nn.Linear(kv_in_dim, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_u, x_ctx, te_ctx, mask):
        """
        x_u:   (B, D_in)
        x_ctx: (B, L, D_in)
        te_ctx:(B, L, 2T) or None
        mask:  (B, L)  (1 for valid, 0 for pad)
        """
        B, L, _ = x_ctx.size()
        H = self.num_heads
        Dh = self.head_dim

        q = self.q_proj(x_u).view(B, H, 1, Dh)  # (B,H,1,Dh)

        if self.use_time and te_ctx is not None:
            kv = torch.cat([x_ctx, te_ctx], dim=-1)  # (B,L,D_in+2T)
        else:
            kv = x_ctx  # (B,L,D_in)

        k = self.k_proj(kv).view(B, H, L, Dh)
        v = self.v_proj(kv).view(B, H, L, Dh)

        att_logits = (q * k).sum(-1) / self.scale  # (B,H,L)

        if mask is not None:
            att_logits = att_logits.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        att = torch.softmax(att_logits, dim=-1)    # (B,H,L)
        att = self.dropout(att)

        out = (att.unsqueeze(-1) * v).sum(2)       # (B,H,Dh)
        out = out.reshape(B, -1)                   # (B, hidden)
        return out


class TGATBlockAblation(nn.Module):
    def __init__(self, hidden: int, time_dim: int,
                 num_heads: int = 4, dropout: float = 0.2,
                 use_time: bool = True,
                 use_context: bool = True):
        super().__init__()
        self.use_context = use_context
        self.use_time = use_time

        self.time_enc = TimeEncoding(time_dim) if use_time else None
        self.attn = TemporalAttentionAblation(
            in_dim=hidden,
            time_dim=time_dim,
            hidden=hidden,
            num_heads=num_heads,
            dropout=dropout,
            use_time=use_time,
        )
        self.res = nn.Linear(hidden, hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden)
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, ctx, t_ctx, mask):
        """
        h:    (B, hidden)
        ctx:  (B, L, hidden)
        t_ctx:(B, L)
        mask: (B, L)
        """
        if not self.use_context:
            # context를 아예 안 쓰는 ablation: self-attention 없이 FFN + residual만
            z = self.norm1(self.res(h))
            z2 = self.ff(self.dropout(z))
            z = self.norm2(z + z2)
            return z

        if self.use_time and self.time_enc is not None:
            te = self.time_enc(t_ctx)   # (B,L,2T)
        else:
            te = None

        agg = self.attn(h, ctx, te, mask)         # (B,hidden)
        z = self.norm1(agg + self.res(h))
        z2 = self.ff(self.dropout(z))
        z = self.norm2(z + z2)
        return z


# ----------------- TGAT Predictor (3 heads) + Ablations -----------------
class TGATPredictorAblation(nn.Module):
    """
    TGAT-style 멀티태스크 (3-head):
      - cls_head:      next activity
      - reg_next_head: next Δt (normalized [0,1])
      - reg_rem_head:  remaining time (days, normalized [0,1])

    ablation_config:
      - use_time_enc:   bool
      - use_context:    bool
      - use_act_emb:    bool
      - only_act_emb:   bool
    """
    def __init__(self,
                 in_dim: int,
                 num_classes: int,
                 ablation_config: Dict,
                 act_emb_dim: int = 32,
                 time_dim: int = 8,
                 hidden: int = 256,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 mlp_hidden: int = 256,
                 dropout: float = 0.2):
        super().__init__()

        cfg = {
            "use_time_enc": True,
            "use_context": True,
            "use_act_emb": True,
            "only_act_emb": False,
        }
        cfg.update(ablation_config or {})
        self.cfg = cfg

        if hidden % num_heads != 0:
            hidden = (hidden // num_heads) * num_heads or num_heads

        self.num_classes = num_classes
        self.hidden = hidden
        self.num_layers = num_layers
        self.time_dim = time_dim

        self.act_emb_dim = act_emb_dim
        self.act_emb = nn.Embedding(num_classes + 1, act_emb_dim, padding_idx=0)

        # input_pre_dim 결정
        D = in_dim
        E = act_emb_dim

        if cfg["only_act_emb"]:
            input_pre_dim = E
        elif not cfg["use_act_emb"]:
            input_pre_dim = D
        else:
            input_pre_dim = D + E

        # 원래 feature(+act_emb)를 hidden으로 projection
        self.in_proj_self = nn.Linear(input_pre_dim, hidden)
        self.in_proj_ctx  = nn.Linear(input_pre_dim, hidden)

        # TGAT blocks (모두 in_dim = hidden)
        self.blocks = nn.ModuleList([
            TGATBlockAblation(
                hidden=hidden,
                time_dim=time_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_time=cfg["use_time_enc"],
                use_context=cfg["use_context"],
            )
            for _ in range(num_layers)
        ])

        self.readout = nn.Sequential(
            nn.Linear(hidden, mlp_hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.cls_head      = nn.Linear(mlp_hidden, num_classes)
        self.reg_next_head = nn.Linear(mlp_hidden, 1)
        self.reg_rem_head  = nn.Linear(mlp_hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_input(self, x_u, x_ctx, act_u_idx, act_ctx_idx):
        """
        ablation config에 따라 x_u / x_ctx / act_emb를 조합해서
        projection 이전 입력을 만든다.
        """
        B, L, D = x_ctx.shape
        device = x_u.device

        if self.cfg["use_act_emb"] or self.cfg["only_act_emb"]:
            e_u   = self.act_emb(act_u_idx)        # (B,E)
            e_ctx = self.act_emb(act_ctx_idx)      # (B,L,E)
        else:
            # 활동 임베딩 안 쓸 때는 0으로 채워진 텐서 사용
            e_u   = torch.zeros(B, self.act_emb_dim, device=device)
            e_ctx = torch.zeros(B, L, self.act_emb_dim, device=device)

        if self.cfg["only_act_emb"]:
            x_u_cat  = e_u                  # (B,E)
            x_ctx_cat= e_ctx                # (B,L,E)
        elif not self.cfg["use_act_emb"]:
            x_u_cat  = x_u                  # (B,D)
            x_ctx_cat= x_ctx                # (B,L,D)
        else:
            x_u_cat  = torch.cat([x_u, e_u], dim=-1)       # (B,D+E)
            x_ctx_cat= torch.cat([x_ctx, e_ctx], dim=-1)   # (B,L,D+E)

        return x_u_cat, x_ctx_cat

    def forward(self, x_u, x_ctx, t_ctx, act_u_idx, act_ctx_idx):
        """
        x_u: (B,D), x_ctx: (B,L,D), t_ctx: (B,L), act_u_idx: (B,), act_ctx_idx: (B,L)
        """
        B, L, _ = x_ctx.shape
        mask = (act_ctx_idx > 0).float() if act_ctx_idx is not None else torch.ones(B, L, device=x_u.device)

        x_u_cat, x_ctx_cat = self._build_input(x_u, x_ctx, act_u_idx, act_ctx_idx)

        h   = self.in_proj_self(x_u_cat)        # (B,hidden)
        ctx = self.in_proj_ctx(x_ctx_cat)       # (B,L,hidden)

        for blk in self.blocks:
            h = blk(h, ctx, t_ctx, mask)

        z = self.readout(h)
        logits  = self.cls_head(z)
        dt_hat  = torch.sigmoid(self.reg_next_head(z))  # [0,1]
        rem_hat = torch.sigmoid(self.reg_rem_head(z))   # [0,1]
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


def train_one_epoch(model, loader, opt, device, ce_crit, reg_crit,
                    lambda_dt=1.0, lambda_rem=1.0, clip=1.0):
    model.train()
    sum_total = sum_ce = sum_dt = sum_rem = 0.0
    n = 0
    opt.zero_grad(set_to_none=True)

    for x_u, x_ctx, t_ctx, y_dt, y_cls, act_u, act_ctx, y_rem in tqdm(loader, desc="Training", leave=False):
        x_u, x_ctx, t_ctx = x_u.to(device), x_ctx.to(device), t_ctx.to(device)
        y_dt, y_cls, y_rem = y_dt.to(device), y_cls.to(device), y_rem.to(device)
        act_u, act_ctx = act_u.to(device), act_ctx.to(device)

        logits, y_hat_dt, y_hat_rem = model(x_u, x_ctx, t_ctx, act_u, act_ctx)

        loss_ce  = ce_crit(logits, y_cls)
        loss_dt  = reg_crit(y_hat_dt,  y_dt)
        loss_rem = reg_crit(y_hat_rem, y_rem)
        loss = loss_ce + lambda_dt * loss_dt + lambda_rem * loss_rem

        loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step(); opt.zero_grad(set_to_none=True)

        bs = x_u.size(0)
        sum_total += loss.item() * bs
        sum_ce    += loss_ce.item() * bs
        sum_dt    += loss_dt.item() * bs
        sum_rem   += loss_rem.item() * bs
        n += bs

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

    mae_dt_min = []   # 분 단위
    mae_rem_d  = []   # 일 단위

    sum_abs_dt_norm = 0.0
    sum_abs_rem_norm = 0.0

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

        # 정규화 MAE(0~1)
        sum_abs_dt_norm  += torch.abs(y_hat_dt - y_dt).sum().item()
        sum_abs_rem_norm += torch.abs(y_hat_rem - y_rem).sum().item()

        # 역정규화 MAE(분/일)
        y_dt_pred = (y_hat_dt.detach().cpu().numpy().ravel() * dt_denom + dt_min)
        y_dt_true = (y_dt.detach().cpu().numpy().ravel()      * dt_denom + dt_min)
        y_rem_pred= (y_hat_rem.detach().cpu().numpy().ravel() * rem_denom + rem_min)
        y_rem_true= (y_rem.detach().cpu().numpy().ravel()     * rem_denom + rem_min)
        mae_dt_min.append(np.abs(y_dt_pred - y_dt_true))
        mae_rem_d.append(np.abs(y_rem_pred - y_rem_true))

    from sklearn.metrics import f1_score
    dt_mae_min = float(np.concatenate(mae_dt_min).mean()) if mae_dt_min else float("nan")
    rem_mae_d  = float(np.concatenate(mae_rem_d).mean()) if mae_rem_d else float("nan")

    mae_dt_norm  = sum_abs_dt_norm  / n if n > 0 else float("nan")
    mae_rem_norm = sum_abs_rem_norm / n if n > 0 else float("nan")

    return {
        "loss": sum_total / n, "loss_ce": sum_ce / n, "loss_dt": sum_dt / n, "loss_rem": sum_rem / n,
        "acc": correct1 / n, "top3": correct3 / n, "f1_macro": f1_score(all_true, all_pred, average="macro"),
        "mae_dt_minutes": dt_mae_min, "mae_rem_days": rem_mae_d,
        "mae_dt_norm": mae_dt_norm, "mae_rem_norm": mae_rem_norm,
    }


# ----------------- Ablation config -----------------
def get_ablation_config(mode: str):
    """
    mode에 따라 ablation 설정과 num_heads를 돌려준다.
    """
    base_cfg = {
        "use_time_enc": True,
        "use_context": True,
        "use_act_emb": True,
        "only_act_emb": False,
    }
    num_heads = 4

    if mode == "baseline":
        pass

    elif mode == "no_time_encoding":
        base_cfg["use_time_enc"] = False

    elif mode == "no_context_time":
        # 구현상 attention에서 time을 안 쓰는 것이므로 no_time과 동일하게 처리
        base_cfg["use_time_enc"] = False

    elif mode == "no_context":
        base_cfg["use_context"] = False

    elif mode == "single_head":
        num_heads = 1   # Heads Ablation (1 vs 4)

    elif mode == "no_act_emb":
        base_cfg["use_act_emb"] = False
        base_cfg["only_act_emb"] = False

    elif mode == "only_act_emb":
        base_cfg["use_act_emb"] = True
        base_cfg["only_act_emb"] = True

    else:
        raise ValueError(f"Unknown ablation mode: {mode}")

    return base_cfg, num_heads


# ----------------- run_experiment for ablation -----------------
def run_experiment_ablation(
    mode: str = "baseline",
    seed: int = 42,
    epochs: int = 50,
    root: str = "./tgn_input_2014_cases_Interaction",
    L: int = 20,
    bs: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    dropout: float = 0.2,
    hidden: int = 256,
    num_layers: int = 3,
    mlp_hidden: int = 256,
    lambda_dt: float = 1.0,
    lambda_rem: float = 1.0,
):
    print(f"\n==== Ablation mode: {mode} ====")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ab_cfg, num_heads = get_ablation_config(mode)

    # 1) Dataset / Loader
    _cuda_sync()
    t_prep_start = time.perf_counter()

    train_ds = TemporalGraphDataset(root, L=L, split="train")
    val_ds   = TemporalGraphDataset(root, L=L, split="val")
    test_ds  = TemporalGraphDataset(root, L=L, split="test")

    def make_ld(ds, bs, shuffle):
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle, collate_fn=collate,
            num_workers=max(2, os.cpu_count()//2),
            pin_memory=True, persistent_workers=True, prefetch_factor=2
        )

    train_ld = make_ld(train_ds, bs, True)
    val_ld   = make_ld(val_ds,   bs, False)
    test_ld  = make_ld(test_ds,  bs, False)

    _cuda_sync()
    t_prep_end = time.perf_counter()
    prep_time = t_prep_end - t_prep_start

    # 2) Model / Optimizer
    model = TGATPredictorAblation(
        in_dim=train_ds.feature_dim,
        num_classes=train_ds.num_classes,
        ablation_config=ab_cfg,
        act_emb_dim=32,
        time_dim=8,
        hidden=hidden,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_hidden=mlp_hidden,
        dropout=dropout,
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"[TGAT Ablation] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce_crit  = nn.CrossEntropyLoss()
    reg_crit = nn.L1Loss()

    BEST_PATH = f"tgat_multi_ablation_{mode}_seed{seed}.pt"
    best_acc = 0.0

    # 3) Train + Validation
    _cuda_sync()
    t_train_start = time.perf_counter()

    for ep in range(1, epochs+1):
        tr = train_one_epoch(model, train_ld, opt, device, ce_crit, reg_crit,
                             lambda_dt=lambda_dt, lambda_rem=lambda_rem, clip=1.0)
        va = evaluate(model, val_ld, device, ce_crit, reg_crit,
                      lambda_dt=lambda_dt, lambda_rem=lambda_rem, desc=f"Val[{mode}]")

        print(f"[{mode}][EP{ep:03d}] "
              f"train_total={tr['loss']:.4f} (CE={tr['loss_ce']:.4f}, dt={tr['loss_dt']:.4f}, rem={tr['loss_rem']:.4f}) | "
              f"val_total={va['loss']:.4f} (CE={va['loss_ce']:.4f}, dt={va['loss_dt']:.4f}, rem={va['loss_rem']:.4f}) | "
              f"val_acc={va['acc']:.4f} | val_top3={va['top3']:.4f} | f1M={va['f1_macro']:.4f} | "
              f"MAE(dt_min)={va['mae_dt_minutes']:.2f} | MAE(rem_days)={va['mae_rem_days']:.2f}")

        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        if va["acc"] > best_acc:
            best_acc = va["acc"]
            torch.save({
                "model": state,
                "meta": {
                    "activity_vocab": train_ds.idx2activity,
                    "feature_dim": train_ds.feature_dim,
                    "L": L,
                    "dt_min": train_ds.dt_min, "dt_denom": train_ds.dt_denom,
                    "rem_min": train_ds.rem_min, "rem_denom": train_ds.rem_denom,
                    "ablation_mode": mode,
                    "ablation_config": ab_cfg,
                }
            }, BEST_PATH)

    _cuda_sync()
    t_train_end = time.perf_counter()
    train_time = t_train_end - t_train_start

    # 4) Test
    ckpt = torch.load(BEST_PATH, map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])

    _cuda_sync()
    t_test_start = time.perf_counter()

    te = evaluate(model, test_ld, device, ce_crit, reg_crit,
                  lambda_dt=lambda_dt, lambda_rem=lambda_rem, desc=f"Test[{mode}]")

    _cuda_sync()
    t_test_end = time.perf_counter()
    test_time = t_test_end - t_test_start

    print(f"[{mode}][TEST] total={te['loss']:.4f} (CE={te['loss_ce']:.4f}, dt={te['loss_dt']:.4f}, rem={te['loss_rem']:.4f}) | "
          f"acc={te['acc']:.4f} | top3={te['top3']:.4f} | f1M={te['f1_macro']:.4f} | "
          f"MAE(dt_min)={te['mae_dt_minutes']:.2f} | MAE(rem_days)={te['mae_rem_days']:.2f} | "
          f"MAE(dt_norm)={te['mae_dt_norm']:.4f} | MAE(rem_norm)={te['mae_rem_norm']:.4f}")

    result = {
        "mode": mode,
        "seed": seed,
        "prep_time": prep_time,
        "train_time": train_time,
        "test_time": test_time,
        "val_best_acc": best_acc,
        "test_acc": te["acc"],
        "test_dt_MAE_norm": te["mae_dt_norm"],
        "test_rem_MAE_norm": te["mae_rem_norm"],
        "test_dt_MAE_minutes": te["mae_dt_minutes"],
        "test_rem_MAE_days": te["mae_rem_days"],
    }
    return result


def run_ablation_suite():
    root = "./tgn_input_2019_cases_cID"
    # root = "./tgn_input_2014_cases_Interaction"
    # root = "./tgn_input_2016_cases_SessionID_2"
    # root = "./tgn_input_af_per_case"
    modes = [
        "no_time_encoding",   # 1. No Time Encoding
        "no_context_time",    # 2. No Context Time
        "no_context",         # 3. No Context
        "single_head",        # 5. Heads Ablation (1 vs 4)
        "no_act_emb",         # 6. No activity embedding
        "only_act_emb",       # 7. Only Activity Embedding
        "baseline"
    ]

    all_results = []
    for m in modes:
        res = run_experiment_ablation(
            mode=m,
            seed=42,
            epochs=50,
            root=root,
        )
        all_results.append(res)

    print("\n==== Ablation Summary ====")
    for r in all_results:
        print(f"{r['mode']}: test_acc={r['test_acc']:.4f}, "
              f"dt_MAE(min)={r['test_dt_MAE_minutes']:.2f}, "
              f"rem_MAE(days)={r['test_rem_MAE_days']:.2f}")


def main():
    # 단일 모드 테스트용
    # root = "./tgn_input_2019_cases_cID"
    # root = "./tgn_input_2014_cases_Interaction"
    # root = "./tgn_input_2016_cases_SessionID_2"
    root = "./tgn_input_af_per_case"  
    res = run_experiment_ablation(
        mode="baseline",
        seed=42,
        epochs=50,
        root=root,
    )
    print("\n[Single TGAT ablation run result]")
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    # 하나만 먼저 확인해보고 싶으면 main()
    # 전체 ablation 돌리려면 run_ablation_suite() 호출
    run_ablation_suite()
    # main()
