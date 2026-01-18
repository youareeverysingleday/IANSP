
import os, sys, csv, glob, time, math, json, random, bisect, itertools
import gc
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

_PREPROCESS_ONLY = ("--preprocess-only" in sys.argv)

if _PREPROCESS_ONLY:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
    os.environ.setdefault("ROCR_VISIBLE_DEVICES", "")


import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist  
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.optim as optim

from torch.utils.data.distributed import DistributedSampler

from dataclasses import dataclass
from collections import defaultdict


import holidays

from GetInformationFromContext import parse_context

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



def build_user_index(df, user_col="userID", time_col="stime"):

    df = df.copy()

    df[time_col] = pd.to_datetime(df[time_col])

    df_sorted = df.sort_values([user_col, time_col]).reset_index(drop=True)

    gb = df_sorted.groupby(user_col, sort=False).indices
    user2pos = {int(uid): list(idxs) for uid, idxs in gb.items()}

    return df_sorted, user2pos


def split_labels_with_history(user2pos, train_ratio=0.8, seq_len=64):

    train_labels, test_labels = [], []

    for uid, idxs in user2pos.items():
        n = len(idxs)

        if n <= seq_len:
            continue

        cut = max(seq_len, int(n * train_ratio))

        for label_pos in range(seq_len, n):
            row_idx = idxs[label_pos]

            if label_pos < cut:
                train_labels.append(row_idx)
            else:
                test_labels.append(row_idx)
    return train_labels, test_labels

class UserOrderSampler(torch.utils.data.Sampler):

    def __init__(self, user_to_sample_indices, seed=42, shuffle_users=True):
        self.user_to_sample_indices = user_to_sample_indices
        self.users = list(user_to_sample_indices.keys())
        self.seed = seed
        self.shuffle_users = shuffle_users
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        users = self.users.copy()
        if self.shuffle_users:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(users)
        for u in users:
            for idx in self.user_to_sample_indices[u]:
                yield idx

    def __len__(self):
        return sum(len(v) for v in self.user_to_sample_indices.values())
    
class TimeBucketConfig:

    def __init__(self, year_min=2010, year_size=30):
        self.year_min = year_min
        self.year_size = year_size

def to_time_buckets_np(dt64: np.datetime64, cfg: TimeBucketConfig):

    ts = pd.Timestamp(dt64)
    year = ts.year
    year_idx = max(0, min(cfg.year_size - 1, year - cfg.year_min))
    month_idx = ts.month - 1
    quarter_idx = (ts.month - 1) // 3
    day_idx = ts.day - 1
    dow_idx = ts.dayofweek
    hour_idx = ts.hour
    minute_idx = ts.minute
    ampm_idx = 0 if hour_idx < 12 else 1
    cn_holidays = holidays.CN()
    is_weekend = dow_idx >= 5 
    is_official_holiday = ts.date() in cn_holidays
    holiday_idx = 1 if (is_weekend or is_official_holiday) else 0

    return (year_idx, month_idx, quarter_idx, 
            day_idx, dow_idx, hour_idx, ampm_idx, 
            holiday_idx, minute_idx)

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as _mp
import pickle as _pickle

def _geolife_build_one_user(task):
    (
        uid, uidx,
        sub_global_idx,
        grid_ids, dur_vals, stime_vals, ctx_vals,
        grid_id_to_idx, tb_cfg, ctx_col,
        max_seq_len, min_history, stride, label_set,
    ) = task

    n = len(grid_ids)
    if n < min_history + 1:
        return None

    grid_seq = np.fromiter((grid_id_to_idx.get(int(g), 0) for g in grid_ids), dtype=np.int64, count=n)

    dur_idx = np.asarray(dur_vals, dtype=np.int64)

    stime = np.asarray(stime_vals)
    stime_minute = np.zeros(n, np.float32)

    year_idx = np.zeros(n, np.int64)
    month_idx = np.zeros(n, np.int64)
    quarter_idx = np.zeros(n, np.int64)
    day_idx = np.zeros(n, np.int64)
    dow_idx = np.zeros(n, np.int64)
    hour_idx = np.zeros(n, np.int64)
    ampm_idx = np.zeros(n, np.int64)
    holiday_idx = np.zeros(n, np.int64)
    minute_idx = np.zeros(n, np.int64)

    for i in range(n):
        y, mo, q, d, dow, h, ap, hol, mi = to_time_buckets_np(stime[i], tb_cfg)
        year_idx[i] = y; month_idx[i] = mo; quarter_idx[i] = q; day_idx[i] = d
        dow_idx[i] = dow; hour_idx[i] = h; ampm_idx[i] = ap; holiday_idx[i] = hol; minute_idx[i] = mi

    if ctx_col is None or ctx_vals is None:
        ctx_positions = np.zeros(0, dtype=np.int64)
        ctx_start = np.zeros(0, dtype=np.int64)
        ctx_end = np.zeros(0, dtype=np.int64)
        ctx_time = np.zeros(0, dtype=np.float32)
        ctx_sigma = np.zeros(0, dtype=np.float32)
    else:
        ctx_positions, ctx_start, ctx_end, ctx_time, ctx_sigma = [], [], [], [], []
        for i in range(n):
            ctx_str = ctx_vals[i]
            if isinstance(ctx_str, str) and "will move" in ctx_str:
                ref_time = pd.Timestamp(stime[i]).to_pydatetime()
                parsed = parse_context(ctx_str, reference_time=ref_time)
                if parsed is not None:
                    ctx_positions.append(i)
                    ctx_start.append(grid_id_to_idx.get(int(parsed["start_grid"]), 0))
                    ctx_end.append(grid_id_to_idx.get(int(parsed["end_grid"]), 0))
                    ctx_time.append(float(parsed["time_text_minute"]))
                    ctx_sigma.append(float(parsed["sigma_minute"]))

        ctx_positions = np.array(ctx_positions, dtype=np.int64)
        ctx_start = np.array(ctx_start, dtype=np.int64)
        ctx_end = np.array(ctx_end, dtype=np.int64)
        ctx_time = np.array(ctx_time, dtype=np.float32)
        ctx_sigma = np.array(ctx_sigma, dtype=np.float32)

    user_data = {
        "grid": grid_seq,
        "dur": dur_idx,
        "stime_minute": stime_minute,
        "year": year_idx,
        "month": month_idx,
        "quarter": quarter_idx,
        "day": day_idx,
        "dow": dow_idx,
        "hour": hour_idx,
        "ampm": ampm_idx,
        "holiday": holiday_idx,
        "minute": minute_idx,
        "ctx_positions": ctx_positions,
        "ctx_start": ctx_start,
        "ctx_end": ctx_end,
        "ctx_time": ctx_time,
        "ctx_sigma": ctx_sigma,
        "sub_global_idx": np.asarray(sub_global_idx, dtype=np.int64),
    }

    end_positions = []
    for end_pos in range(min_history - 1, n - 1, stride):
        label_pos = end_pos + 1
        if label_pos < max_seq_len:
            continue
        label_row_global = int(sub_global_idx[label_pos])
        if (label_set is not None) and (label_row_global not in label_set):
            continue
        end_positions.append(int(end_pos))

    if len(end_positions) == 0:
        return None

    return (uidx, user_data, end_positions)

def _geolife_cache_save(cache_dir, uidx, payload):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"user_{uidx}.pkl")
    with open(path, "wb") as f:
        _pickle.dump(payload, f, protocol=_pickle.HIGHEST_PROTOCOL)
    return path

def _geolife_cache_load(cache_dir, uidx):
    path = os.path.join(cache_dir, f"user_{uidx}.pkl")
    with open(path, "rb") as f:
        return _pickle.load(f)



class GeoLifeUnifiedDatasetFast(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_indices=None,             
        max_seq_len=64,
        max_ctx_num=16,
        min_history=2,
        stride=1,
        time_bucket_cfg=None,
        user_col="userID",
        time_col="stime",
        grid_col="grid",
        dur_col="duration",
        ctx_col: Optional[str] = "context_fuzzy",
        preprocess_cache_dir: Optional[str] = None,
        preprocess_num_workers: Optional[int] = None,
        preprocess_force_rebuild: bool = False,
        show_progress: bool = False,
        # feature_Gate = {"userID": True, "time": True, "context": "context_fuzzy"}
    ):
        super().__init__()

        if ctx_col is not None and ctx_col not in df.columns:
            raise ValueError(f"ctx_col={ctx_col!r} not found in df.columns. "
                             f"Available ctx columns: {[c for c in df.columns if 'context' in c]}")

        self.max_seq_len = int(max_seq_len)
        self.max_ctx_num = int(max_ctx_num)
        self.min_history = int(min_history)
        self.stride = int(stride)
        self.tb_cfg = time_bucket_cfg or TimeBucketConfig()

        self.label_set = None if label_indices is None else set(int(x) for x in label_indices)

        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[user_col, time_col, grid_col, dur_col]).reset_index(drop=True)

        df[user_col] = df[user_col].astype(int)
        df[grid_col] = df[grid_col].astype(int)
        df[dur_col] = df[dur_col].astype(float)

        df = df.sort_values([user_col, time_col]).reset_index(drop=True)
        self.df = df 
        user_ids = np.sort(df[user_col].unique())
        grid_ids = np.sort(df[grid_col].unique())
        self.user_id_to_idx = {int(u): i for i, u in enumerate(user_ids)}
        self.grid_id_to_idx = {int(g): i for i, g in enumerate(grid_ids)}

        self.idx_to_user_id = {i: int(u) for i, u in enumerate(user_ids)}
        self.idx_to_grid_id = {i: int(g) for i, g in enumerate(grid_ids)}
        self.num_users = len(user_ids)
        self.num_grids = len(grid_ids)

        self.users = []
        self.user_data = {}
        self.user_to_sample_indices = {}
        self.samples = []  

        label_set = set(label_indices) if label_indices is not None else None

        use_cache = (preprocess_cache_dir is not None) and (not preprocess_force_rebuild)
        if use_cache:
            os.makedirs(preprocess_cache_dir, exist_ok=True)

        tasks = []
        grouped = df.groupby(user_col, sort=False)

        for uid, sub in grouped:
            uid = int(uid)
            uidx = self.user_id_to_idx[uid]

            sub_global_idx = sub.index.to_numpy(np.int64)   # shape [n]

            sub = sub.reset_index(drop=True)
            n = len(sub)
            if n < self.min_history + 1:
                continue

            if use_cache:
                cache_path = os.path.join(preprocess_cache_dir, f"user_{uidx}.pkl")
                if os.path.exists(cache_path):
                    try:
                        payload = _geolife_cache_load(preprocess_cache_dir, uidx)
                        self.user_data[uidx] = payload["user_data"]
                        end_positions = payload["end_positions"]

                        self.user_to_sample_indices[uidx] = []
                        for end_pos in end_positions:
                            sample_id = len(self.samples)
                            self.samples.append((uidx, int(end_pos)))
                            self.user_to_sample_indices[uidx].append(sample_id)

                        if len(self.user_to_sample_indices[uidx]) > 0:
                            self.users.append(uidx)
                        continue
                    except Exception as e:
                        pass

            grid_ids = sub[grid_col].to_numpy()
            dur_vals = sub[dur_col].to_numpy()
            stime_vals = sub[time_col].to_numpy()
            ctx_vals = None if ctx_col is None else sub[ctx_col].tolist()

            tasks.append((
                uid, uidx,
                sub_global_idx,
                grid_ids, dur_vals, stime_vals, ctx_vals,
                self.grid_id_to_idx, self.tb_cfg, ctx_col,
                self.max_seq_len, self.min_history, self.stride, label_set,
            ))

        if len(tasks) > 0:
            if preprocess_num_workers is None:
                preprocess_num_workers = max(1, (os.cpu_count() or 8) - 1)

            mp_ctx = _mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=preprocess_num_workers, mp_context=mp_ctx) as ex:
                futures = [ex.submit(_geolife_build_one_user, t) for t in tasks]
                iterator = as_completed(futures)
                if show_progress:
                    iterator = tqdm(iterator, total=len(futures), desc="build users", unit="user")
                for f in iterator:
                    res = f.result()
                    if res is None:
                        continue
                    uidx, user_data, end_positions = res

                    self.user_data[uidx] = user_data
                    self.user_to_sample_indices[uidx] = []
                    for end_pos in end_positions:
                        sample_id = len(self.samples)
                        self.samples.append((uidx, int(end_pos)))
                        self.user_to_sample_indices[uidx].append(sample_id)

                    if len(self.user_to_sample_indices[uidx]) > 0:
                        self.users.append(uidx)

                    if preprocess_cache_dir is not None:
                        try:
                            _geolife_cache_save(preprocess_cache_dir, uidx, {
                                "user_data": user_data,
                                "end_positions": end_positions,
                            })
                        except Exception:
                            pass

        self.users = sorted(self.users)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        uidx, end_pos = self.samples[idx]
        d = self.user_data[uidx]

        start_pos = max(0, end_pos - self.max_seq_len + 1)
        hist_len = end_pos - start_pos + 1

        L = self.max_seq_len
        grid_seq = torch.zeros(L, dtype=torch.long)
        dur_idx = torch.zeros(L, dtype=torch.long)
        seq_mask = torch.zeros(L, dtype=torch.long)

        year_idx = torch.zeros(L, dtype=torch.long)
        month_idx = torch.zeros(L, dtype=torch.long)
        quarter_idx = torch.zeros(L, dtype=torch.long)
        day_idx = torch.zeros(L, dtype=torch.long)
        dow_idx = torch.zeros(L, dtype=torch.long)
        hour_idx = torch.zeros(L, dtype=torch.long)
        ampm_idx = torch.zeros(L, dtype=torch.long)
        holiday_idx = torch.zeros(L, dtype=torch.long)
        minute_idx = torch.zeros(L, dtype=torch.long)

        sl = slice(start_pos, end_pos + 1)

        grid_seq[:hist_len] = torch.from_numpy(d["grid"][sl])
        dur_idx[:hist_len] = torch.from_numpy(d["dur"][sl])
        seq_mask[:hist_len] = 1

        year_idx[:hist_len] = torch.from_numpy(d["year"][sl])
        month_idx[:hist_len] = torch.from_numpy(d["month"][sl])
        quarter_idx[:hist_len] = torch.from_numpy(d["quarter"][sl])
        day_idx[:hist_len] = torch.from_numpy(d["day"][sl])
        dow_idx[:hist_len] = torch.from_numpy(d["dow"][sl])
        hour_idx[:hist_len] = torch.from_numpy(d["hour"][sl])
        ampm_idx[:hist_len] = torch.from_numpy(d["ampm"][sl])
        holiday_idx[:hist_len] = torch.from_numpy(d["holiday"][sl])
        minute_idx[:hist_len] = torch.from_numpy(d["minute"][sl])

        # label（时间+位置）
        last_stime_minute = float(d["stime_minute"][end_pos])
        next_stime_minute = float(d["stime_minute"][end_pos + 1])

        delta_next_minute = max(1.0, next_stime_minute - last_stime_minute)
        log_delta_next = math.log(delta_next_minute + 1e-6)
        next_grid = int(d["grid"][end_pos + 1])

        C = self.max_ctx_num
        ctx_start_grid = torch.zeros(C, dtype=torch.long)
        ctx_end_grid = torch.zeros(C, dtype=torch.long)
        ctx_time_text_min = torch.zeros(C, dtype=torch.float32)
        ctx_sigma_min = torch.zeros(C, dtype=torch.float32)
        ctx_delta_min = torch.zeros(C, dtype=torch.float32)
        ctx_mask = torch.zeros(C, dtype=torch.long)

        pos_list = d["ctx_positions"]
        if pos_list.size > 0:
            j = bisect.bisect_right(pos_list.tolist(), end_pos)
            if j > 0:
                start_j = max(0, j - C)
                sel = slice(start_j, j)
                kN = j - start_j

                ctx_start_grid[:kN] = torch.from_numpy(d["ctx_start"][sel])
                ctx_end_grid[:kN] = torch.from_numpy(d["ctx_end"][sel])
                ctx_time_text_min[:kN] = torch.from_numpy(d["ctx_time"][sel])
                ctx_sigma_min[:kN] = torch.from_numpy(d["ctx_sigma"][sel])
                ctx_delta_min[:kN] = ctx_time_text_min[:kN] - float(last_stime_minute)
                ctx_mask[:kN] = 1

        return {
            "user_id": torch.tensor(uidx, dtype=torch.long),

            "grid_seq": grid_seq,
            "dur_idx": dur_idx,
            "seq_mask": seq_mask,

            "year_idx": year_idx,
            "month_idx": month_idx,
            "quarter_idx": quarter_idx,
            "day_idx": day_idx,
            "dow_idx": dow_idx,
            "hour_idx": hour_idx,
            "ampm_idx": ampm_idx,
            "holiday_idx": holiday_idx,
            "minute_idx": minute_idx,

            "last_stime_minute": torch.tensor(last_stime_minute, dtype=torch.float32),
            "log_delta_next": torch.tensor(log_delta_next, dtype=torch.float32),
            "next_grid": torch.tensor(next_grid, dtype=torch.long),

            "ctx_start_grid": ctx_start_grid,
            "ctx_end_grid": ctx_end_grid,
            "ctx_time_text_min": ctx_time_text_min,
            "ctx_sigma_min": ctx_sigma_min,
            "ctx_delta_min": ctx_delta_min,
            "ctx_mask": ctx_mask,
        }



def collate_fn(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def make_dataloader_user_shuffle_fast(dataset, batch_size, 
                                      shuffle_users=True, seed=42, num_workers=8):
    sampler = UserOrderSampler(dataset.user_to_sample_indices, seed=seed, shuffle_users=shuffle_users)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=True,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return loader, sampler

def move_batch_to_device(batch: dict, device: str):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

@dataclass
class UnifiedConfig:
    num_users: int             
    num_grids: int            
    max_seq_len: int = 64     
    max_ctx_num: int = 16      
    d_model: int = 128    
    n_heads: int = 4   
    n_layers: int = 2 
    d_user: int = 32   
    d_grid: int = 64  
    d_time_bucket: int = 72  
    d_dur: int = 16  
    d_ctx: int = 64  
    dist_type: str = "student_t" 

    use_novelty_gate: bool = True
    novelty_gate_hidden: int = 128
    novelty_tau: float = 1.0          
    novelty_mix_space: str = "prob"  

    time_bucket_sizes: Dict[str, int] = None

    def __post_init__(self):
        if self.time_bucket_sizes is None:
            self.time_bucket_sizes = {
                "year": 30,        
                "month": 12,
                "quarter": 4,
                "day": 31,
                "dow": 7,
                "hour": 24,
                "ampm": 2,
                "is_holiday": 2,
                "minute": 60,
            }


class NoveltyGate(nn.Module):

    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d_in]
        alpha = torch.sigmoid(self.net(x))  # [B,1]
        return alpha

class TrajectoryEncoder(nn.Module):

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg

        self.user_emb = nn.Embedding(cfg.num_users, cfg.d_user)

        self.grid_emb = nn.Embedding(cfg.num_grids, cfg.d_grid)

        per_bucket_dim = cfg.d_time_bucket // len(cfg.time_bucket_sizes)

        self.time_embs = nn.ModuleDict()
        for name, size in cfg.time_bucket_sizes.items():

            self.time_embs[name] = nn.Embedding(size, per_bucket_dim)

        self.dur_emb = nn.Embedding(128, cfg.d_dur)  # duration 已离散化

        in_dim = cfg.d_user + cfg.d_grid + cfg.d_time_bucket + cfg.d_dur

        self.input_proj = nn.Linear(in_dim, cfg.d_model)

        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:

        grid_seq = batch["grid_seq"]
        B, L = grid_seq.shape

        user_emb = self.user_emb(batch["user_id"])          # [B,d_user]
        user_emb = user_emb.unsqueeze(1).expand(-1, L, -1)  # [B,L,d_user]

        grid_emb = self.grid_emb(grid_seq)                  # [B,L,d_grid]

        time_parts = [
            self.time_embs["year"](batch["year_idx"]),
            self.time_embs["month"](batch["month_idx"]),
            self.time_embs["quarter"](batch["quarter_idx"]),
            self.time_embs["day"](batch["day_idx"]),
            self.time_embs["dow"](batch["dow_idx"]),
            self.time_embs["hour"](batch["hour_idx"]),
            self.time_embs["ampm"](batch["ampm_idx"]),
            self.time_embs["is_holiday"](batch["holiday_idx"]),
            self.time_embs["minute"](batch["minute_idx"]),
        ]

        time_emb = torch.cat(time_parts, dim=-1)          

        dur_emb = self.dur_emb(batch["dur_idx"])           

        x = torch.cat([user_emb, grid_emb, time_emb, dur_emb], dim=-1) 
        x = self.input_proj(x)                                            

        pos_ids = torch.arange(L, device=x.device).unsqueeze(0)  # [1,L] 
        x = x + self.pos_emb(pos_ids)

        src_key_padding_mask = (batch["seq_mask"] == 0)        # [B,L]
        H = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B,L,d_model]

        seq_len = batch["seq_mask"].sum(dim=1)    # [B]
        last_idx = (seq_len - 1).clamp(min=0)     # [B]
        h_traj = H[torch.arange(B, device=x.device), last_idx]  # [B,d_model]
        return h_traj

class ContextModule(nn.Module):

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        self.grid_emb = nn.Embedding(cfg.num_grids, cfg.d_grid)

        self.time_mlp = nn.Sequential(
            nn.Linear(3, cfg.d_ctx),
            nn.ReLU(),
            nn.Linear(cfg.d_ctx, cfg.d_ctx),
            nn.ReLU(),
        )

        self.ctx_proj = nn.Linear(cfg.d_grid * 2 + cfg.d_ctx, cfg.d_ctx)

        self.gamma_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        t_hat_minute: torch.Tensor,
    ):

        device = t_hat_minute.device
        start_grid = batch["ctx_start_grid"].to(device)        # [B,C]
        end_grid = batch["ctx_end_grid"].to(device)            # [B,C]
        time_text_min = batch["ctx_time_text_min"].to(device)  # [B,C]
        sigma_min = batch["ctx_sigma_min"].to(device)          # [B,C]
        delta_min = batch["ctx_delta_min"].to(device)          # [B,C]
        ctx_mask = batch["ctx_mask"].to(device).float()        # [B,C]

        B, C = start_grid.shape

        t_hat_day = t_hat_minute.unsqueeze(1) / (60.0 * 24.0)     # [B,1]
        time_text_day = time_text_min / (60.0 * 24.0)             # [B,C]
        sigma_day = torch.clamp(sigma_min / (60.0 * 24.0), min=1e-3)   
        delta_day = delta_min / (60.0 * 24.0)                     # [B,C]

        diff = t_hat_day - time_text_day                          # [B,C]
        s_time = torch.exp(- (diff ** 2) / (2.0 * sigma_day ** 2))  # [B,C]

        s_time = s_time * ctx_mask

        log_sigma = torch.log(sigma_day + 1e-6)            # [B,C]
        gamma_input = log_sigma.unsqueeze(-1)              # [B,C,1]
        gamma = self.gamma_mlp(gamma_input).squeeze(-1)    # [B,C] ∈ (0,1)

        start_emb = self.grid_emb(start_grid)              # [B,C,d_grid]
        end_emb = self.grid_emb(end_grid)                  # [B,C,d_grid]

        time_feat = torch.stack([time_text_day, sigma_day, delta_day], dim=-1)  # [B,C,3]
        time_vec = self.time_mlp(time_feat)                                     # [B,C,d_ctx]

        ctx_cat = torch.cat([start_emb, end_emb, time_vec], dim=-1)             # [B,C, 2*d_grid+d_ctx]
        c_k = self.ctx_proj(ctx_cat)                                            # [B,C,d_ctx]

        s_tilde = s_time * gamma            # [B,C]
        u = s_tilde.sum(dim=1, keepdim=True)  # [B,1]

        eps = 1e-8
        w = s_tilde / (s_tilde.sum(dim=1, keepdim=True) + eps)   # [B,C]
        w_expanded = w.unsqueeze(-1)                             # [B,C,1]

        c_ctx = (w_expanded * c_k).sum(dim=1)                    # [B,d_ctx]

        return c_ctx, u



class UnifiedTimeLocationModel(nn.Module):

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.dist_type in ["student_t", "lognormal"]

        self.traj_encoder = TrajectoryEncoder(cfg)
        self.ctx_module = ContextModule(cfg)

        self.time_head_coarse = nn.Linear(cfg.d_model, 2)   
        if cfg.dist_type == "student_t":
            self.time_head_coarse_nu = nn.Linear(cfg.d_model, 1)

        self.time_refine_mlp = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.d_ctx + cfg.d_grid, cfg.d_model),
            nn.ReLU(),
        )
        self.time_head_mu = nn.Linear(cfg.d_model, 1)
        self.time_head_log_sigma = nn.Linear(cfg.d_model, 1)
        if cfg.dist_type == "student_t":
            self.time_head_log_nu = nn.Linear(cfg.d_model, 1)

        self.time_emb_mlp = nn.Sequential(
            nn.Linear(3, cfg.d_model),   # mu, log_sigma, 
            nn.ReLU(),
        )

        self.grid_emb = nn.Embedding(cfg.num_grids, cfg.d_grid)

        self.pos_mlp = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.d_ctx + cfg.d_model, cfg.d_model),
            nn.ReLU(),
        )
        self.pos_head = nn.Linear(cfg.d_model, cfg.num_grids)

        self.register_buffer("user_prior_prob", None, persistent=False)   # [U,G]
        self.register_buffer("global_prior_prob", None, persistent=False) # [G]

        gate_in_dim = cfg.d_model + 1 + cfg.d_model
        self.novelty_gate = NoveltyGate(gate_in_dim, hidden=cfg.novelty_gate_hidden)

    def set_freq_priors(self, user_prior_prob: torch.Tensor, global_prior_prob: torch.Tensor):
        """
        user_prior_prob: [num_users, num_grids]
        global_prior_prob: [num_grids]
        """
        self.user_prior_prob = user_prior_prob
        self.global_prior_prob = global_prior_prob

    def _time_nll(self, y: torch.Tensor, mu: torch.Tensor,
                  log_sigma: torch.Tensor, nu: Optional[torch.Tensor] = None) -> torch.Tensor:
   
        sigma = torch.exp(torch.clamp(log_sigma, min=-5.0, max=5.0)) + 1e-6
        if self.cfg.dist_type == "lognormal":
            const = 0.5 * math.log(2 * math.pi)
            nll = const + log_sigma + (y - mu) ** 2 / (2.0 * sigma ** 2)
        else:
            nu = torch.exp(torch.clamp(nu, min=-2.0, max=4.0)) + 1.0   # ~[1.1,55]
            z = (y - mu) / sigma
            nll = torch.log(sigma) + 0.5 * (nu + 1.0) * torch.log1p(z ** 2 / nu)
        return nll.mean()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        for k in ["user_id", "grid_seq", "year_idx", "month_idx", "quarter_idx",
                  "day_idx", "dow_idx", "hour_idx", "ampm_idx", "holiday_idx",
                  "minute_idx", "dur_idx", "seq_mask"]:
            batch[k] = batch[k].to(device)

        h_traj = self.traj_encoder(batch)  # [B,d_model]

        coarse_out = self.time_head_coarse(h_traj)          # [B,2]
        mu0, log_sigma0 = coarse_out[:, 0], coarse_out[:, 1]  # [B], [B]
        if self.cfg.dist_type == "student_t":
            log_nu0 = self.time_head_coarse_nu(h_traj).squeeze(-1)  # [B]
        else:
            log_nu0 = None

        last_stime_minute = batch["last_stime_minute"].to(device)   # [B]
        delta_hat = torch.exp(torch.clamp(mu0, min=-5.0, max=10.0)) # [B]
        t_hat_minute = last_stime_minute + delta_hat                # [B]

        for k in ["ctx_start_grid", "ctx_end_grid", "ctx_time_text_min",
                  "ctx_sigma_min", "ctx_delta_min", "ctx_mask"]:
            batch[k] = batch[k].to(device)

        c_ctx, u = self.ctx_module(batch, t_hat_minute)       # [B,d_ctx], [B,1]

        if self.cfg.dist_type == "student_t":
            time_feat = torch.stack([mu0, log_sigma0, log_nu0], dim=-1)  # [B,3]
        else:
            time_feat = torch.stack([mu0, log_sigma0, torch.zeros_like(mu0)], dim=-1)
        time_vec = self.time_emb_mlp(time_feat)                         # [B,d_model]

        h_joint = torch.cat([h_traj, c_ctx, time_vec], dim=-1)          # [B, d_model+d_ctx+d_model]
        z_pos = self.pos_mlp(h_joint)                                   # [B,d_model]

        logits_pos_raw = self.pos_head(z_pos)  # [B, num_grids]

        next_grid = batch["next_grid"].to(device)  # [B]

        if (self.cfg.use_novelty_gate
            and (self.user_prior_prob is not None)
            and (self.global_prior_prob is not None)):

            gate_x = torch.cat([h_traj, u, time_vec], dim=-1)  # [B, d_model+1+d_model]
            alpha = self.novelty_gate(gate_x)                  # [B,1]

            uidx = batch["user_id"].to(device)                  # [B]
            p_user = self.user_prior_prob[uidx]                 # [B,G]
            p_global = self.global_prior_prob.to(device).unsqueeze(0).expand_as(p_user)  # [B,G]

            p_freq = 0.8 * p_user + 0.2 * p_global             # [B,G]

            tau = float(getattr(self.cfg, "novelty_tau", 1.0))
            if tau != 1.0:
                p_freq = torch.pow(torch.clamp(p_freq, 1e-8, 1.0), 1.0 / tau)
                p_freq = p_freq / (p_freq.sum(dim=1, keepdim=True) + 1e-8)

            p_model = torch.softmax(logits_pos_raw, dim=-1)    # [B,G]

            p_mix = alpha * p_model + (1.0 - alpha) * p_freq
            p_mix = torch.clamp(p_mix, 1e-12, 1.0)

            pos_loss = -torch.log(
                p_mix[torch.arange(p_mix.size(0), device=device), next_grid]
            ).mean()

            logits_pos = torch.log(p_mix)

        else:
            alpha = None
            pos_loss = F.cross_entropy(logits_pos_raw, next_grid)
            logits_pos = logits_pos_raw
    
        p_pos = F.softmax(logits_pos, dim=-1)                           # [B,num_grids]
        z_loc = torch.matmul(p_pos, self.grid_emb.weight)               # [B,d_grid]

        h_time_refine = torch.cat([h_traj, c_ctx, z_loc], dim=-1)       # [B, d_model+d_ctx+d_grid]
        h_time_refine = self.time_refine_mlp(h_time_refine)             # [B,d_model]

        mu = self.time_head_mu(h_time_refine).squeeze(-1)               # [B]
        log_sigma = self.time_head_log_sigma(h_time_refine).squeeze(-1) # [B]
        if self.cfg.dist_type == "student_t":
            log_nu = self.time_head_log_nu(h_time_refine).squeeze(-1)   # [B]
        else:
            log_nu = None

        y = batch["log_delta_next"].to(device)                          # [B]
        time_nll = self._time_nll(y, mu, log_sigma, log_nu)

        total_loss = time_nll + pos_loss

        out = {
            "loss": total_loss,
            "time_nll": time_nll.detach(),
            "pos_loss": pos_loss.detach(),
            "mu": mu.detach(),
            "log_sigma": log_sigma.detach(),
            "logits_pos": logits_pos,
            "c_ctx": c_ctx.detach(),
            "u_ctx": u.detach(),
        }

        if log_nu is not None:
            out["log_nu"] = log_nu.detach() # [B,G]
        if alpha is not None:
            out["alpha_novelty"] = alpha.detach()

        return out
    

def is_ddp() -> bool:
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and int(os.environ["WORLD_SIZE"]) > 1


def ddp_setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def ddp_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

def ddp_world() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def unwrap_model(m):
    return m.module if hasattr(m, "module") else m

def ddp_all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def build_df_train_segment(df_sorted, user2pos, train_ratio, seq_len):

    train_rows_all = []
    for uid, pos_list in user2pos.items():
        pos_list = list(pos_list)
        n = len(pos_list)
        if n < seq_len + 1:
            continue

        cut = int(train_ratio * n)
        cut = max(cut, seq_len)       
        cut = min(cut, n)             

        train_rows_all.extend(pos_list[:cut])

    train_rows_all = sorted(set(train_rows_all))
    return df_sorted.iloc[train_rows_all].copy()

@torch.no_grad()
def ranking_metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, ks=(1,5,10)):

    B, G = logits.shape
    target = target.view(-1)
    max_k = max(ks)

    topk_idx = torch.topk(logits, k=max_k, dim=1).indices

    hits = (topk_idx == target.unsqueeze(1))

    rank_positions = torch.arange(1, max_k + 1, device=logits.device).view(1, -1).expand(B, -1)
    ranks = torch.where(hits, rank_positions, torch.zeros_like(rank_positions))

    rr = torch.where(ranks > 0, 1.0 / ranks.float(), torch.zeros_like(ranks, dtype=torch.float))
    rr_max = rr.max(dim=1).values  # [B]

    discounts = 1.0 / torch.log2(torch.arange(2, max_k + 2, device=logits.device).float())  # [max_k]
    dcg = (hits.float() * discounts.view(1, -1)).sum(dim=1)  # [B]
    ndcg_full = dcg  # since single relevant, best dcg=1.0

    out = {}
    for K in ks:
        hK = hits[:, :K]
        acc = hK.any(dim=1).float().mean().item()

        ranksK = torch.where(hK, rank_positions[:, :K], torch.zeros_like(rank_positions[:, :K]))
        rrK = torch.where(ranksK > 0, 1.0 / ranksK.float(), torch.zeros_like(ranksK, dtype=torch.float))
        mrr = rrK.max(dim=1).values.mean().item()

        discountsK = discounts[:K]
        ndcg = (hK.float() * discountsK.view(1, -1)).sum(dim=1).mean().item()

        out[f"acc@{K}"] = acc
        out[f"mrr@{K}"] = mrr
        out[f"ndcg@{K}"] = ndcg
    return out


def build_user_topM_uidx_gidx(df_train, train_ds, M=200):

    user_topM = {}
    for uid, sub in df_train.groupby("userID"):
        uid = int(uid)
        if uid not in train_ds.user_id_to_idx:
            continue
        uidx = int(train_ds.user_id_to_idx[uid])

        gidx = sub["grid"].map(train_ds.grid_id_to_idx).dropna().astype(int)
        vc = gidx.value_counts()
        user_topM[uidx] = vc.index[:M].tolist()
    return user_topM

def build_global_top_gidx(df_train, train_ds, M=100):
    gidx = df_train["grid"].map(train_ds.grid_id_to_idx).dropna().astype(int)
    vc = gidx.value_counts()
    return vc.index[:M].astype(int).tolist()



@torch.no_grad()
def rerank_metrics_from_logits(
    logits_full,                 # [B,G]
    target_full,                 # [B] in same id space as logits columns
    user_ids,                    # [B]
    user_topM_dict,              # {uid: [grid_id,...]}
    global_top_list=None,        # list of grid_id
    ks=(1,5,10),
):

    device = logits_full.device
    B, G = logits_full.shape
    max_k = max(ks)

    cands = []
    cand_lens = []
    for i in range(B):
        uid = int(user_ids[i])
        lst = []
        if uid in user_topM_dict:
            lst += user_topM_dict[uid]
        if global_top_list is not None:
            lst += global_top_list
        seen = set()
        uniq = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        cands.append(uniq)
        cand_lens.append(len(uniq))
    Kcand = max(cand_lens)

    cand_idx = torch.zeros((B, Kcand), dtype=torch.long, device=device)
    cand_mask = torch.zeros((B, Kcand), dtype=torch.bool, device=device)
    for i, lst in enumerate(cands):
        k = len(lst)
        cand_idx[i, :k] = torch.tensor(lst, dtype=torch.long, device=device)
        cand_mask[i, :k] = True

    logits_cand = torch.gather(logits_full, dim=1, index=cand_idx)

    logits_cand = logits_cand.masked_fill(~cand_mask, float("-inf"))

    topk_idx_in_cand = torch.topk(logits_cand, k=min(max_k, Kcand), dim=1).indices  # [B,max_k]
    topk_grid = cand_idx.gather(1, topk_idx_in_cand)  # [B,max_k] map full grid id

    hits = (topk_grid == target_full.unsqueeze(1))

    out = {}
    for K in ks:
        out[f"acc@{K}"] = hits[:, :K].any(dim=1).float().mean().item()

        rank_pos = torch.arange(1, K+1, device=device).view(1, -1).expand(B, -1)
        ranks = torch.where(hits[:, :K], rank_pos, torch.zeros_like(rank_pos))
        rr = torch.where(ranks > 0, 1.0 / ranks.float(), torch.zeros_like(ranks, dtype=torch.float))
        out[f"mrr@{K}"] = rr.max(dim=1).values.mean().item()

        # NDCG@K (single relevant)
        discounts = 1.0 / torch.log2(torch.arange(2, K+2, device=device).float())
        dcg = (hits[:, :K].float() * discounts.view(1, -1)).sum(dim=1)
        out[f"ndcg@{K}"] = dcg.mean().item()

    in_cand = (cand_idx == target_full.unsqueeze(1)).any(dim=1).float().mean().item()
    out["cand_hit_rate"] = in_cand
    out["Kcand_mean"] = float(np.mean(cand_lens))
    return out



# -----------------------------
# Logging: CSV
# -----------------------------
LOG_HEADER = [
    "step","epoch","train_loss","val_loss","lr","epoch_time_sec",
    "train_acc@1","val_acc@1","train_mrr@1","val_mrr@1","train_ndcg@1","val_ndcg@1",
    "train_acc@5","val_acc@5","train_mrr@5","val_mrr@5","train_ndcg@5","val_ndcg@5",
    "train_acc@10","val_acc@10","train_mrr@10","val_mrr@10","train_ndcg@10","val_ndcg@10",

    "train_rr_acc@1","val_rr_acc@1","train_rr_mrr@1","val_rr_mrr@1","train_rr_ndcg@1","val_rr_ndcg@1",
    "train_rr_acc@5","val_rr_acc@5","train_rr_mrr@5","val_rr_mrr@5","train_rr_ndcg@5","val_rr_ndcg@5",
    "train_rr_acc@10","val_rr_acc@10","train_rr_mrr@10","val_rr_mrr@10","train_rr_ndcg@10","val_rr_ndcg@10",

    "train_cand_hit_rate","val_cand_hit_rate",
    "context_mode",
]


def append_log_row(log_file: str, row: dict):
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_exists = os.path.exists(log_file)

    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_HEADER)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_HEADER})


def apply_context_ablation(
    batch,
    context_mode: str = "full",
    ablate_user: bool = False,
    ablate_time: bool = False,
) -> dict:

    if context_mode == "none":
        for k in ["ctx_start_grid", "ctx_end_grid",
                  "ctx_time_text_min", "ctx_sigma_min", "ctx_delta_min"]:
            if k in batch and hasattr(batch[k], "zero_"):
                batch[k].zero_()
        if "ctx_mask" in batch and hasattr(batch["ctx_mask"], "zero_"):
            batch["ctx_mask"].zero_()

    elif context_mode == "time_only":
        for k in ["ctx_start_grid", "ctx_end_grid"]:
            if k in batch and hasattr(batch[k], "zero_"):
                batch[k].zero_()

    elif context_mode == "grid_only":
        for k in ["ctx_time_text_min", "ctx_sigma_min", "ctx_delta_min"]:
            if k in batch and hasattr(batch[k], "zero_"):
                batch[k].zero_()

    elif context_mode == "full":
        pass
    else:
        raise ValueError(f"Unknown context_mode={context_mode!r}")

    if ablate_user:
        if "user_id" in batch and hasattr(batch["user_id"], "fill_"):
            batch["user_id"].fill_(0)

    if ablate_time:
        time_bucket_keys = [
            "year_idx", "month_idx", "quarter_idx", "day_idx", "dow_idx",
            "hour_idx", "ampm_idx", "holiday_idx", "minute_idx"
        ]
        for k in time_bucket_keys:
            if k in batch and hasattr(batch[k], "zero_"):
                batch[k].zero_()

        for k in ["last_stime_minute", "log_delta_next", "delta_next_min"]:
            if k in batch and hasattr(batch[k], "zero_"):
                batch[k].zero_()

        for k in ["ctx_time_text_min", "ctx_sigma_min", "ctx_delta_min"]:
            if k in batch and hasattr(batch[k], "zero_"):
                batch[k].zero_()

    return batch

def save_checkpoint(path, model, optimizer, epoch, global_step, best_val_loss=None, extra=None):
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": unwrap_model(model).state_dict(),      
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "extra": extra,
    }
    
    torch.save(ckpt, path)

def load_latest_checkpoint(checkpoint_dir: str):
    pattern = os.path.join(checkpoint_dir, "checkpoint_*.pth")
    files = glob.glob(pattern)
    if not files:
        return None
    latest = max(files, key=os.path.getctime)
    return latest

@torch.no_grad()
def evaluate_one_epoch(model, loader, device, ks=(1,5,10), max_batches=None, use_amp=True,
                       rerank_dict=None,
                       context_mode="full", ablate_user=False, ablate_time=False):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    acc = {K: 0.0 for K in ks}
    mrr = {K: 0.0 for K in ks}
    ndcg = {K: 0.0 for K in ks}

    rr_acc, rr_mrr, rr_ndcg = {K: 0.0 for K in ks}, {K: 0.0 for K in ks}, {K: 0.0 for K in ks}
    cand_hit_rate = 0.0

    for bidx, batch in enumerate(loader):
        if (max_batches is not None) and (bidx >= max_batches):
            break
        batch = move_batch_to_device(batch, device)
        batch = apply_context_ablation(batch, context_mode=context_mode,
                                      ablate_user=ablate_user, ablate_time=ablate_time)


        with torch.autocast(device_type=("cuda" if "cuda" in device else "cpu"),
                            dtype=torch.float16, enabled=(use_amp and ("cuda" in device))):
            out = model(batch)

            if isinstance(out, dict) and ("loss" in out) and torch.is_tensor(out["loss"]):
                loss = out["loss"]
            else:
                logits = out["logits_pos"]
                target = batch["next_grid"]
                loss = F.cross_entropy(logits, target, reduction="mean")

        total_loss += float(loss.item())
        n_batches += 1

        logits = out["logits_pos"]
        target = batch["next_grid"]
        met = ranking_metrics_from_logits(logits, target, ks=ks)
        for K in ks:
            acc[K] += met[f"acc@{K}"]
            mrr[K] += met[f"mrr@{K}"]
            ndcg[K] += met[f"ndcg@{K}"]

        if rerank_dict is not None:
            user_ids = batch["user_id"] 
            rr_met = rerank_metrics_from_logits(
                logits, target, user_ids, 
                user_topM_dict=rerank_dict["user_topM"],
                global_top_list=rerank_dict["global_top"],
                ks=ks
            )
            cand_hit_rate += rr_met["cand_hit_rate"]
            for K in ks:
                rr_acc[K] += rr_met[f"acc@{K}"]
                rr_mrr[K] += rr_met[f"mrr@{K}"]
                rr_ndcg[K] += rr_met[f"ndcg@{K}"]

    if n_batches == 0:
        return {
            "loss": float("nan"),
            **{f"acc@{K}": float("nan") for K in ks},
            **{f"mrr@{K}": float("nan") for K in ks},
            **{f"ndcg@{K}": float("nan") for K in ks},
            **{f"rr_acc@{K}": float("nan") for K in ks},
            **{f"rr_mrr@{K}": float("nan") for K in ks},
            **{f"rr_ndcg@{K}": float("nan") for K in ks},
        }

    if dist.is_available() and dist.is_initialized():
        dev = torch.device(device) if isinstance(device, str) else device
        pack = [total_loss, float(n_batches)]
        for K in ks:
            pack += [acc[K], mrr[K], ndcg[K]]
            if rerank_dict is not None:
                pack += [rr_acc[K], rr_mrr[K], rr_ndcg[K]]
        if rerank_dict is not None:
            pack += [cand_hit_rate]

        t = torch.tensor(pack, device=dev, dtype=torch.float64)
        ddp_all_reduce_sum(t)
        pack = t.tolist()

        total_loss = pack[0]
        n_batches = int(pack[1])

        idx = 2
        for K in ks:
            acc[K], mrr[K], ndcg[K] = pack[idx], pack[idx+1], pack[idx+2]
            idx += 3
            if rerank_dict is not None:
                rr_acc[K], rr_mrr[K], rr_ndcg[K] = pack[idx], pack[idx+1], pack[idx+2]
                idx += 3
        if rerank_dict is not None:
            cand_hit_rate = pack[idx]

    outm = {"loss": total_loss / n_batches}
    for K in ks:
        outm[f"acc@{K}"] = acc[K] / n_batches
        outm[f"mrr@{K}"] = mrr[K] / n_batches
        outm[f"ndcg@{K}"] = ndcg[K] / n_batches
        if rerank_dict is not None:
            outm.update({
                f"rr_acc@{K}": rr_acc[K]/n_batches, 
                f"rr_mrr@{K}": rr_mrr[K]/n_batches,
                f"rr_ndcg@{K}": rr_ndcg[K]/n_batches
            })
            outm["cand_hit_rate"] = cand_hit_rate / max(1, n_batches)
    return outm

def train_one_epoch(model, loader, optimizer, device, epoch, ks=(1,5,10),
                    grad_clip=1.0, use_amp=True, scaler=None, sampler=None,
                    rerank_dict=None,
                    context_mode="full", ablate_user=False, ablate_time=False):
    model.train()
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    total_loss = 0.0
    n_batches = 0

    acc = {K: 0.0 for K in ks}
    mrr = {K: 0.0 for K in ks}
    ndcg = {K: 0.0 for K in ks}

    rr_acc, rr_mrr, rr_ndcg = {K: 0.0 for K in ks}, {K: 0.0 for K in ks}, {K: 0.0 for K in ks}
    cand_hit_rate = 0.0

    pbar = tqdm(loader, desc=f"[Train epoch {epoch}]", unit="batch", leave=False)

    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        batch = apply_context_ablation(batch, context_mode=context_mode,
                                      ablate_user=ablate_user, ablate_time=ablate_time)


        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=("cuda" if "cuda" in device else "cpu"),
                            dtype=torch.float16, enabled=(use_amp and ("cuda" in device))):
            out = model(batch)
            if ("loss" in out) and torch.is_tensor(out["loss"]):
                loss = out["loss"]
            else:
                logits = out["logits_pos"]
                target = batch["next_grid"]
                loss = F.cross_entropy(logits, target, reduction="mean")

        if scaler is not None and (use_amp and ("cuda" in device)):
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

        # metrics
        logits = out["logits_pos"]
        target = batch["next_grid"]
        met = ranking_metrics_from_logits(logits, target, ks=ks)
        for K in ks:
            acc[K] += met[f"acc@{K}"]
            mrr[K] += met[f"mrr@{K}"]
            ndcg[K] += met[f"ndcg@{K}"]

        if rerank_dict is not None:
            user_ids = batch["user_id"] 
            rr_met = rerank_metrics_from_logits(
                logits, target, user_ids, 
                user_topM_dict=rerank_dict["user_topM"],
                global_top_list=rerank_dict["global_top"],
                ks=ks
            )
            cand_hit_rate += rr_met["cand_hit_rate"]
            for K in ks:
                rr_acc[K] += rr_met[f"acc@{K}"]
                rr_mrr[K] += rr_met[f"mrr@{K}"]
                rr_ndcg[K] += rr_met[f"ndcg@{K}"]

        pbar.set_postfix(loss=f"{(total_loss/n_batches):.4f}", acc10=f"{(acc[10]/n_batches):.3f}")

    pbar.close()

    if dist.is_available() and dist.is_initialized():
        dev = torch.device(device) if isinstance(device, str) else device
        pack = [total_loss, float(n_batches)]
        for K in ks:
            pack += [acc[K], mrr[K], ndcg[K]]
            if rerank_dict is not None:
                pack += [rr_acc[K], rr_mrr[K], rr_ndcg[K]]
        if rerank_dict is not None:
            pack += [cand_hit_rate]

        t = torch.tensor(pack, device=dev, dtype=torch.float64)
        ddp_all_reduce_sum(t)
        pack = t.tolist()

        total_loss = pack[0]
        n_batches = int(pack[1])

        idx = 2
        for K in ks:
            acc[K], mrr[K], ndcg[K] = pack[idx], pack[idx+1], pack[idx+2]
            idx += 3
            if rerank_dict is not None:
                rr_acc[K], rr_mrr[K], rr_ndcg[K] = pack[idx], pack[idx+1], pack[idx+2]
                idx += 3
        if rerank_dict is not None:
            cand_hit_rate = pack[idx]

    outm = {"loss": total_loss / max(1, n_batches)}
    for K in ks:
        outm[f"acc@{K}"] = acc[K] / max(1, n_batches)
        outm[f"mrr@{K}"] = mrr[K] / max(1, n_batches)
        outm[f"ndcg@{K}"] = ndcg[K] / max(1, n_batches)
        if rerank_dict is not None:
            outm.update({
                f"rr_acc@{K}": rr_acc[K]/n_batches, 
                f"rr_mrr@{K}": rr_mrr[K]/n_batches,
                f"rr_ndcg@{K}": rr_ndcg[K]/n_batches,
                
            })
            outm["cand_hit_rate"] = cand_hit_rate / max(1, n_batches)
            
    return outm

def plot_from_log_csv(log_file: str, 
                      save_path: str):
    import pandas as pd
    df = pd.read_csv(log_file)
    if df.empty:
        print("[plot] log is empty")
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    x = df["epoch"].values

    plt.figure()
    plt.plot(x, df["train_loss"].values, label="train_loss")
    plt.plot(x, df["val_loss"].values, label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(save_path.replace(".png","_loss.png"))
    plt.close()

    for K in [1,5,10]:
        plt.figure()
        plt.plot(x, df[f"train_acc@{K}"].values, label=f"train_acc@{K}")
        plt.plot(x, df[f"val_acc@{K}"].values, label=f"val_acc@{K}")
        plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend()
        plt.tight_layout()
        plt.savefig(save_path.replace(".png", f"_acc@{K}.png"))
        plt.close()

        plt.figure()
        plt.plot(x, df[f"train_mrr@{K}"].values, label=f"train_mrr@{K}")
        plt.plot(x, df[f"val_mrr@{K}"].values, label=f"val_mrr@{K}")
        plt.xlabel("epoch"); plt.ylabel("mrr"); plt.legend()
        plt.tight_layout()
        plt.savefig(save_path.replace(".png", f"_mrr@{K}.png"))
        plt.close()

        plt.figure()
        plt.plot(x, df[f"train_ndcg@{K}"].values, label=f"train_ndcg@{K}")
        plt.plot(x, df[f"val_ndcg@{K}"].values, label=f"val_ndcg@{K}")
        plt.xlabel("epoch"); plt.ylabel("ndcg"); plt.legend()
        plt.tight_layout()
        plt.savefig(save_path.replace(".png", f"_ndcg@{K}.png"))
        plt.close()

        rr_train_col = f"train_rr_acc@{K}"
        rr_val_col   = f"val_rr_acc@{K}"
        if rr_train_col in df.columns and rr_val_col in df.columns:
            plt.figure()
            plt.plot(x, df[rr_train_col].values, label=rr_train_col)
            plt.plot(x, df[rr_val_col].values, label=rr_val_col)
            plt.xlabel("epoch"); plt.ylabel("rr_acc"); plt.legend()
            plt.tight_layout()
            plt.savefig(save_path.replace(".png", f"_rr_acc@{K}.png"))
            plt.close()

    print("[plot] saved:", save_path.replace(".png","_loss.png"),
          "and acc/mrr/ndcg figs")

def Training_Unified(
    train_loader,
    val_loader,
    device,
    model,
    optimizer,
    df_train=None,  
    topKs=(1,5,10),
    checkpoint_dir="./runs/unified_run/checkpoints",
    log_file="./runs/unified_run/train_log.csv",
    curves_prefix="./runs/unified_run/curves.png",
    EPOCHS=50,
    grad_clip=1.0,
    use_amp=True,
    val_max_batches=None,
    context_mode="full",
    ablate_user=False,
    ablate_time=False,
):

    if "cuda" in device:
        torch.cuda.empty_cache()
        gc.collect()

    if ddp_rank() == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    latest = load_latest_checkpoint(checkpoint_dir)
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    rerank_dict = None
    if df_train is not None:
        rerank_dict = {
            "user_topM": build_user_topM_uidx_gidx(df_train, train_ds=train_loader.dataset, M=200),
            "global_top": build_global_top_gidx(df_train, train_ds=train_loader.dataset, M=100),
        }
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print("[rerank] built: user_topM size =", len(rerank_dict["user_topM"]),
                "global_top size =", len(rerank_dict["global_top"]))
    
    if latest is not None:
        print(f"[Train] load latest checkpoint: {latest}")
        ckpt = torch.load(latest, map_location="cpu")
        unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        bvl = ckpt.get("best_val_loss", None)
        if bvl is not None:
            best_val_loss = float(bvl)
        print(f"[Train] resume from epoch={start_epoch}, global_step={global_step}, best_val_loss={best_val_loss:.4f}")
    else:
        print("[Train] start from scratch")

    if "cuda" in device:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    model.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and ("cuda" in device)))


    for epoch in range(start_epoch, start_epoch + EPOCHS):
        epoch_t0 = time.time()

        # ---- Train epoch ----
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            ks=topKs,
            grad_clip=grad_clip,
            use_amp=use_amp,
            scaler=scaler,
            sampler=getattr(train_loader, "sampler", None),
            rerank_dict=rerank_dict,
            context_mode=context_mode,
            ablate_user=ablate_user,
            ablate_time=ablate_time,
        )

        val_metrics = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            ks=topKs,
            max_batches=val_max_batches,
            use_amp=use_amp,
            rerank_dict=rerank_dict,
            context_mode=context_mode,
            ablate_user=ablate_user,
            ablate_time=ablate_time,

        )

        epoch_time = time.time() - epoch_t0
        lr_now = float(optimizer.param_groups[0]["lr"])
        global_step += len(train_loader)

        # ---- Log row ----
        row = {
            "step": global_step,
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "val_loss": float(val_metrics["loss"]),
            "lr": lr_now,
            "epoch_time_sec": float(epoch_time),
        }
        for K in topKs:
            row[f"train_acc@{K}"] = float(train_metrics[f"acc@{K}"])
            row[f"train_mrr@{K}"] = float(train_metrics[f"mrr@{K}"])
            row[f"train_ndcg@{K}"] = float(train_metrics[f"ndcg@{K}"])
            row[f"val_acc@{K}"] = float(val_metrics[f"acc@{K}"])
            row[f"val_mrr@{K}"] = float(val_metrics[f"mrr@{K}"])
            row[f"val_ndcg@{K}"] = float(val_metrics[f"ndcg@{K}"])

            if rerank_dict is not None:
                row[f"train_rr_acc@{K}"]  = float(train_metrics.get(f"rr_acc@{K}", float("nan")))
                row[f"train_rr_mrr@{K}"]  = float(train_metrics.get(f"rr_mrr@{K}", float("nan")))
                row[f"train_rr_ndcg@{K}"] = float(train_metrics.get(f"rr_ndcg@{K}", float("nan")))
                row[f"val_rr_acc@{K}"]    = float(val_metrics.get(f"rr_acc@{K}", float("nan")))
                row[f"val_rr_mrr@{K}"]    = float(val_metrics.get(f"rr_mrr@{K}", float("nan")))
                row[f"val_rr_ndcg@{K}"]   = float(val_metrics.get(f"rr_ndcg@{K}", float("nan")))
        row["train_cand_hit_rate"] = float(train_metrics.get("cand_hit_rate", float("nan")))
        row["val_cand_hit_rate"]   = float(val_metrics.get("cand_hit_rate", float("nan")))

        row["context_mode"] = context_mode

        append_log_row(log_file, row)

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}.pth")
        save_checkpoint(
            ckpt_path, model, optimizer,
            epoch=epoch, global_step=global_step, best_val_loss=best_val_loss,
            extra=None
        )

        # ---- Save best ----
        if not math.isnan(val_metrics["loss"]) and val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_path = os.path.join(checkpoint_dir, "best.pth")
            save_checkpoint(
                best_path, model, optimizer,
                epoch=epoch, global_step=global_step, best_val_loss=best_val_loss,
                extra=None,
            )
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model_state_dict.pt"))
            print(f"[Best] epoch={epoch} best_val_loss={best_val_loss:.4f}")

        print(
            f"[{epoch}] "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"train_acc@10={train_metrics['acc@10']:.3f} val_acc@10={val_metrics['acc@10']:.3f} "
            f"lr={lr_now:.2e} time={epoch_time:.1f}s" 
        )

        gc.collect()
    try:
        plot_from_log_csv(log_file, curves_prefix)
    except Exception as e:
        print("[Warn] plot_from_log_csv failed:", str(e))

    # ---- housekeeping ----
    if "cuda" in device:
        torch.cuda.empty_cache()
    
    print("[Done] Log:", log_file)
    print("[Done] Checkpoints:", checkpoint_dir)
    print("[Done] Curves prefix:", curves_prefix)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print("DEVICE =", DEVICE)

GeoLife_All_PATH = "./Data/GeoLife_all.csv"
GeoLife_Routine_PATH = "./Data/GeoLife_Routine_top3.csv"
GeoLife_NonRoutine_PATH = "./Data/GeoLife_Nonroutine_top3.csv"
GeoLife_Sampled_PATH = "./Data/GeoLifeSampled/"

MoreUser_All_PATH = "./Data/MoreUser_all.csv"
MoreUser_Routine_PATH = "./Data/MoreUser_Routine_top3.csv"
MoreUser_NonRoutine_PATH = "./Data/MoreUser_Nonroutine_top3.csv"
MoreUser_Sampled_PATH = "./Data/MoreUserSampled/"

def _list_csv_files(dir_path: str):
    """Return sorted list of CSV files under dir_path."""
    import glob
    files = glob.glob(os.path.join(dir_path, "*.csv"))
    return sorted(files)

def _dataset_variants_for_experiment(experiments_name: str, datachoose: str):
    """
    For E12/E13/E14, one experiment corresponds to multiple sampled CSV files in a directory.
    This function returns a list of (variant_tag, csv_path).

    - For normal experiments: [(experiments_name, path_to_single_csv)]
    - For E12/E13/E14: [(experiments_name + "__" + <file_stem>, <csv_path>), ...]
    """
    import os
    if datachoose == "GeoLife":
        if experiments_name in ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]:
            return [(experiments_name, GeoLife_All_PATH)]
        elif experiments_name in ["E8", "E10"]:
            return [(experiments_name, GeoLife_Routine_PATH)]
        elif experiments_name in ["E9", "E11"]:
            return [(experiments_name, GeoLife_NonRoutine_PATH)]
        elif experiments_name in ["E12", "E13", "E14"]:
            base_dir = os.path.join(GeoLife_Sampled_PATH, experiments_name)
            files = _list_csv_files(base_dir)
            if len(files) == 0:
                raise FileNotFoundError(f"No sampled CSV found under: {base_dir}")
            out = []
            for f in files:
                stem = os.path.splitext(os.path.basename(f))[0]
                out.append((f"{experiments_name}__{stem}", f))
            return out
        else:
            raise ValueError(f"Unknown experiments_name={experiments_name} for GeoLife")

    elif datachoose == "MoreUser":
        if experiments_name in ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]:
            return [(experiments_name, MoreUser_All_PATH)]
        elif experiments_name in ["E8", "E10"]:
            return [(experiments_name, MoreUser_Routine_PATH)]
        elif experiments_name in ["E9", "E11"]:
            return [(experiments_name, MoreUser_NonRoutine_PATH)]
        elif experiments_name in ["E12", "E13", "E14"]:
            base_dir = os.path.join(MoreUser_Sampled_PATH, experiments_name)
            files = _list_csv_files(base_dir)
            if len(files) == 0:
                raise FileNotFoundError(f"No sampled CSV found under: {base_dir}")
            out = []
            for f in files:
                stem = os.path.splitext(os.path.basename(f))[0]
                out.append((f"{experiments_name}__{stem}", f))
            return out
        else:
            raise ValueError(f"Unknown experiments_name={experiments_name} for MoreUser")
    else:
        raise ValueError("DATACHOOSE 参数错误！ must be 'GeoLife' or 'MoreUser'")


DATACHOOSE = "GeoLife" 

SEED = 42
TEST_RATIO = 0.2

MAX_SEQ_LEN = 64
MAX_CTX_NUM = 16
BATCH_SIZE = 12288

NUM_WORKERS = 0

EPOCHS = 200

LR = 1e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP = 5.0

LOG_EVERY_STEPS = 100
EVAL_EVERY_STEPS = 500
VAL_MAX_BATCHES = 200   
SAVE_DIR = "./runs/unified_run"
RESUME = True     

DIST_TYPE = "student_t" # 或 "lognormal"


Experiments = {
    "E1": {"model_structure": "encoder_only", "userID": False, "time": False, "context": "none"},
    "E2": {"model_structure": "encoder_only", "userID": True, "time": True, "context": "none"},
    "E3": {"model_structure": "encoder_only", "userID": True, "time": True, "context": "fuzzy"},
    "E4": {"model_structure": "encoder_only", "userID": True, "time": True, "context": "exact"},
    "E5": {"model_structure": "encoder_only", "userID": True, "time": False, "context": "none"},
    "E6": {"model_structure": "encoder_only", "userID": False, "time": True, "context": "none"},
    "E7": {"model_structure": "encoder_only", "userID": True, "time": True, "context": "fuzzy"},
    "E8": {"model_structure": "encoder_only", "userID": True, "time": True, "context": "fuzzy"},
    "E9": {"model_structure": "encoder_only", "userID": True, "time": True, "context": "fuzzy"},
    "E10": {"model_structure": "encoder_only", "userID": True, "time": True, "context": "none"},
    "E11": {"model_structure": "encoder_only", "userID": True, "time": True, "context": "none"},

    "E12": {"model_structure": "encoder_only", "userID": True, "time": True, "context": "none"},
    "E13": {"model_structure": "encoder_only", "userID": True, "time": True, "context": "none"},
    "E14": {"model_structure": "encoder_only", "userID": True, "time": True, "context": "none"},
}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Only run CPU preprocessing + cache build. Do NOT init DDP or touch GPU.")
    parser.add_argument("--train-only", action="store_true",
                        help="Only run training. Assume cache already exists (recommended for DDP).")
    parser.add_argument("--cache-dir", type=str, default="./cache_geolife",
                        help="Root directory for preprocessing cache.")
    parser.add_argument("--preprocess-workers", type=int, default=0,
                        help="Number of CPU worker processes for preprocessing (0 = auto).")
    parser.add_argument("--force-rebuild-cache", action="store_true",
                        help="Force rebuild preprocessing cache even if it already exists.")
    args = parser.parse_args()

    preprocess_only = args.preprocess_only and (not is_ddp())
    train_only = args.train_only or is_ddp()

    if preprocess_only:
        DEVICE = "cpu"
    else:
        if torch.cuda.is_available():
            if is_ddp():
                local_rank = ddp_setup()
                DEVICE = f"cuda:{local_rank}"
            else:
                DEVICE = "cuda"
        else:
            DEVICE = "cpu"

    if ddp_rank() == 0:
        print("DEVICE =", DEVICE, "| ddp =", is_ddp(), "| world_size =", ddp_world())

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    for experiments_name, exp_cfg in Experiments.items():
        userIDGate = exp_cfg["userID"]
        timeGate   = exp_cfg["time"]
        contextGate = exp_cfg["context"]

        print(f"Run {experiments_name}- userID={userIDGate}, time={timeGate}, context={contextGate}")

        CTX_COL = None
        context_mode = None 
        ablate_user = not userIDGate 
        ablate_time = not timeGate 
        df = None

        if contextGate == "none": # E1, E2, E5, E6
            context_mode = "none"

            CTX_COL = None 

        elif contextGate == "fuzzy": # E3, E7, E8, E9
            context_mode = "full"
            CTX_COL = "context_fuzzy" 

        elif contextGate == "exact": # E4
            context_mode = "full"
            CTX_COL = "context_precise" 
        else:
            print("Config Error！")
    
        print(f"{experiments_name} cfg -> context_mode={context_mode}, CTX_COL={CTX_COL}, ablate_user={ablate_user}, ablate_time={ablate_time}")

        variants = _dataset_variants_for_experiment(experiments_name, DATACHOOSE)
        for run_id, csv_path in variants:
            if ddp_rank() == 0:
                print(f"[Data] {experiments_name} -> variant={run_id}")
                print(f"[Data] csv_path = {csv_path}")
            df = pd.read_csv(csv_path)

            df_sorted, user2pos = build_user_index(df, user_col="userID", time_col="stime")

            TRAIN_RATIO = 1.0 - TEST_RATIO   
            train_label_rows, test_label_rows = split_labels_with_history(
                user2pos, train_ratio=TRAIN_RATIO, seq_len=MAX_SEQ_LEN
            )

            train_ds = GeoLifeUnifiedDatasetFast(
                preprocess_cache_dir=os.path.join(args.cache_dir, run_id),
                preprocess_num_workers=(args.preprocess_workers if args.preprocess_workers>0 else None),
                preprocess_force_rebuild=args.force_rebuild_cache,
                df=df_sorted,
                label_indices=train_label_rows,
                max_seq_len=MAX_SEQ_LEN,
                max_ctx_num=MAX_CTX_NUM,
                ctx_col=CTX_COL,
                
            )

            test_ds = GeoLifeUnifiedDatasetFast(
                preprocess_cache_dir=os.path.join(args.cache_dir, run_id),
                preprocess_num_workers=(args.preprocess_workers if args.preprocess_workers>0 else None),
                preprocess_force_rebuild=args.force_rebuild_cache,
                df=df_sorted,
                label_indices=test_label_rows,
                max_seq_len=MAX_SEQ_LEN,
                max_ctx_num=MAX_CTX_NUM,
                ctx_col=CTX_COL,
               
            )
            if preprocess_only:
                if (not is_ddp()) or ddp_rank() == 0:
                    print(f"[Preprocess-only] Cache built for {run_id} at {os.path.join(args.cache_dir, run_id)}")
                continue

            df_train_stats = build_df_train_segment(df_sorted, user2pos, TRAIN_RATIO, MAX_SEQ_LEN)

            user_id_to_idx = train_ds.user_id_to_idx
            idx_to_user_id = train_ds.idx_to_user_id 
            grid_id_to_idx = train_ds.grid_id_to_idx
            idx_to_grid_id = train_ds.idx_to_grid_id

            print("[Mapping] num_users =", len(user_id_to_idx), "num_grids =", len(grid_id_to_idx))

            if is_ddp():
                train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
                test_sampler  = DistributedSampler(test_ds, shuffle=False, drop_last=False)

                train_loader = DataLoader(
                    train_ds,
                    batch_size=BATCH_SIZE,
                    sampler=train_sampler,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=True,
                    persistent_workers=(NUM_WORKERS > 0),
                )
                test_loader = DataLoader(
                    test_ds,
                    batch_size=BATCH_SIZE,
                    sampler=test_sampler,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=True,
                    persistent_workers=(NUM_WORKERS > 0),
                )
            else:
                train_loader, train_sampler = make_dataloader_user_shuffle_fast(
                    train_ds, batch_size=BATCH_SIZE, shuffle_users=True, seed=SEED, num_workers=NUM_WORKERS
                )
                test_loader, _ = make_dataloader_user_shuffle_fast(
                    test_ds, batch_size=BATCH_SIZE, shuffle_users=False, seed=SEED, num_workers=NUM_WORKERS
                )

            print("train samples:", len(train_ds), "test samples:", len(test_ds))
            print("num_users:", train_ds.num_users, "num_grids:", train_ds.num_grids)

            if ddp_rank() == 0:
                print(f"=== Run: {experiments_name} (context_mode={contextGate}) ===")

            cfg = UnifiedConfig(
                num_users=train_ds.num_users,
                num_grids=train_ds.num_grids,
                max_seq_len=MAX_SEQ_LEN,
                max_ctx_num=MAX_CTX_NUM,
                dist_type=DIST_TYPE,
            )

            model = UnifiedTimeLocationModel(cfg).to(DEVICE)
            if is_ddp():
                from torch.nn.parallel import DistributedDataParallel as DDP
                local_rank = int(os.environ["LOCAL_RANK"])
                model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True,)
            optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            checkpoint_dir = f"./runs/unified_run_{run_id}/checkpoints/"
            log_file = f"./runs/unified_run_{run_id}/train_log.csv"
            curves_prefix = f"./runs/unified_run_{run_id}/plots/curves.png"

            Training_Unified(
                train_loader=train_loader,
                val_loader=test_loader,
                device=DEVICE,
                model=model,
                optimizer=optimizer,
                df_train=df_train_stats,
                topKs=(1,5,10),
                checkpoint_dir=checkpoint_dir,
                log_file=log_file,
                curves_prefix=curves_prefix,
                EPOCHS=EPOCHS,
                grad_clip=1.0,
                use_amp=True,
                val_max_batches=None,
                context_mode=context_mode,       
                ablate_user=ablate_user,
                ablate_time=ablate_time,
            )

            if ddp_rank() == 0:
                print(f"Completed run: {run_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")



    ddp_barrier()
    ddp_cleanup()

if __name__ == "__main__":
    main()
