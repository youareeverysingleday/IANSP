

import os
import argparse

import gzip
import pickle
import time

def load_pickle_gz(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

import pickle
import gzip
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


import torch.nn as nn

class SafeEmbedding(nn.Embedding):

    def forward(self, input):  
        if torch.is_tensor(input):
            if input.dtype not in (torch.int32, torch.int64):
                input = input.to(torch.int64)
            input = input.clamp_(0, self.num_embeddings - 1)
        return super().forward(input)

class SafeEmbeddingBag(nn.EmbeddingBag):
    def forward(self, input, offsets=None, per_sample_weights=None): 
        if torch.is_tensor(input):
            if input.dtype not in (torch.int32, torch.int64):
                input = input.to(torch.int64)
            input = input.clamp_(0, self.num_embeddings - 1)
        return super().forward(input, offsets, per_sample_weights)

def replace_embeddings_with_safe(module: nn.Module) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Embedding) and not isinstance(child, SafeEmbedding):
            safe = SafeEmbedding(
                num_embeddings=child.num_embeddings,
                embedding_dim=child.embedding_dim,
                padding_idx=child.padding_idx,
                max_norm=child.max_norm,
                norm_type=child.norm_type,
                scale_grad_by_freq=child.scale_grad_by_freq,
                sparse=child.sparse,
                _weight=child.weight
            )
            setattr(module, name, safe)
            count += 1
        elif isinstance(child, nn.EmbeddingBag) and not isinstance(child, SafeEmbeddingBag):
            safe = SafeEmbeddingBag(
                num_embeddings=child.num_embeddings,
                embedding_dim=child.embedding_dim,
                max_norm=child.max_norm,
                norm_type=child.norm_type,
                scale_grad_by_freq=child.scale_grad_by_freq,
                mode=child.mode,
                sparse=child.sparse,
                include_last_offset=child.include_last_offset,
                padding_idx=getattr(child, "padding_idx", None),
                _weight=child.weight,
            )
            setattr(module, name, safe)
            count += 1
        else:
            count += replace_embeddings_with_safe(child)
    return count

def is_ddp():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

def ddp_setup(backend="nccl"):
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def ddp_rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

def ddp_world():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def _load_pickle(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

class CachedGeoLifeDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dict):
        self.max_seq_len = cache_dict["max_seq_len"]
        self.max_ctx_num = cache_dict["max_ctx_num"]
        self.num_users = cache_dict["num_users"]
        self.num_grids = cache_dict["num_grids"]
        self.user_id_to_idx = cache_dict["user_id_to_idx"]
        self.grid_id_to_idx = cache_dict["grid_id_to_idx"]
        self.idx_to_user_id = cache_dict["idx_to_user_id"]
        self.idx_to_grid_id = cache_dict["idx_to_grid_id"]
        self.samples = cache_dict["samples"]
        self.user_data = cache_dict["user_data"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        import math, bisect
        uidx, end_pos = self.samples[idx]
        d = self.user_data[uidx]

        start_pos = max(0, end_pos - self.max_seq_len + 1)
        hist_len = end_pos - start_pos + 1
        sl = slice(start_pos, end_pos + 1)

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


GeoLife_All_PATH = "./Data/GeoLife_all.csv"
GeoLife_Routine_PATH = "./Data/GeoLife_Routine_top3.csv"
GeoLife_NonRoutine_PATH = "./Data/GeoLife_Nonroutine_top3.csv"

MoreUser_All_PATH = "./Data/MoreUser_all.csv"
MoreUser_Routine_PATH = "./Data/MoreUser_Routine_top3.csv"
MoreUser_NonRoutine_PATH = "./Data/MoreUser_Nonroutine_top3.csv"


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

DIST_TYPE = "student_t" # "lognormal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--experiment", type=str, default=None)
    ap.add_argument("--variant_tag", type=str, default=None,
                    help="For E12/E13/E14: train a specific variant tag (e.g. E12__moreuser_S1_fixedTotal_...). If not set, auto-train all variants found in cache_dir.")
    ap.add_argument("--datachoose", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--ddp_backend", type=str, default=os.environ.get("DDP_BACKEND", "nccl"),
                    help="DDP backend. On ROCm/Hygon this is typically 'nccl' (RCCL underneath).")
    ap.add_argument("--force_math_sdpa", action="store_true", default=True,
                    help="Force SDPA to use math kernel (disable flash/mem_efficient). Helps avoid ROCm/HSA VMFaults.")
    ap.add_argument("--no_force_math_sdpa", action="store_false", dest="force_math_sdpa",
                    help="Do not force SDPA math kernel.")
    ap.add_argument("--amp", action="store_true", default=None, help="Enable AMP (override NSP.use_amp).")
    ap.add_argument("--no_amp", action="store_true", default=False, help="Disable AMP explicitly.")
    ap.add_argument("--safe_mode", action="store_true", default=True,
                    help="Enable extra safety switches for ROCm/Hygon (disable fused/flash kernels, prefer deterministic).")
    ap.add_argument("--no_safe_mode", action="store_false", dest="safe_mode",
                    help="Disable safety switches.")
    ap.add_argument("--launch_blocking", action="store_true", default=False,
                    help="Set HIP_LAUNCH_BLOCKING=1 for easier debugging (slower).")
    args = ap.parse_args()

    if torch.cuda.is_available():
        if is_ddp():
            local_rank = ddp_setup(backend=args.ddp_backend)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.force_math_sdpa and device.type == "cuda":
        try:
            from torch.backends.cuda import sdp_kernel
            sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except Exception:
            try:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception:
                pass

    if args.safe_mode:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass

    if args.launch_blocking:
        os.environ["HIP_LAUNCH_BLOCKING"] = "1"

    import NextStayPrediction as NSP
    if args.experiment is not None:
        NSP.experiments_name = args.experiment
    if args.datachoose is not None:
        NSP.DATACHOOSE = args.datachoose

    experiments_name = NSP.experiments_name
    DATACHOOSE = NSP.DATACHOOSE

    use_amp = getattr(NSP, "use_amp", True)
    if args.safe_mode:
        use_amp = False
    if args.amp is True:
        use_amp = True
    if args.no_amp:
        use_amp = False

    from pathlib import Path
    cache_dir = Path(args.cache_dir)

    def discover_variant_tags() -> list[str]:
        if args.variant_tag:
            return [args.variant_tag]
        if experiments_name in ["E12", "E13", "E14"]:
            pattern = f"{DATACHOOSE}_{experiments_name}__*_train_cache.pkl.gz"
            files = sorted(cache_dir.glob(pattern))
            tags: list[str] = []
            suffix = "_train_cache.pkl.gz"
            prefix = f"{DATACHOOSE}_"
            for p in files:
                name = p.name
                if not name.endswith(suffix):
                    continue
                core = name[:-len(suffix)]
                if core.startswith(prefix):
                    core = core[len(prefix):]
                tags.append(core)
            return tags
        return [experiments_name]

    variant_tags = discover_variant_tags()
    if len(variant_tags) == 0:
        raise FileNotFoundError(
            f"No cache variants found for experiment={experiments_name} in {cache_dir}. "
            f"Expected caches like {DATACHOOSE}_{experiments_name}__*_train_cache.pkl.gz"
        )

    if ddp_rank() == 0:
        print(f"[variant] will train {len(variant_tags)} variant(s):")
        for t in variant_tags:
            print(f"  - {t}")

    for vi, variant_tag in enumerate(variant_tags, start=1):
        train_cache_path = cache_dir / f"{DATACHOOSE}_{variant_tag}_train_cache.pkl.gz"
        test_cache_path  = cache_dir / f"{DATACHOOSE}_{variant_tag}_test_cache.pkl.gz"

        # backward compat for E1..E11 caches (no __ tag)
        if (not train_cache_path.exists()) and (variant_tag == experiments_name):
            train_cache_path = cache_dir / f"{DATACHOOSE}_{experiments_name}_train_cache.pkl.gz"
            test_cache_path  = cache_dir / f"{DATACHOOSE}_{experiments_name}_test_cache.pkl.gz"

        if not train_cache_path.exists() or not test_cache_path.exists():
            raise FileNotFoundError(
                f"Cache not found for variant '{variant_tag}'.\n{train_cache_path}\n{test_cache_path}\n"
                f"Run preprocess_cpu_cache_with_progress_patched.py first."
            )

        if ddp_rank() == 0:
            print(f"[{vi}/{len(variant_tags)}] [cache] loading: {train_cache_path}   {test_cache_path}")

        train_cache = load_pickle_gz(train_cache_path)
        test_cache  = load_pickle_gz(test_cache_path)
        train_ds = CachedGeoLifeDataset(train_cache)
        test_ds  = CachedGeoLifeDataset(test_cache)

        BATCH_SIZE = args.batch_size or getattr(NSP, "BATCH_SIZE", 64)

        if is_ddp():
            train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
            test_sampler  = DistributedSampler(test_ds, shuffle=False, drop_last=False)
        else:
            train_sampler = None
            test_sampler  = None

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            sampler=test_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            collate_fn=collate_fn,
        )

        run_dir = Path("runs") / f"unified_run_{variant_tag}"
        ckpt_dir = run_dir / "checkpoints"
        plots_dir = run_dir / "plots"
        log_file = run_dir / "train_log.csv"
        curves_prefix = plots_dir / "curves.png"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        DIST_TYPE = getattr(NSP, "DIST_TYPE", "student_t")
        cfg = NSP.UnifiedConfig(
            num_users=train_cache["num_users"],
            num_grids=train_cache["num_grids"],
            max_seq_len=train_cache.get("max_seq_len", MAX_SEQ_LEN),
            max_ctx_num=train_cache.get("max_ctx_num", MAX_CTX_NUM),
            dist_type=DIST_TYPE,
        )

        model = NSP.UnifiedTimeLocationModel(cfg)

        replaced = replace_embeddings_with_safe(model)
        if replaced > 0 and (not is_ddp() or ddp_rank() == 0):
            print(f"[debug] safe-embedding enabled: replaced {replaced} embedding modules")

        model = model.to(device)
        if is_ddp():
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=True,  # safer for gated branches
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=getattr(NSP, "LR", 1e-4),
            weight_decay=getattr(NSP, "WEIGHT_DECAY", 0.01),
        )
        NSP.Training_Unified(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            optimizer=optimizer,
            device=str(device),
            use_amp=use_amp,
            EPOCHS=getattr(NSP, "EPOCHS", 10),
            log_file=str(log_file),
            checkpoint_dir=str(ckpt_dir),
            curves_prefix=str(curves_prefix),
        )

        # ---- per-variant cleanup ----
        del train_cache, test_cache, train_ds, test_ds, train_loader, test_loader, optimizer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ddp_cleanup()


if __name__ == "__main__":
    main()