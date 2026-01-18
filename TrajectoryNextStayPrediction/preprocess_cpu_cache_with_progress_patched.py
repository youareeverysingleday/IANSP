
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""

os.environ.setdefault("NSP_QUIET_IMPORT", "1")  # silence noisy prints when NextStayPrediction is imported
import argparse
import gzip
import pickle
import time
from pathlib import Path

from tqdm import tqdm
import NextStayPrediction as NSP

print("[preprocess] script path =", os.path.abspath(__file__))

def _save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)



def _iter_dataset_variants(NSP):

    import os, glob

    DATACHOOSE = getattr(NSP, "DATACHOOSE", "GeoLife")
    exp = getattr(NSP, "experiments_name", "E1")

    def list_csvs(d):
        return sorted(glob.glob(os.path.join(d, "*.csv")))

    if DATACHOOSE == "GeoLife":
        if exp in ["E1","E2","E3","E4","E5","E6","E7"]:
            yield (exp, NSP.GeoLife_All_PATH)
        elif exp in ["E8","E10"]:
            yield (exp, NSP.GeoLife_Routine_PATH)
        elif exp in ["E9","E11"]:
            yield (exp, NSP.GeoLife_NonRoutine_PATH)
        elif exp in ["E12","E13","E14"]:
            base = os.path.join(getattr(NSP, "GeoLife_Sampled_PATH", "./Data/GeoLifeSampled/"), exp)
            files = list_csvs(base)
            if len(files) == 0:
                raise FileNotFoundError(f"No sampled CSV found under: {base}")
            for f in files:
                stem = os.path.splitext(os.path.basename(f))[0]
                yield (f"{exp}__{stem}", f)
        else:
            raise ValueError(f"Unknown experiments_name={exp} for GeoLife")

    elif DATACHOOSE == "MoreUser":
        if exp in ["E1","E2","E3","E4","E5","E6","E7"]:
            yield (exp, NSP.MoreUser_All_PATH)
        elif exp in ["E8","E10"]:
            yield (exp, NSP.MoreUser_Routine_PATH)
        elif exp in ["E9","E11"]:
            yield (exp, NSP.MoreUser_NonRoutine_PATH)
        elif exp in ["E12","E13","E14"]:
            base = os.path.join(getattr(NSP, "MoreUser_Sampled_PATH", "./Data/MoreUserSampled/"), exp)
            files = list_csvs(base)
            if len(files) == 0:
                raise FileNotFoundError(f"No sampled CSV found under: {base}")
            for f in files:
                stem = os.path.splitext(os.path.basename(f))[0]
                yield (f"{exp}__{stem}", f)
        else:
            raise ValueError(f"Unknown experiments_name={exp} for MoreUser")
    else:
        raise ValueError(f"DATACHOOSE must be GeoLife or MoreUser, got {DATACHOOSE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default="./cache_geolife", help="output cache dir")
    ap.add_argument("--workers", type=int, default=0,
                    help="CPU worker processes for preprocessing. 0/negative = auto (cpu_count-1).")
    ap.add_argument("--no_progress", action="store_true",
                    help="Disable progress bar output.")
    ap.add_argument("--experiment", type=str, default=None, help="override experiments_name (e.g. E1)")
    ap.add_argument("--datachoose", type=str, default=None, help="override DATACHOOSE (GeoLife/MoreUser)")
    ap.add_argument("--force", action="store_true", help="overwrite cache if exists")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cpu_total = os.cpu_count() or 1
    preprocess_workers = (max(1, cpu_total - 1) if int(args.workers) <= 0 else int(args.workers))
    show_progress = (not args.no_progress)
    print(f"[preprocess] cpu_total={cpu_total} preprocess_workers={preprocess_workers}")

    pbar = None
    if show_progress:
        pbar = tqdm(total=6, desc="preprocess", unit="step")
    def _step(msg: str):
        if msg:
            print(msg)
        if pbar is not None:
            pbar.update(1)
    t0_all = time.time()

    _step("[1/6] imported NextStayPrediction (GPU hidden)")

    if args.experiment is not None:
        NSP.experiments_name = args.experiment
    if args.datachoose is not None:
        NSP.DATACHOOSE = args.datachoose

    DATACHOOSE = getattr(NSP, "DATACHOOSE", "MoreUser")
    experiments_name = getattr(NSP, "experiments_name", "E1")

    import pandas as pd

    variants = list(_iter_dataset_variants(NSP))
    print(f"[preprocess] variants = {len(variants)}")

    exp_cfg = getattr(NSP, "Experiments", {}).get(experiments_name, {})
    context_gate = exp_cfg.get("context", "none")
    ctx_col = None
    if context_gate in ("fuzzy", "exact"):
        ctx_col = "context_fuzzy" if context_gate == "fuzzy" else "context_precise"

    TRAIN_RATIO = 1.0 - float(getattr(NSP, "TEST_RATIO", 0.2))
    MAX_SEQ_LEN = int(getattr(NSP, "MAX_SEQ_LEN", 64))
    MAX_CTX_NUM = int(getattr(NSP, "MAX_CTX_NUM", 16))
    if ctx_col is None:
        MAX_CTX_NUM = 0

    for i, (variant_tag, csv_path) in enumerate(variants, start=1):
        print(f"\n[preprocess] ===== ({i}/{len(variants)}) variant={variant_tag} =====")
        train_cache = cache_dir / f"{DATACHOOSE}_{variant_tag}_train_cache.pkl.gz"
        test_cache  = cache_dir / f"{DATACHOOSE}_{variant_tag}_test_cache.pkl.gz"

        if train_cache.exists() and test_cache.exists() and not args.force:
            print(f"[preprocess] cache exists, skip.\n  {train_cache}\n  {test_cache}")
            continue

        t0_variant = time.time()

        print(f"[preprocess] loading df: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"[preprocess] loaded df: rows={len(df):,} cols={len(df.columns)}")

        t0 = time.time()
        df_sorted, user2pos = NSP.build_user_index(df, user_col="userID", time_col="stime")
        print(f"[preprocess] built user index in {time.time()-t0:.1f}s | users={len(user2pos):,}")

        # Split labels
        t0 = time.time()
        train_label_rows, test_label_rows = NSP.split_labels_with_history(
            user2pos, train_ratio=TRAIN_RATIO, seq_len=MAX_SEQ_LEN
        )
        print(f"[preprocess] split labels in {time.time()-t0:.1f}s | train={len(train_label_rows):,} test={len(test_label_rows):,}")

        ds_kwargs = dict(
            df=df_sorted,
            preprocess_num_workers=preprocess_workers,
            max_seq_len=MAX_SEQ_LEN,
            max_ctx_num=MAX_CTX_NUM,
            ctx_col=ctx_col,
            show_progress=show_progress,   
            user_col="userID",
            time_col="stime",
            grid_col="grid",
            dur_col="duration",
        )

        print(f"[preprocess] building train dataset ... (context={'off' if ctx_col is None else ctx_col})")
        t0 = time.time()
        train_ds = NSP.GeoLifeUnifiedDatasetFast(label_indices=train_label_rows, **ds_kwargs)
        print(f"[preprocess] train dataset built in {time.time()-t0:.1f}s | samples={len(train_ds):,} users={train_ds.num_users:,} grids={train_ds.num_grids:,}")

        print(f"[preprocess] building test dataset ...")
        t0 = time.time()
        test_ds = NSP.GeoLifeUnifiedDatasetFast(label_indices=test_label_rows, **ds_kwargs)
        print(f"[preprocess] test dataset built in {time.time()-t0:.1f}s | samples={len(test_ds):,}")

        def pack(ds):
            return {
                "max_seq_len": ds.max_seq_len,
                "max_ctx_num": ds.max_ctx_num,
                "num_users": ds.num_users,
                "num_grids": ds.num_grids,
                "user_id_to_idx": ds.user_id_to_idx,
                "grid_id_to_idx": ds.grid_id_to_idx,
                "idx_to_user_id": ds.idx_to_user_id,
                "idx_to_grid_id": ds.idx_to_grid_id,
                "samples": ds.samples,
                "user_data": ds.user_data,
                "DATACHOOSE": DATACHOOSE,
                "experiments_name": experiments_name,
                "variant_tag": variant_tag,
                "CTX_COL": ctx_col,
            }

        print("[preprocess] saving caches ...")
        _save_pickle(pack(train_ds), train_cache)
        _save_pickle(pack(test_ds), test_cache)
        print(f"[preprocess] saved:\n  train: {train_cache}\n  test : {test_cache}")
        print(f"[preprocess] variant time: {time.time()-t0_variant:.1f}s")

    print(f"\n[preprocess] all done. total time: {time.time()-t0_all:.1f}s")
    if pbar is not None:
        pbar.close()


if __name__ == "__main__":
    main()
