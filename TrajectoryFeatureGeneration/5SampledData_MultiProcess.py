

import os
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


def _infer_columns(df: pd.DataFrame, user_col: str | None, time_col: str | None):
    if user_col is None:
        for c in ["user_id", "userID", "uid", "UserId", "user"]:
            if c in df.columns:
                user_col = c
                break
    if time_col is None:
        for c in ["timestamp", "time", "datetime", "date_time", "t"]:
            if c in df.columns:
                time_col = c
                break
    if user_col is None or time_col is None:
        raise ValueError(f"The user/time column cannot be automatically recognized; \
                         please pass it explicitly. Current column name:{list(df.columns)}")
    return user_col, time_col


def _ensure_sorted(df: pd.DataFrame, user_col: str, time_col: str) -> pd.DataFrame:
    return df.sort_values([user_col, time_col]).reset_index(drop=True)


def _save_df(df: pd.DataFrame, save_dir: str, filename: str):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = Path(save_dir) / filename
    df.to_csv(path, index=False)
    print(f"[Saved] {path}")


def _sample_users_with_min_stays(counts: pd.Series,
                                 U: int,
                                 min_stays: int,
                                 rng: np.random.Generator) -> np.ndarray:
    eligible = counts[counts >= min_stays].index.to_numpy()
    if len(eligible) < U:
        raise ValueError(f"There are only {len(eligible)} users who satisfy min_stays={min_stays},\
                          which is insufficient.U={U}")
    return rng.choice(eligible, size=U, replace=False)

def _worker_take_idx(args):
    uid, L, idx, slice_mode, seed = args
    rng = np.random.default_rng(seed)

    n = len(idx)
    if n < L:
        return np.empty(0, dtype=idx.dtype)

    if slice_mode == "prefix":
        return idx[:L]

    if slice_mode == "random_contiguous":
        if n == L:
            return idx
        start = rng.integers(0, n - L + 1)
        return idx[start:start + L]

    raise ValueError("slice_mode must be 'random_contiguous' or 'prefix'")


def _parallel_collect_indices(sampled_users: np.ndarray,
                              lengths: np.ndarray,
                              user2idx: dict,
                              slice_mode: str,
                              base_seed: int,
                              n_jobs: int | None):

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 1) - 1)

    def stable_uid_seed(u):
        s = str(u).encode("utf-8")
        h = 0
        for b in s:
            h = (h * 131 + b) % (2**32)
        return (base_seed + h) % (2**32)

    tasks = []
    for uid, L in zip(sampled_users, lengths):
        idx = user2idx[uid]
        seed = stable_uid_seed(uid)
        tasks.append((uid, int(L), idx, slice_mode, seed))

    if n_jobs == 1:
        out = [_worker_take_idx(t) for t in tasks]
        return out

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        out = list(ex.map(_worker_take_idx, tasks, chunksize=64))
    return out


def sample_fixed_total_stays_mp(
    csv_path: str,
    U: int,
    N_total: int = 1_000_000,
    min_stays: int = 65,
    user_col: str | None = None,
    time_col: str | None = None,
    random_state: int = 42,
    slice_mode: str = "random_contiguous",  # or "prefix"
    save_dir: str = "./samples",
    n_jobs: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df = pd.read_csv(csv_path)
    user_col, time_col = _infer_columns(df, user_col, time_col)
    df = _ensure_sorted(df, user_col, time_col)

    user2idx = df.groupby(user_col).indices  # dict: uid -> np.ndarray(row indices)
    counts = pd.Series({u: len(ix) for u, ix in user2idx.items()})

    base = N_total // U
    rem = N_total - base * U
    lengths = np.full(U, base, dtype=int)
    if rem > 0:
        lengths[:rem] += 1
    rng.shuffle(lengths)  

    per_user_min_needed = max(min_stays, base + (1 if rem > 0 else 0))
    sampled_users = _sample_users_with_min_stays(counts, U, per_user_min_needed, rng)

    idx_list = _parallel_collect_indices(
        sampled_users=sampled_users,
        lengths=lengths,
        user2idx=user2idx,
        slice_mode=slice_mode,
        base_seed=random_state,
        n_jobs=n_jobs,
    )

    all_idx = np.concatenate(idx_list)
    out = df.loc[all_idx].copy()
    out = _ensure_sorted(out, user_col, time_col)

    if len(out) != N_total:
        out = out.iloc[:N_total].copy()
        out = _ensure_sorted(out, user_col, time_col)

    filename = f"geolife_S1_fixedTotal_N{N_total}_U{U}_min{min_stays}_seed{random_state}_{slice_mode}_mp{n_jobs or 'auto'}.csv"
    _save_df(out, save_dir, filename)
    return out

def sample_fixed_per_user_stays_mp(
    csv_path: str,
    U: int,
    L: int = 500,
    user_col: str | None = None,
    time_col: str | None = None,
    random_state: int = 42,
    slice_mode: str = "random_contiguous",
    save_dir: str = "./samples",
    n_jobs: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df = pd.read_csv(csv_path)
    user_col, time_col = _infer_columns(df, user_col, time_col)
    df = _ensure_sorted(df, user_col, time_col)

    user2idx = df.groupby(user_col).indices
    counts = pd.Series({u: len(ix) for u, ix in user2idx.items()})
    sampled_users = _sample_users_with_min_stays(counts, U, L, rng)

    lengths = np.full(U, L, dtype=int)

    idx_list = _parallel_collect_indices(
        sampled_users=sampled_users,
        lengths=lengths,
        user2idx=user2idx,
        slice_mode=slice_mode,
        base_seed=random_state,
        n_jobs=n_jobs,
    )

    out = df.loc[np.concatenate(idx_list)].copy()
    out = _ensure_sorted(out, user_col, time_col)

    filename = f"geolife_S2_fixedPerUser_U{U}_L{L}_seed{random_state}_{slice_mode}_mp{n_jobs or 'auto'}.csv"
    _save_df(out, save_dir, filename)
    return out


def sample_fixed_user_count_mp(
    csv_path: str,
    U: int = 2000,
    L: int = 500,
    min_stays: int | None = None,
    user_col: str | None = None,
    time_col: str | None = None,
    random_state: int = 42,
    slice_mode: str = "random_contiguous",
    save_dir: str = "./samples",
    n_jobs: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df = pd.read_csv(csv_path)
    user_col, time_col = _infer_columns(df, user_col, time_col)
    df = _ensure_sorted(df, user_col, time_col)

    user2idx = df.groupby(user_col).indices
    counts = pd.Series({u: len(ix) for u, ix in user2idx.items()})

    need = max(L, (min_stays if min_stays is not None else 0))
    sampled_users = _sample_users_with_min_stays(counts, U, need, rng)
    lengths = np.full(U, L, dtype=int)

    idx_list = _parallel_collect_indices(
        sampled_users=sampled_users,
        lengths=lengths,
        user2idx=user2idx,
        slice_mode=slice_mode,
        base_seed=random_state,
        n_jobs=n_jobs,
    )

    out = df.loc[np.concatenate(idx_list)].copy()
    out = _ensure_sorted(out, user_col, time_col)

    filename = f"geolife_S3_fixedUsers_U{U}_L{L}_seed{random_state}_{slice_mode}_mp{n_jobs or 'auto'}.csv"
    _save_df(out, save_dir, filename)
    return out



def run_all_sampling_grids_mp(
    csv_path: str,
    save_dir: str,
    random_state: int = 42,
    slice_mode: str = "random_contiguous",
    n_jobs: int | None = None,
):
    U_list_1 = [1000, 2000, 4000, 8000, 9000]

    for U in U_list_1:
        sample_fixed_total_stays_mp(
            csv_path=csv_path, U=U, N_total=1_000_000, min_stays=65,
            user_col="userID",time_col="stime",
            random_state=random_state, slice_mode=slice_mode,
            save_dir=save_dir, n_jobs=n_jobs
        )

    U_list_2 = [500, 1000, 2000, 4000, 6000]

    for U in U_list_2:
        sample_fixed_per_user_stays_mp(
            csv_path=csv_path, U=U, L=500,
            user_col="userID",time_col="stime",
            random_state=random_state, slice_mode=slice_mode,
            save_dir=save_dir, n_jobs=n_jobs
        )

    for L in [100, 200, 500, 800, 1000]:
        sample_fixed_user_count_mp(
            csv_path=csv_path, U=2000, L=L,
            user_col="userID",time_col="stime",
            random_state=random_state, slice_mode=slice_mode,
            save_dir=save_dir, n_jobs=n_jobs
        )


if __name__ == "__main__":
    csv_path = "./Data/MoreUser/all.csv"
    run_all_sampling_grids_mp(
        csv_path=csv_path,
        save_dir="./Data/MoreUser/Sampled/",
        random_state=42,
        slice_mode="random_contiguous",
        n_jobs=None, 
    )
