
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
    SCIPY_OK = True
except Exception as e:
    SCIPY_OK = False
    stats = None


def compute_gaps_minutes(df: pd.DataFrame,
                         user_col="userID",
                         stime_col="stime",
                         etime_col="etime") -> pd.Series:
    df = df[[user_col, stime_col, etime_col]].copy()
    df[stime_col] = pd.to_datetime(df[stime_col], errors="coerce")
    df[etime_col] = pd.to_datetime(df[etime_col], errors="coerce")
    df = df.dropna(subset=[user_col, stime_col, etime_col])
    df[user_col] = df[user_col].astype(int)

    df = df.sort_values([user_col, stime_col]).reset_index(drop=True)

    df["next_stime"] = df.groupby(user_col, sort=False)[stime_col].shift(-1)

    gap = (df["next_stime"] - df[etime_col]).dt.total_seconds() / 60.0  # minutes

    gap = gap.replace([np.inf, -np.inf], np.nan).dropna()

    return gap


def safe_log(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:

    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0.0)
    return np.log(x + eps)


def fit_distributions(raw_gaps_min: np.ndarray,
                      log_gaps: np.ndarray):

    if not SCIPY_OK:
        raise RuntimeError("SciPy is not available.")

    params = {}

    mu, sigma = stats.norm.fit(log_gaps)
    params["normal_loggap"] = {"mu": float(mu), "sigma": float(sigma)}

    df_t, loc_t, scale_t = stats.t.fit(log_gaps)
    params["studentt_loggap"] = {"nu": float(df_t), "loc": float(loc_t), "scale": float(scale_t)}

    raw_pos = raw_gaps_min[np.isfinite(raw_gaps_min) & (raw_gaps_min > 0)]

    if raw_pos.size < 50:
        params["lognorm_rawgap"] = None
        print("[Warn] Too few positive gaps for LogNormal fit. Skip lognorm.")
    else:
        s, loc_lg, scale_lg = stats.lognorm.fit(raw_pos, floc=0)
        params["lognorm_rawgap"] = {"shape": float(s), "loc": float(loc_lg), "scale": float(scale_lg)}

    return params



def qqplot_student_t(ax, log_gaps: np.ndarray, nu: float, loc: float, scale: float, max_points: int = 20000):

    log_gaps = np.asarray(log_gaps, dtype=np.float64)
    log_gaps = log_gaps[np.isfinite(log_gaps)]
    log_gaps = np.sort(log_gaps)

    if log_gaps.size == 0:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", fontsize=9)
        return

    if log_gaps.size > max_points:
        idx = np.linspace(0, log_gaps.size - 1, max_points).astype(int)
        log_gaps = log_gaps[idx]

    n = log_gaps.size
    p = (np.arange(1, n + 1) - 0.5) / n
    theo = stats.t.ppf(p, df=nu, loc=loc, scale=scale)

    ax.scatter(theo, log_gaps, s=10, facecolors="none", edgecolors="black", linewidths=0.6, rasterized=True)

    q1, q2 = int(0.25 * n), int(0.75 * n)
    slope = (log_gaps[q2] - log_gaps[q1]) / (theo[q2] - theo[q1] + 1e-12)
    intercept = log_gaps[q1] - slope * theo[q1]
    ax.plot(theo, slope * theo + intercept, linewidth=2.0, linestyle="-", color="tab:red")

    ax.set_xlabel("Theoretical quantiles", fontsize=9)
    ax.set_ylabel("Ordered values", fontsize=9)


def plot_hist_with_fits(raw_gaps_min: np.ndarray,
                        log_gaps: np.ndarray,
                        params: dict,
                        out_dir: str,
                        bins_raw: int = 200,
                        bins_log: int = 200):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(7.2, 4.2))
    plt.hist(raw_gaps_min, bins=bins_raw, density=True)
    plt.xlabel("Inter-stay gap (minutes)")
    plt.ylabel("Density")
    plt.title("Raw inter-stay gaps (minutes)")

    if SCIPY_OK and params is not None and "lognorm_rawgap" in params:
        s = params["lognorm_rawgap"]["shape"]
        loc = params["lognorm_rawgap"]["loc"]
        scale = params["lognorm_rawgap"]["scale"]
        x_max = np.percentile(raw_gaps_min, 99.5)
        xs = np.linspace(0, max(1e-6, x_max), 800)
        pdf = stats.lognorm.pdf(xs, s=s, loc=loc, scale=scale)
        plt.plot(xs, pdf)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_rawgap_with_lognorm.pdf"), dpi=300)
    plt.savefig(os.path.join(out_dir, "hist_rawgap_with_lognorm.png"), dpi=300)
    plt.close()
    plt.figure(figsize=(7.2, 4.2))
    plt.hist(log_gaps, bins=bins_log, density=True)
    plt.xlabel("log(gap_minutes + eps)")
    plt.ylabel("Density")
    plt.title("Log inter-stay gaps")

    if SCIPY_OK and params is not None:
        x_min, x_max = np.percentile(log_gaps, [0.5, 99.5])
        xs = np.linspace(x_min, x_max, 800)

        mu = params["normal_loggap"]["mu"]
        sigma = params["normal_loggap"]["sigma"]
        plt.plot(xs, stats.norm.pdf(xs, loc=mu, scale=sigma))

        nu = params["studentt_loggap"]["nu"]
        loc = params["studentt_loggap"]["loc"]
        scale = params["studentt_loggap"]["scale"]
        plt.plot(xs, stats.t.pdf(xs, df=nu, loc=loc, scale=scale))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_loggap_with_normal_studentt.pdf"), dpi=300)
    plt.savefig(os.path.join(out_dir, "hist_loggap_with_normal_studentt.png"), dpi=300)
    plt.close()


def qq_plots(raw_gaps_min: np.ndarray,
             log_gaps: np.ndarray,
             params: dict,
             out_dir: str):
    if not SCIPY_OK:
        raise RuntimeError("SciPy is not available. Please install scipy to use QQ plots.")
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(5.2, 5.2))
    mu = params["normal_loggap"]["mu"]
    sigma = params["normal_loggap"]["sigma"]
    stats.probplot(log_gaps, dist=stats.norm, sparams=(mu, sigma), plot=plt)
    plt.title("QQ-Plot: log gaps vs Normal")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "qq_loggap_normal.pdf"), dpi=300)
    plt.savefig(os.path.join(out_dir, "qq_loggap_normal.png"), dpi=300)
    plt.close()
    plt.figure(figsize=(5.2, 5.2))
    nu = params["studentt_loggap"]["nu"]
    loc = params["studentt_loggap"]["loc"]
    scale = params["studentt_loggap"]["scale"]
    ax = plt.gca()
    qqplot_student_t(ax, log_gaps, nu=nu, loc=loc, scale=scale, max_points=20000)
    plt.title("QQ-Plot: log gaps vs Student-t")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "qq_loggap_studentt.pdf"), dpi=300)
    plt.savefig(os.path.join(out_dir, "qq_loggap_studentt.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(5.2, 5.2))
    s = params["lognorm_rawgap"]["shape"]
    loc = params["lognorm_rawgap"]["loc"]
    scale = params["lognorm_rawgap"]["scale"]
    raw_pos = raw_gaps_min[raw_gaps_min > 0]
    if raw_pos.size >= 50:
        stats.probplot(raw_pos, dist=stats.lognorm, sparams=(s, loc, scale), plot=plt)
        plt.title("QQ-Plot: raw gaps (>0) vs LogNormal")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "qq_rawgap_lognorm.pdf"), dpi=300)
        plt.savefig(os.path.join(out_dir, "qq_rawgap_lognorm.png"), dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to GeoLife stays CSV containing userID, stime, etime")
    ap.add_argument("--out_dir", type=str, default="./gap_figs", help="Directory to save plots")
    ap.add_argument("--user_col", type=str, default="userID")
    ap.add_argument("--stime_col", type=str, default="stime")
    ap.add_argument("--etime_col", type=str, default="etime")
    ap.add_argument("--eps", type=float, default=1e-6, help="epsilon for log(gap+eps)")
    ap.add_argument("--min_gap_min", type=float, default=0.0, help="Filter: keep gaps >= this minutes (e.g., 0)")
    ap.add_argument("--max_gap_pct", type=float, default=99.9, help="Optional tail clip percentile (e.g., 99.9). Use 100 for no clip.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    gaps_min = compute_gaps_minutes(df, user_col=args.user_col, stime_col=args.stime_col, etime_col=args.etime_col)

    gaps_min = gaps_min[gaps_min >= args.min_gap_min]

    if args.max_gap_pct < 100.0:
        cap = np.percentile(gaps_min.values, args.max_gap_pct)
        gaps_min = gaps_min[gaps_min <= cap]

    raw = gaps_min.values.astype(np.float64)
    print(f"[Gap sign] <=0 ratio: {(raw<=0).mean():.4f}  "
        f"(zero: {(raw==0).mean():.4f}, neg: {(raw<0).mean():.4f})")
    logg = safe_log(raw, eps=args.eps)

    print(f"[Stats] #gaps = {raw.size}")
    print(f"[Stats] raw gaps minutes: mean={raw.mean():.3f}, median={np.median(raw):.3f}, p95={np.percentile(raw,95):.3f}, p99={np.percentile(raw,99):.3f}")
    print(f"[Stats] log gaps: mean={logg.mean():.3f}, std={logg.std():.3f}")

    if not SCIPY_OK:
        print("\n[Warn] SciPy not available. Install with: pip install scipy\n"
            "Will only plot histograms without MLE fits/QQ plots.")
        params = None
        os.makedirs(args.out_dir, exist_ok=True)
        plt.figure(figsize=(7.2, 4.2))
        plt.hist(raw, bins=200, density=True)
        plt.xlabel("Inter-stay gap (minutes)"); plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "hist_rawgap.pdf"), dpi=300)
        plt.savefig(os.path.join(args.out_dir, "hist_rawgap.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(7.2, 4.2))
        plt.hist(logg, bins=200, density=True)
        plt.xlabel("log(gap_minutes + eps)"); plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "hist_loggap.pdf"), dpi=300)
        plt.savefig(os.path.join(args.out_dir, "hist_loggap.png"), dpi=300)
        plt.close()
        return

    params = fit_distributions(raw_gaps_min=raw, log_gaps=logg)
    if params is not None and params.get("lognorm_rawgap") is not None:
        print("\n[MLE Parameters]")
        print("Normal on log-gaps:")
        print(f"  mu={params['normal_loggap']['mu']:.6f}, sigma={params['normal_loggap']['sigma']:.6f}")

        print("Student-t on log-gaps:")
        print(f"  nu(df)={params['studentt_loggap']['nu']:.6f}, loc={params['studentt_loggap']['loc']:.6f}, scale={params['studentt_loggap']['scale']:.6f}")

        print("LogNormal on raw gaps (minutes):")
        print(f"  shape(s)={params['lognorm_rawgap']['shape']:.6f}, loc={params['lognorm_rawgap']['loc']:.6f}, scale={params['lognorm_rawgap']['scale']:.6f}")

        # Plots
        plot_hist_with_fits(raw_gaps_min=raw, log_gaps=logg, params=params, out_dir=args.out_dir)
        qq_plots(raw_gaps_min=raw, log_gaps=logg, params=params, out_dir=args.out_dir)

        print(f"\n[Done] Figures saved to: {os.path.abspath(args.out_dir)}")
        print("  - hist_rawgap_with_lognorm.pdf")
        print("  - hist_loggap_with_normal_studentt.pdf")
        print("  - qq_loggap_normal.pdf")
        print("  - qq_loggap_studentt.pdf")
        print("  - qq_rawgap_lognorm.pdf (if enough positive gaps)")


if __name__ == "__main__":
    main()

