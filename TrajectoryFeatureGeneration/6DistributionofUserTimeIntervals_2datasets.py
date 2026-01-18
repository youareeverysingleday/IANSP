
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
    stats = None


def compute_gaps_minutes(df: pd.DataFrame,
                         user_col="userID",
                         stime_col="stime",
                         etime_col="etime") -> np.ndarray:
    df = df[[user_col, stime_col, etime_col]].copy()
    df[stime_col] = pd.to_datetime(df[stime_col], errors="coerce")
    df[etime_col] = pd.to_datetime(df[etime_col], errors="coerce")
    df = df.dropna(subset=[user_col, stime_col, etime_col])

    df = df.sort_values([user_col, stime_col]).reset_index(drop=True)
    df["next_stime"] = df.groupby(user_col, sort=False)[stime_col].shift(-1)
    gap = (df["next_stime"] - df[etime_col]).dt.total_seconds() / 60.0
    gap = gap.replace([np.inf, -np.inf], np.nan).dropna()
    return gap.values.astype(np.float64)


def safe_log(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:

    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0.0)
    return np.log(x + eps)


def fit_normal_studentt(log_gaps: np.ndarray):
    if not SCIPY_OK:
        raise RuntimeError("SciPy is required for distribution fitting/QQ plots.")
    mu, sigma = stats.norm.fit(log_gaps)
    nu, loc, scale = stats.t.fit(log_gaps)
    return (mu, sigma), (nu, loc, scale)


def qqplot_normal(ax, log_gaps: np.ndarray, mu: float, sigma: float):
    (osm, osr), (slope, intercept, r) = stats.probplot(
        log_gaps, dist=stats.norm, sparams=(mu, sigma)
    )
    ax.scatter(osm, osr, s=10, facecolors="none", edgecolors="black", linewidths=0.8, rasterized=True)
    ax.plot(osm, slope * osm + intercept, linewidth=2.0, linestyle="-", color="tab:red")
    ax.set_xlabel("Theoretical quantiles", fontsize=9)
    ax.set_ylabel("Ordered values", fontsize=9)

def hist_log_with_fits(ax,
                       log_gaps: np.ndarray,
                       mu: float,
                       sigma: float,
                       nu: float,
                       loc: float,
                       scale: float,
                       bins: int = 120):

    ax.hist(
        log_gaps,
        bins=bins,
        density=True,
        color="#D9D9D9",
        edgecolor="black",
        linewidth=0.4,
        rasterized=True,
    )

    x_min, x_max = np.percentile(log_gaps, [0.5, 99.5])
    xs = np.linspace(x_min, x_max, 800)

    ax.plot(xs, stats.norm.pdf(xs, loc=mu, scale=sigma),
            linewidth=2.0, linestyle="-", color="tab:blue", label="Normal fit")

    ax.plot(xs, stats.t.pdf(xs, df=nu, loc=loc, scale=scale),
            linewidth=2.0, linestyle="--", color="tab:orange", label="Student-t fit")

    ax.set_xlabel("log(gap_minutes + eps)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(frameon=False, fontsize=8)


def qqplot_student_t(ax,
                     log_gaps: np.ndarray,
                     nu: float,
                     loc: float,
                     scale: float,
                     max_points: int = 20000):

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

    ax.scatter(theo, log_gaps, s=10, facecolors="none",
               edgecolors="black", linewidths=0.6, rasterized=True)

    q1, q2 = int(0.25 * n), int(0.75 * n)
    slope = (log_gaps[q2] - log_gaps[q1]) / (theo[q2] - theo[q1] + 1e-12)
    intercept = log_gaps[q1] - slope * theo[q1]
    ax.plot(theo, slope * theo + intercept, linewidth=2.0, linestyle="-", color="tab:red")

    ax.set_xlabel("Theoretical quantiles", fontsize=9)
    ax.set_ylabel("Ordered values", fontsize=9)


def make_2x2_figure(geolife_csv: str,
                    moreuser_csv: str,
                    out_pdf: str,
                    user_col: str,
                    stime_col: str,
                    etime_col: str,
                    eps: float,
                    min_gap_min: float,
                    max_gap_pct: float):
    if not SCIPY_OK:
        raise RuntimeError("SciPy is required. Please install: pip install scipy")

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,  # >= 0.5pt
        "lines.markersize": 5,
        "pdf.fonttype": 42,      
        "ps.fonttype": 42,
    })

    df_g = pd.read_csv(geolife_csv)
    df_m = pd.read_csv(moreuser_csv)

    gaps_g = compute_gaps_minutes(df_g, user_col=user_col, stime_col=stime_col, etime_col=etime_col)
    gaps_m = compute_gaps_minutes(df_m, user_col=user_col, stime_col=stime_col, etime_col=etime_col)

    gaps_g = gaps_g[np.isfinite(gaps_g)]
    gaps_m = gaps_m[np.isfinite(gaps_m)]

    gaps_g = gaps_g[gaps_g >= min_gap_min]
    gaps_m = gaps_m[gaps_m >= min_gap_min]

    if max_gap_pct < 100.0 and gaps_g.size > 0:
        cap = np.percentile(gaps_g, max_gap_pct)
        gaps_g = gaps_g[gaps_g <= cap]
    if max_gap_pct < 100.0 and gaps_m.size > 0:
        cap = np.percentile(gaps_m, max_gap_pct)
        gaps_m = gaps_m[gaps_m <= cap]

    log_g = safe_log(gaps_g, eps=eps)
    log_m = safe_log(gaps_m, eps=eps)

    (mu_g, sigma_g), (nu_g, loc_g, scale_g) = fit_normal_studentt(log_g)
    (mu_m, sigma_m), (nu_m, loc_m, scale_m) = fit_normal_studentt(log_m)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    (ax1, ax2), (ax3, ax4) = axes

    hist_log_with_fits(ax1, log_g, mu_g, sigma_g, nu_g, loc_g, scale_g)
    ax1.set_title("(a) GeoLife: log gaps + fits", fontsize=9)

    qqplot_student_t(ax2, log_g, nu=nu_g, loc=loc_g, scale=scale_g, max_points=20000)
    ax2.set_title("(b) GeoLife: QQ vs Student-t", fontsize=9)

    hist_log_with_fits(ax3, log_m, mu_m, sigma_m, nu_m, loc_m, scale_m)
    ax3.set_title("(c) MoreUser: log gaps + fits", fontsize=9)

    qqplot_student_t(ax4, log_m, nu=nu_m, loc=loc_m, scale=scale_m, max_points=20000)
    ax4.set_title("(d) MoreUser: QQ vs Student-t", fontsize=9)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.10, wspace=0.28, hspace=0.40)

    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    print(f"[Done] Saved: {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geolife_csv", type=str, required=True)
    ap.add_argument("--moreuser_csv", type=str, required=True)
    ap.add_argument("--out_pdf", type=str, required=True)
    ap.add_argument("--user_col", type=str, default="userID")
    ap.add_argument("--stime_col", type=str, default="stime")
    ap.add_argument("--etime_col", type=str, default="etime")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--min_gap_min", type=float, default=0.0)
    ap.add_argument("--max_gap_pct", type=float, default=99.9)
    args = ap.parse_args()

    make_2x2_figure(
        geolife_csv=args.geolife_csv,
        moreuser_csv=args.moreuser_csv,
        out_pdf=args.out_pdf,
        user_col=args.user_col,
        stime_col=args.stime_col,
        etime_col=args.etime_col,
        eps=args.eps,
        min_gap_min=args.min_gap_min,
        max_gap_pct=args.max_gap_pct,
    )


if __name__ == "__main__":
    main()
