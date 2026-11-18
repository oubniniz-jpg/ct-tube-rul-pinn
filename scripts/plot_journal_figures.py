"""
Journal figures for the CT X-ray tube RUL paper.
=================================================
Regenerates the two paper figures from the real experimental results, in a
journal style (Times New Roman, restrained palette, grayscale-safe markers):

  results/figures/workload_lifespan_curve.png   (digital twin, Innovation 1)
  results/figures/scarcity_curve.png            (augmentation, Innovation 2)

Inputs (produced by the simulation / experiment scripts):
  data/processed/sim_tube_lifespans.csv    - simulator lifespan by workload
  data/processed/real_tube_lifespans.csv   - real CHU + Tetouan tube lifespans
  results/ct_tubes/scarcity_seeds10_raw.csv - RUL RMSE vs #train tubes (10 seeds)

Usage:
  python scripts/plot_journal_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- Times New Roman (falls back to a serif face if unavailable) ----
_times = r"C:\Windows\Fonts\times.ttf"
if os.path.exists(_times):
    fm.fontManager.addfont(_times)
    _family = "Times New Roman"
else:
    _family = "serif"
plt.rcParams.update({
    "font.family": _family,
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#333333",
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "savefig.dpi": 400, "savefig.bbox": "tight",
})

INK = "#2b2b2b"; CHU = "#8c2d2d"; TET = "#2f4b7c"; AGA = "#3f7d44"; BAND = "#b8b8b8"
FIGDIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIGDIR, exist_ok=True)


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def figure_workload_lifespan():
    sim = pd.read_csv(os.path.join(ROOT, "data", "processed", "sim_tube_lifespans.csv"))
    real = pd.read_csv(os.path.join(ROOT, "data", "processed", "real_tube_lifespans.csv"))
    g = sim.groupby("workload_scans_per_day")["lifespan_days"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.axvline(85, color="#bfbfbf", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(15.5, 60, "age-driven regime", fontsize=10.5, style="italic", color="#6a6a6a")
    ax.text(150, 60, "usage-driven regime", fontsize=10.5, style="italic", color="#6a6a6a", ha="center")

    ax.fill_between(g["workload_scans_per_day"], g["mean"] - g["std"], g["mean"] + g["std"],
                    color=BAND, alpha=0.25, lw=0, zorder=2)
    ax.plot(g["workload_scans_per_day"], g["mean"], "-o", color=INK, lw=1.5, ms=5,
            mfc="white", mec=INK, mew=1.2, zorder=4, label="Simulator mean $\\pm$1 SD")

    hosp = {"CHU Ibn Rochd": (CHU, 100, "CHU Ibn Rochd (real, $n{=}11$)"),
            "Hopital Civil Tetouan": (TET, 13, "T\u00e9touan (real, $n{=}2$)")}
    for h, (c, x, lab) in hosp.items():
        grp = real[real.hospital == h]["lifespan_days"]
        xj = x * (1 + np.random.RandomState(0).uniform(-0.05, 0.05, len(grp)))
        ax.scatter(xj, grp, s=16, color=c, alpha=0.35, zorder=3, edgecolor="none")
        ax.scatter([x], [grp.mean()], marker="*", s=260, color=c, edgecolor="white",
                   linewidth=0.9, zorder=6, label=lab)
    ax.scatter([20], [g[g.workload_scans_per_day == 20]["mean"].values[0]], marker="D", s=64,
               facecolor="white", edgecolor=AGA, linewidth=1.6, zorder=6, label="Agadir (predicted)")

    ax.set_xscale("log")
    ax.set_xticks([13, 20, 40, 70, 100, 150, 250])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(11, 290); ax.set_ylim(0, 1050)
    ax.set_xlabel("Workload (CT examinations per day)")
    ax.set_ylabel("X-ray tube lifespan (days)")
    ax.grid(axis="y", color="#e6e6e6", lw=0.7, zorder=0)
    _despine(ax)
    ax.legend(loc="upper right", fontsize=10, frameon=False, handletextpad=0.5)
    out = os.path.join(FIGDIR, "workload_lifespan_curve.png")
    fig.savefig(out); plt.close(fig)
    print("saved", out)


def figure_scarcity():
    df = pd.read_csv(os.path.join(ROOT, "results", "ct_tubes", "scarcity_seeds10_raw.csv"))
    s = df.groupby("n_train_tubes")["rmse_days"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.axhline(7.3, color="#9fb89f", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.errorbar(s["n_train_tubes"], s["mean"], yerr=s["std"], marker="o", ms=6, lw=1.5,
                color=TET, mfc="white", mec=TET, mew=1.3, capsize=3, elinewidth=1,
                zorder=4, label="RUL error (mean $\\pm$1 SD, 10 seeds)")
    ax.annotate("2 tubes: 81-day error\n(model unusable)", xy=(2, 80.65), xytext=(3.4, 65),
                fontsize=10, color=CHU, arrowprops=dict(arrowstyle="-|>", color=CHU, lw=1))
    ax.text(14, 13.5, "plateau $\\approx$ 7 days ($<$2% of tube life)", fontsize=10, color="#4f6f4f")

    ax.set_xscale("log")
    ax.set_xticks([2, 6, 14, 29, 42])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(1.7, 50); ax.set_ylim(0, 118)
    ax.set_xlabel("Number of simulator tubes used for training")
    ax.set_ylabel("RUL error, RMSE (days)")
    ax.grid(axis="y", color="#e6e6e6", lw=0.7, zorder=0)
    _despine(ax)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    out = os.path.join(FIGDIR, "scarcity_curve.png")
    fig.savefig(out); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    figure_workload_lifespan()
    figure_scarcity()
    print("font in use:", plt.rcParams["font.family"])
