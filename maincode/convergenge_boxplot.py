
# Access drive contents
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# =========================
# FILE PATHS (fill these)
# =========================
file_paths = {
    "Wine": {
        "CLFSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/clfsto/wine_iter_clfsto.csv",
        "CSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/sto/wine_iter_sto.csv",
        "SSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/ssto/wine_iter_ssto.csv",
        "PSO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/pso/wine_iter_pso.csv",
        "GA": "/content/drive/My Drive/Colab Notebooks/Clustering/results/ga/wine_iter_GA.csv",
        "GWO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/gwo/wine_iter_gwo.csv",
        "WOA": "/content/drive/My Drive/Colab Notebooks/Clustering/results/woa/wine_iter_woa.csv",
        "K-Means": "/content/drive/My Drive/Colab Notebooks/Clustering/results/kmeans/wine_iter_kmeans1.csv"
    },
    "Iris":  {
        "CLFSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/clfsto/iris_iter_clfsto.csv",
        "CSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/sto/iris_iter_sto.csv",
        "SSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/ssto/iris_iter_ssto.csv",
        "GA": "/content/drive/My Drive/Colab Notebooks/Clustering/results/ga/iris_iter_GA.csv",
        "GWO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/gwo/iris_iter_gwo.csv",
        "WOA": "/content/drive/My Drive/Colab Notebooks/Clustering/results/woa/iris_iter_woa.csv",
        "K-Means": "/content/drive/My Drive/Colab Notebooks/Clustering/results/kmeans/iris_iter_kmeans.csv"
    },
    "Heart": {
        "CLFSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/clfsto/heart_iter_clfsto.csv",
        "CSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/sto/heart_iter_sto.csv",
        "SSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/ssto/heart_iter_ssto.csv",
        "GA": "/content/drive/My Drive/Colab Notebooks/Clustering/results/ga/heart_iter_GA.csv",
        "GWO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/gwo/heart_iter_gwo.csv",
        "WOA": "/content/drive/My Drive/Colab Notebooks/Clustering/results/woa/heart_iter_woa.csv",
        "K-Means": "/content/drive/My Drive/Colab Notebooks/Clustering/results/kmeans/heart_iter_kmeans.csv"
    },
    "Glass": {
        "CLFSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/clfsto/glass_iter_clfsto.csv",
        "CSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/sto/glass_iter_sto.csv",
        "SSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/ssto/glass_iter_ssto.csv",
        "GA": "/content/drive/My Drive/Colab Notebooks/Clustering/results/ga/glass_iter_GA.csv",
        "GWO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/gwo/glass_iter_gwo.csv",
        "WOA": "/content/drive/My Drive/Colab Notebooks/Clustering/results/woa/glass_iter_woa.csv",
        "K-Means": "/content/drive/My Drive/Colab Notebooks/Clustering/results/kmeans/glass_iter_kmeans.csv"
   },
    "Balance": {
        "CLFSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/clfsto/balance_iter_clfsto.csv",
        "CSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/sto/balance_iter_sto.csv",
        "SSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/ssto/balance_iter_ssto.csv",
        "GA": "/content/drive/My Drive/Colab Notebooks/Clustering/results/ga/balance_iter_GA.csv",
        "GWO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/gwo/balance_iter_gwo.csv",
        "WOA": "/content/drive/My Drive/Colab Notebooks/Clustering/results/woa/balance_iter_woa.csv",
        "K-Means": "/content/drive/My Drive/Colab Notebooks/Clustering/results/kmeans/balance_iter_kmeans.csv"
    },
    "Cancer": {
        "CLFSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/clfsto/cancer_iter_clfsto.csv",
        "CSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/sto/cancer_iter_sto.csv",
        "SSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/ssto/cancer_iter_ssto.csv",
        "GA": "/content/drive/My Drive/Colab Notebooks/Clustering/results/ga/cancer_iter_GA.csv",
        "GWO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/gwo/cancer_iter_gwo.csv",
        "WOA": "/content/drive/My Drive/Colab Notebooks/Clustering/results/woa/cancer_iter_woa.csv",
        "K-Means": "/content/drive/My Drive/Colab Notebooks/Clustering/results/kmeans/cancer_iter_kmeans.csv"
    },
    "Diabetes": {
        "CLFSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/clfsto/diabetes_iter_clfsto.csv",
        "CSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/sto/diabetes_iter_sto.csv",
        "SSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/ssto/diabetes_iter_ssto.csv",
        "GA": "/content/drive/My Drive/Colab Notebooks/Clustering/results/ga/diabetes_iter_GA.csv",
        "GWO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/gwo/diabetes_iter_gwo.csv",
        "WOA": "/content/drive/My Drive/Colab Notebooks/Clustering/results/woa/diabetes_woa.csv",
        "K-Means": "/content/drive/My Drive/Colab Notebooks/Clustering/results/kmeans/diabetes_iter_kmeans.csv"
    },
    "Ecoli": {
        "CLFSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/clfsto/ecoli_iter_clfsto.csv",
        "CSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/sto/ecoli_iter_sto.csv",
        "SSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/ssto/ecoli_iter_ssto.csv",
        "GWO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/gwo/ecoli_iter_gwo.csv",
        "WOA": "/content/drive/My Drive/Colab Notebooks/Clustering/results/woa/ecoli_woa.csv",
        "K-Means": "/content/drive/My Drive/Colab Notebooks/Clustering/results/kmeans/ecoli_iter_kmeans.csv"
    },
    "Seeds": {
        "CLFSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/clfsto/seeds_iter_clfsto.csv",
        "CSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/sto/seeds_iter_sto.csv",
        "SSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/ssto/seeds_iter_ssto.csv",
        "GA": "/content/drive/My Drive/Colab Notebooks/Clustering/results/ga/seeds_iter_GA.csv",
        "GWO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/gwo/seeds_iter_gwo.csv",
        "WOA": "/content/drive/My Drive/Colab Notebooks/Clustering/results/woa/seeds_woa.csv",
        "K-Means": "/content/drive/My Drive/Colab Notebooks/Clustering/results/kmeans/seeds_iter_kmeans.csv"
    },
    "Liver": {
        "CLFSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/clfsto/liver_iter_clfsto.csv",
        "CSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/sto/liver_iter_sto.csv",
        "SSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/ssto/liver_iter_ssto.csv",
        "GA": "/content/drive/My Drive/Colab Notebooks/Clustering/results/ga/liver_iter_GA.csv",
        "GWO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/gwo/liver_iter_gwo.csv",
        "WOA": "/content/drive/My Drive/Colab Notebooks/Clustering/results/woa/liver_iter_woa.csv",
        "K-Means": "/content/drive/My Drive/Colab Notebooks/Clustering/results/kmeans/liver_iter_kmeans.csv"
    },
}

datasets_order = ["Wine","Iris","Glass","Balance","Heart","Cancer","Diabetes","Ecoli","Seeds","Liver"]

# =========================
# STYLES (match your single-plot style)
# =========================
styles = {
    "CLFSTO": {"color": "blue",   "linestyle": "--"},
    "CSTO":   {"color": "green",  "linestyle": "-"},
    "SSTO":   {"color": "cyan",   "linestyle": "-"},
    "PSO":    {"color": "red",    "linestyle": "-"},
    "GA":     {"color": "orange", "linestyle": "-"},
    "GWO":    {"color": "purple", "linestyle": "-"},
    "WOA":    {"color": "brown",  "linestyle": "-"},
    "K-Means":{"color": "black",  "linestyle": "-"},
}

# Optional fixed y-lims per dataset (fill if you want tight ranges)
per_ylim = {
   # "Liver": (0.94, 1.10),
    #"Wine": (0.95, 1.10),
    # Add others as needed...
}

# =========================
# BUILD FIGURE (2 x 5) with global legend
# =========================
align_start_at_zero = True
out_dir = "/content/drive/My Drive/Colab Notebooks/Clustering/results/con_curve"
os.makedirs(out_dir, exist_ok=True)

fig, axes = plt.subplots(5, 2, figsize=(18, 22))
axes = axes.flatten()

# Create legend proxies so ALL algorithms appear in the legend
legend_handles = []
legend_labels = []
for alg, st in styles.items():
    legend_handles.append(Line2D([0], [0], color=st["color"], linestyle=st["linestyle"]))
    legend_labels.append(alg)

for i, dataset in enumerate(datasets_order):
    ax = axes[i]
    alg_files = file_paths.get(dataset, {})
    if not alg_files:
        ax.set_visible(False)
        continue

    max_iter = 0
    plotted_any = False

    for alg in styles.keys():
        path = alg_files.get(alg)
        if path is None or not os.path.isfile(path):
            print(f"[WARN] Missing file for {dataset} – {alg}: {path}")
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
            continue

        if ("Iteration" not in df.columns) or ("Fitness" not in df.columns):
            print(f"[WARN] Bad columns in {path} (need Iteration,Fitness)")
            continue

        x = df["Iteration"].values
        y = df["Fitness"].values

        # Start curves from 0 on x-axis and ensure they touch the y-axis
        if align_start_at_zero:
            x = x - x.min()

        ax.plot(
            x, y,
            label=alg,
            color=styles[alg]["color"],
            linestyle=styles[alg]["linestyle"]
        )
        plotted_any = True
        if len(x) > 0:
            max_iter = max(max_iter, x.max())

    ax.set_title(f"{dataset} Dataset", fontsize=14)
    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_ylabel("Fitness", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Force x to start at 0 and use a reasonable number of ticks (prevents Cancer overlap)
    ax.set_xlim(left=0)
    if max_iter > 0:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))  # ~6 ticks max

    # Apply per-dataset y-limits if provided
    if dataset in per_ylim:
        ymin, ymax = per_ylim[dataset]
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

    # If nothing plotted (e.g., all paths missing), hide the axes to keep the grid clean
    if not plotted_any:
        ax.set_visible(False)

# Hide any unused axes
for j in range(len(datasets_order), len(axes)):
    axes[j].set_visible(False)

# Global legend outside top-right
fig.legend(
    legend_handles, legend_labels,
    title="Algorithms",
    fontsize=12, title_fontsize=13,
    loc="upper right",
    bbox_to_anchor=(1.12, 0.98)
)

plt.tight_layout(rect=[0, 0, 0.90, 0.97])
#fig.suptitle("Figure 4. Convergence curves across 10 datasets", fontsize=14, y=1.02)

# SAVE (300 dpi, all formats incl. TIFF)
base = os.path.join(out_dir, "figure_4_convergence_curves")
plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
plt.savefig(base + ".jpg", dpi=300, bbox_inches="tight")
plt.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
plt.savefig(base + ".tif", dpi=300, bbox_inches="tight")
plt.show()


import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1) CONFIG
# =========================
# Order of datasets on the grid (2 columns x 5 rows)
DATASETS = ["Wine","Iris","Glass","Balance","Heart","Cancer","Diabetes","Ecoli","Seeds","Liver"]

# Algorithms (order will be used on x-axis)
ALGORITHMS = ["CLFSTO","CSTO","SSTO","PSO","GA","GWO","WOA","K-Means"]

# Use only last N points from each series (set to None to use full series)
LAST_N = 30

# Optional: share same y-scale across subplots for fairer visual comparison
SHARE_Y = True

# Optional fixed y-limits per dataset (leave empty to auto-scale)
PER_YLIM = {
    # Example: "Liver": (0.94, 1.10),
}

# Output directory and base filename
OUT_DIR = "/content/drive/My Drive/Colab Notebooks/Clustering/results/boxplots"
os.makedirs(OUT_DIR, exist_ok=True)
BASE = os.path.join(OUT_DIR, "figure_3_boxplots")


# =========================
# 3) BUILD FIGURE (2 x 5)
# =========================
fig, axes = plt.subplots(5, 2, figsize=(12, 20), sharey=SHARE_Y)
axes = axes.flatten()

for i, ds in enumerate(DATASETS):
    ax = axes[i]
    ds_paths = paths.get(ds, {})
    if not ds_paths:
        ax.set_visible(False)
        continue

    # Collect values per algorithm in the specified order
    values = []
    labels = []
    for alg in ALGORITHMS:
        p = ds_paths.get(alg)
        if p is None or not os.path.isfile(p):
            print(f"[WARN] Missing file for {ds} – {alg}: {p}")
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
            continue

        if "Fitness" not in df.columns:
            print(f"[WARN] Missing 'Fitness' column in {p}")
            continue

        series = df["Fitness"]
        if LAST_N is not None and len(series) >= LAST_N:
            series = series.tail(LAST_N)
        values.append(series.values)
        labels.append(alg)

    if not values:
        ax.set_visible(False)
        continue

    bp = ax.boxplot(values, labels=labels, patch_artist=True)
    # (No custom colors to stay journal-neutral; defaults are fine)

    ax.set_title(f"{ds} Dataset", fontsize=12)
    ax.set_xlabel("")   # keep clean; x tick labels carry algorithm names
    #ax.set_ylabel("Fitness", fontsize=12)
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Optional fixed y-range per dataset
    if ds in PER_YLIM:
        ymin, ymax = PER_YLIM[ds]
        ax.set_ylim(ymin, ymax)

# Hide any unused axes (safety)
for j in range(len(DATASETS), len(axes)):
    axes[j].set_visible(False)

# Title & layout
#fig.suptitle("Figure 3. Box plots comparing performance distributions across datasets", fontsize=14, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])

# =========================
# 4) SAVE (300 dpi, all formats incl. TIFF)
# =========================
plt.savefig(BASE + ".png", dpi=300, bbox_inches="tight")
plt.savefig(BASE + ".jpg", dpi=300, bbox_inches="tight")
plt.savefig(BASE + ".pdf", dpi=300, bbox_inches="tight")
plt.savefig(BASE + ".tif", dpi=300, bbox_inches="tight")
plt.show()