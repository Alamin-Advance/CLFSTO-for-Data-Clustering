
# Access drive contents
"""

from google.colab import drive
drive.mount('/content/drive')

"""# Wilcoxon Test

# wine dataset
"""

# --- Wilcoxon (pairwise) on last N iterations as pseudo-replicates ---
import os, pandas as pd, numpy as np
from scipy.stats import wilcoxon

# === File paths for Wine dataset ===
file_paths = {
    "CLFSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/clfsto/wine_iter_clfsto.csv",
    "CSTO":   "/content/drive/My Drive/Colab Notebooks/Clustering/results/sto/wine_iter_sto.csv",
    "SSTO":   "/content/drive/My Drive/Colab Notebooks/Clustering/results/ssto/wine_iter_ssto.csv",
    "PSO":    "/content/drive/My Drive/Colab Notebooks/Clustering/results/pso/wine_iter_pso.csv",
    "GA":     "/content/drive/My Drive/Colab Notebooks/Clustering/results/ga/wine_iter_GA.csv",
    "GWO":    "/content/drive/My Drive/Colab Notebooks/Clustering/results/gwo/wine_iter_gwo.csv",
    "WOA":    "/content/drive/My Drive/Colab Notebooks/Clustering/results/woa/wine_iter_woa.csv",
    "K-Means":"/content/drive/My Drive/Colab Notebooks/Clustering/results/kmeans/wine_iter_kmeans1.csv"
}

# === Settings ===
REF_ALGO = "CLFSTO"   # reference algorithm
N_LAST   = 30         # number of last iterations used as pseudo-replicates
ALPHA    = 0.05       # significance threshold

def _load_lastN(path, n_last):
    df = pd.read_csv(path)
    vals = pd.to_numeric(df["Fitness"], errors="coerce").dropna().to_numpy()
    if len(vals) < n_last:
        n_last = len(vals)
    return vals[-n_last:], n_last

# --- Load reference values ---
ref_vals, used_N = _load_lastN(file_paths[REF_ALGO], N_LAST)

# --- Pairwise Wilcoxon vs reference ---
rows = []
for algo, path in file_paths.items():
    if algo == REF_ALGO:
        continue
    if not os.path.exists(path):
        rows.append({
            "Algorithm pair": f"{REF_ALGO} vs {algo}",
            "p-value": None,
            "significance (Yes/No)": "No (missing file)"
        })
        continue

    comp_vals, _ = _load_lastN(path, used_N)
    # Align lengths just in case
    m = min(len(ref_vals), len(comp_vals))
    stat, p = wilcoxon(
        ref_vals[-m:], comp_vals[-m:],
        zero_method='wilcox',
        alternative='two-sided',
        method='approx'
    )
    rows.append({
        "Algorithm pair": f"{REF_ALGO} vs {algo}",
        "p-value": f"{p:.4e}",
        "significance (Yes/No)": "Yes" if p < ALPHA else "No"
    })

# --- Results table ---
wilcoxon_table = pd.DataFrame(rows)
print(f"\nWilcoxon test (last {used_N} iterations as pseudo-replicates)")
display(wilcoxon_table)

# --- Friedman (all algorithms) on last N iterations as pseudo-replicates ---
import os, pandas as pd, numpy as np
from scipy.stats import friedmanchisquare

# === File paths for Wine dataset (same structure as Wilcoxon block) ===
file_paths = {
    "CLFSTO": "/content/drive/My Drive/Colab Notebooks/Clustering/results/clfsto/wine_iter_clfsto.csv",
    "CSTO":   "/content/drive/My Drive/Colab Notebooks/Clustering/results/sto/wine_iter_sto.csv",
    "SSTO":   "/content/drive/My Drive/Colab Notebooks/Clustering/results/ssto/wine_iter_ssto.csv",
    "PSO":    "/content/drive/My Drive/Colab Notebooks/Clustering/results/pso/wine_iter_pso.csv",
    "GA":     "/content/drive/My Drive/Colab Notebooks/Clustering/results/ga/wine_iter_GA.csv",
    "GWO":    "/content/drive/My Drive/Colab Notebooks/Clustering/results/gwo/wine_iter_gwo.csv",
    "WOA":    "/content/drive/My Drive/Colab Notebooks/Clustering/results/woa/wine_iter_woa.csv",
    "K-Means":"/content/drive/My Drive/Colab Notebooks/Clustering/results/kmeans/wine_iter_kmeans1.csv"
}

# === Settings ===
N_LAST = 30     # choose 10–20 for fairness (use converged tail of the curve)
ALPHA  = 0.05   # significance threshold

def _load_lastN(path, n_last):
    df = pd.read_csv(path)
    vals = pd.to_numeric(df["Fitness"], errors="coerce").dropna().to_numpy()
    if len(vals) < n_last:
        n_last = len(vals)
    return vals[-n_last:], n_last

# Load last-N arrays for every algorithm that exists
series = {}
for algo, path in file_paths.items():
    if not os.path.exists(path):
        print(f"⚠️ Missing file for {algo}: {path}")
        continue
    vals, used_n = _load_lastN(path, N_LAST)
    if len(vals) == 0:
        print(f"⚠️ No numeric Fitness values for {algo}: {path}")
        continue
    series[algo] = vals

# Need at least 3 algorithms for Friedman to be meaningful
if len(series) < 3:
    raise ValueError(f"Friedman test needs ≥3 algorithms; found {len(series)} with data: {list(series.keys())}")

# Align lengths across algorithms
min_len = min(len(v) for v in series.values())
algos_order = list(series.keys())
samples = [series[a][-min_len:] for a in algos_order]

# Run Friedman test
stat, p = friedmanchisquare(*samples)

# Build output table in requested format
friedman_table = pd.DataFrame([{
    "Algorithm pair": "All algorithms (" + ", ".join(algos_order) + f") | last {min_len} iters",
    "p-value": f"{p:.4e}",
    "significance (Yes/No)": "Yes" if p < ALPHA else "No"
}])

print(f"Friedman test using last {min_len} iterations as pseudo-replicates")
display(friedman_table)

