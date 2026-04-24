import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Dict, List

rng = np.random.default_rng(42)

# ---- LOAD DATA ----
# CSV: rows = prey, columns = predators, entries = relative frequency in diet
df = pd.read_csv("data/FW_012_02.csv", index_col=0)

W = df.values.astype(float)          # quantitative (relative frequencies)
B = (W > 0).astype(int)              # binary structure: 1 if prey appears in diet

n_prey, n_pred = B.shape

@dataclass
class Indices:
    connectance: float
    mean_predator_degree: float
    mean_prey_degree: float
    predator_degree_var: float
    prey_degree_var: float

def compute_indices(A: np.ndarray) -> Indices:
    """
    A: prey x predator binary matrix
    """
    n_prey, n_pred = A.shape
    L = A.sum()
    connectance = L / (n_prey * n_pred)

    prey_deg = A.sum(axis=1)   # row sums
    pred_deg = A.sum(axis=0)   # col sums

    return Indices(
        connectance=float(connectance),
        mean_predator_degree=float(pred_deg.mean()),
        mean_prey_degree=float(prey_deg.mean()),
        predator_degree_var=float(pred_deg.var()),
        prey_degree_var=float(prey_deg.var()),
    )

def null_model_shuffle(A: np.ndarray) -> np.ndarray:
    """
    Fixed connectance: shuffle all entries.
    """
    flat = A.flatten()
    rng.shuffle(flat)
    return flat.reshape(A.shape)


def null_model_swap(A: np.ndarray, n_swaps: int = 10000) -> np.ndarray:
    """
    Fixed row and column sums via 2x2 swaps.
    A must be binary (0/1).
    """
    A = A.copy()
    n_prey, n_pred = A.shape

    for _ in range(n_swaps):
        i1, i2 = rng.choice(n_prey, size=2, replace=False)
        j1, j2 = rng.choice(n_pred, size=2, replace=False)

        a = A[i1, j1]
        b = A[i1, j2]
        c = A[i2, j1]
        d = A[i2, j2]

        # 1 0 / 0 1  <->  0 1 / 1 0
        if (a + d == 2) and (b + c == 0):
            A[i1, j1], A[i1, j2], A[i2, j1], A[i2, j2] = 0, 1, 1, 0
        elif (b + c == 2) and (a + d == 0):
            A[i1, j1], A[i1, j2], A[i2, j1], A[i2, j2] = 1, 0, 0, 1

    return A

try:
    import os
    os.environ["R_HOME"] = r"C:\Program Files\R\R-4.5.3"
    os.environ["PATH"] = r"C:\Program Files\R\R-4.5.3\bin\x64;" + os.environ["PATH"]
    import rpy2.robjects as ro
    print(ro.r("R.version.string"))
    from rpy2.robjects import numpy2ri
    has_rpy2 = True
except ImportError:
    has_rpy2 = False

from rpy2.robjects import conversion, default_converter
from rpy2.robjects import numpy2ri

def null_model_patefield(A: np.ndarray) -> np.ndarray:
    """
    Fixed row and column sums using R's r2dtable (Patefield algorithm).
    """
    if not has_rpy2:
        raise RuntimeError("rpy2 not installed; cannot use Patefield null model.")

    r = ro.r
    r('library(stats)')

    row_sums = A.sum(axis=1)
    col_sums = A.sum(axis=0)

    with conversion.localconverter(default_converter + numpy2ri.converter):
        r_mat_list = r['r2dtable'](
            1,
            ro.FloatVector(row_sums),
            ro.FloatVector(col_sums)
        )
        r_mat = np.array(r_mat_list[0])

    return (r_mat > 0).astype(int)

def generate_null_distributions(
    A: np.ndarray,
    null_fn: Callable[[np.ndarray], np.ndarray],
    n_reps: int = 1000,
) -> Dict[str, List[float]]:
    results = {
        "connectance": [],
        "mean_predator_degree": [],
        "mean_prey_degree": [],
        "predator_degree_var": [],
        "prey_degree_var": [],
    }

    for _ in range(n_reps):
        A_null = null_fn(A)
        idx = compute_indices(A_null)
        results["connectance"].append(idx.connectance)
        results["mean_predator_degree"].append(idx.mean_predator_degree)
        results["mean_prey_degree"].append(idx.mean_prey_degree)
        results["predator_degree_var"].append(idx.predator_degree_var)
        results["prey_degree_var"].append(idx.prey_degree_var)

    return results


def summarize_comparison(obs: Indices, null_results: Dict[str, List[float]]) -> pd.DataFrame:
    summary = []
    for name, vals in null_results.items():
        arr = np.array(vals)
        mu = arr.mean()
        sd = arr.std(ddof=1)
        obs_val = getattr(obs, name)
        z = (obs_val - mu) / sd if sd > 0 else np.nan
        p = (np.sum(arr >= obs_val) + 1) / (len(arr) + 1)  # upper-tail p
        summary.append({
            "index": name,
            "observed": obs_val,
            "null_mean": mu,
            "null_sd": sd,
            "z_score": z,
            "p_upper": p,
        })
    return pd.DataFrame(summary)

if __name__ == "__main__":
    A = B  # binary structure of your predator–prey network

    obs_idx = compute_indices(A)
    print("Observed indices:")
    print(obs_idx)

    n_reps = 500  # bump to 1000+ once you're happy with runtime

    # Null II: Shuffle (fixed connectance)
    shuffle_results = generate_null_distributions(A, null_model_shuffle, n_reps=n_reps)
    shuffle_summary = summarize_comparison(obs_idx, shuffle_results)
    print("\nShuffle null (fixed connectance):")
    print(shuffle_summary)

    # Null III: Swap (fixed row & column sums)
    swap_results = generate_null_distributions(
        A, lambda M: null_model_swap(M, n_swaps=10000), n_reps=n_reps
    )
    swap_summary = summarize_comparison(obs_idx, swap_results)
    print("\nSwap null (fixed marginals):")
    print(swap_summary)

    # # Null I: Patefield (if R available)
    # if has_rpy2:
    #     patefield_results = generate_null_distributions(A, null_model_patefield, n_reps=n_reps)
    #     patefield_summary = summarize_comparison(obs_idx, patefield_results)
    #     print("\nPatefield null (fixed marginals via r2dtable):")
    #     print(patefield_summary)
    # else:
    #     print("\nPatefield null skipped (rpy2 not installed).")
