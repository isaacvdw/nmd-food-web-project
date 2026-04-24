"""
Food-web null-model analysis


Research paper did null-model comparison to re-evaluated five common ecological hypotheses by comparing 
observed data from the 19 pollination networks against three null models of varying complexity 
(Patefield, shuffle, and swap algorithms). If a null model could replicate an observed pattern, 
the pattern was concluded to be a result of first-order properties like species abundance rather
than specialized ecological interactions


Workflow
--------
1. Compute observed metrics
2. Generate null networks (one type at a time)
3. Compute metrics on nulls
4. Build distribution plots + summary table (z, p)
5. Interpret: observed vs null, ecology, sampling effects

Metrics
-------
- Nestedness (NODF)
- H2' (Blüthgen specialisation)
- Degree variance  (predator & prey separately)
- C-score / togetherness
- Robustness (extinction slope, R50)
- Niche overlap (Pianka, mean across predator pairs)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, fields, asdict
from typing import Callable, Dict, List, Tuple

rng = np.random.default_rng(42)

# ──────────────────────────────────────────────
# 1.  DATA LOADING
# ──────────────────────────────────────────────

df = pd.read_csv("data/FW_012_02.csv", index_col=0)
W  = df.values.astype(float)          # quantitative (relative frequencies)
B  = (W > 0).astype(int)              # binary structure


# ──────────────────────────────────────────────
# 2.  METRIC DEFINITIONS
# ──────────────────────────────────────────────

@dataclass
class Metrics:
    nestedness_nodf:      float   # NODF nestedness (0–100)
    h2_specialisation:    float   # H2' (0 = no specialisation, 1 = perfect)
    predator_degree_var:  float   # variance of predator (column) degrees
    prey_degree_var:      float   # variance of prey (row) degrees
    c_score:              float   # Stone & Roberts C-score
    robustness_r50:       float   # prey fraction remaining when 50 % predators lost
    niche_overlap_mean:   float   # mean Pianka overlap across predator pairs


# ── 2a. Nestedness (NODF) ──────────────────────────────────────────────────

def _nodf_pair(v1: np.ndarray, v2: np.ndarray) -> float:
    """NODF contribution for one ordered pair (more-filled, less-filled)."""
    k1, k2 = v1.sum(), v2.sum()
    if k1 <= k2:          # need strictly decreasing degrees
        return 0.0
    shared = (v1 * v2).sum()
    return shared / k2 if k2 > 0 else 0.0


def compute_nestedness_nodf(A: np.ndarray) -> float:
    """
    NODF (Almeida-Neto et al. 2008).
    Computed over rows (prey) and columns (predators) separately, then averaged.
    Returns a value in [0, 100].
    """
    n_prey, n_pred = A.shape

    def _axis_nodf(M: np.ndarray) -> float:
        n = M.shape[0]
        if n < 2:
            return 0.0
        total, pairs = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                total += _nodf_pair(M[i], M[j])
                total += _nodf_pair(M[j], M[i])
                pairs += 1
        return (total / (2 * pairs)) * 100 if pairs else 0.0

    row_nodf = _axis_nodf(A)
    col_nodf = _axis_nodf(A.T)
    return (row_nodf + col_nodf) / 2.0


# ── 2b. H2' specialisation ────────────────────────────────────────────────

def compute_h2(W: np.ndarray) -> float:
    """
    Blüthgen et al. (2006) H2' from a quantitative interaction matrix.
    H2' = (H2_max - H2_obs) / (H2_max - H2_min)  ≈  1 - H2_obs / H2_max
    Uses the binary matrix as a proxy when W is not available (returns NaN).
    """
    W = W.astype(float)
    total = W.sum()
    if total == 0:
        return np.nan
    P = W / total                         # joint probability matrix
    row_m = P.sum(axis=1)                 # prey marginals
    col_m = P.sum(axis=0)                 # predator marginals

    # Observed Shannon diversity of interactions
    with np.errstate(divide='ignore', invalid='ignore'):
        h2_obs = -np.nansum(P * np.log2(np.where(P > 0, P, 1)))

    # Maximum H2 given marginals (independence)
    h2_max_r = -np.nansum(row_m * np.log2(np.where(row_m > 0, row_m, 1)))
    h2_max_c = -np.nansum(col_m * np.log2(np.where(col_m > 0, col_m, 1)))
    h2_max   = h2_max_r + h2_max_c

    if h2_max == 0:
        return 0.0
    # Normalise so 0 = random, 1 = perfectly specialised
    h2_prime = 1.0 - h2_obs / h2_max
    return float(np.clip(h2_prime, 0.0, 1.0))


# ── 2c. Degree variance ────────────────────────────────────────────────────

def compute_degree_variances(A: np.ndarray) -> Tuple[float, float]:
    """Returns (predator_degree_var, prey_degree_var)."""
    pred_deg = A.sum(axis=0)
    prey_deg = A.sum(axis=1)
    return float(pred_deg.var(ddof=1)), float(prey_deg.var(ddof=1))


# ── 2d. C-score ────────────────────────────────────────────────────────────

def compute_c_score(A: np.ndarray) -> float:
    """
    Stone & Roberts (1990) C-score: mean number of checkerboard units
    across all predator pairs.  Higher = more segregation / less co-occurrence.
    """
    n_prey, n_pred = A.shape
    if n_pred < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for j1 in range(n_pred):
        for j2 in range(j1 + 1, n_pred):
            s1, s2 = A[:, j1].sum(), A[:, j2].sum()
            shared = (A[:, j1] * A[:, j2]).sum()
            total += (s1 - shared) * (s2 - shared)
            pairs += 1
    return float(total / pairs) if pairs else 0.0


# ── 2e. Robustness (extinction slope / R50) ────────────────────────────────

def compute_robustness(A: np.ndarray) -> float:
    """
    Robustness R50: fraction of prey remaining when 50 % of predators have
    gone extinct (random sequential removal, averaged over 200 replicates).
    A prey item is 'lost' when all its predators are gone.
    """
    n_prey, n_pred = A.shape
    r50_vals = []
    for _ in range(200):
        order = rng.permutation(n_pred)
        prey_alive = np.ones(n_prey, dtype=bool)
        for k, col in enumerate(order, 1):
            # remove predator col; a prey goes extinct if it has no other predator left
            for i in range(n_prey):
                if prey_alive[i] and A[i, col]:
                    remaining = A[i, :].copy()
                    remaining[col] = 0
                    if remaining[order[k:]].sum() == 0:
                        prey_alive[i] = False
            if k == n_pred // 2:
                r50_vals.append(prey_alive.sum() / n_prey)
                break
    return float(np.mean(r50_vals)) if r50_vals else np.nan


# ── 2f. Niche overlap (Pianka) ─────────────────────────────────────────────

def compute_niche_overlap(W: np.ndarray) -> float:
    """
    Mean Pianka overlap across all predator pairs.
    Uses the quantitative matrix (relative diet frequencies).
    Falls back to binary if W has zero rows.
    """
    n_prey, n_pred = W.shape
    col_sums = W.sum(axis=0)
    # normalise each predator's diet to proportions
    with np.errstate(invalid='ignore'):
        P = np.where(col_sums > 0, W / col_sums, 0.0)

    overlaps = []
    for j1 in range(n_pred):
        for j2 in range(j1 + 1, n_pred):
            num = (P[:, j1] * P[:, j2]).sum()
            den = np.sqrt((P[:, j1] ** 2).sum() * (P[:, j2] ** 2).sum())
            overlaps.append(num / den if den > 0 else 0.0)
    return float(np.mean(overlaps)) if overlaps else np.nan


# ── Master compute function ────────────────────────────────────────────────

def compute_metrics(A: np.ndarray, W_quant: np.ndarray | None = None) -> Metrics:
    """
    Compute all metrics for binary matrix A.
    W_quant is optional quantitative matrix used for H2' and niche overlap;
    if None, falls back to A.
    """
    Q = W_quant if W_quant is not None else A.astype(float)
    pred_var, prey_var = compute_degree_variances(A)
    return Metrics(
        nestedness_nodf     = compute_nestedness_nodf(A),
        h2_specialisation   = compute_h2(Q),
        predator_degree_var = pred_var,
        prey_degree_var     = prey_var,
        c_score             = compute_c_score(A),
        robustness_r50      = compute_robustness(A),
        niche_overlap_mean  = compute_niche_overlap(Q),
    )


# ──────────────────────────────────────────────
# 3.  NULL MODELS
# ──────────────────────────────────────────────

# changes the number of prey and predators, but keeps the same number of links
def null_model_shuffle(A: np.ndarray) -> np.ndarray:
    """Null I – fixed connectance: randomly shuffle all entries."""
    flat = A.flatten()
    rng.shuffle(flat)
    return flat.reshape(A.shape)

# keeps the number of prey and predators and keeps the same number of links
def null_model_swap(A: np.ndarray, n_swaps: int = 10_000) -> np.ndarray:
    """Null II – fixed row & column sums via checkerboard 2×2 swaps."""
    A = A.copy()
    n_prey, n_pred = A.shape
    for _ in range(n_swaps):
        i1, i2 = rng.choice(n_prey, size=2, replace=False)
        j1, j2 = rng.choice(n_pred, size=2, replace=False)
        a, b, c, d = A[i1, j1], A[i1, j2], A[i2, j1], A[i2, j2]
        if (a + d == 2) and (b + c == 0):
            A[i1, j1], A[i1, j2], A[i2, j1], A[i2, j2] = 0, 1, 1, 0
        elif (b + c == 2) and (a + d == 0):
            A[i1, j1], A[i1, j2], A[i2, j1], A[i2, j2] = 1, 0, 0, 1
    return A


# ──────────────────────────────────────────────
# 4.  NULL DISTRIBUTION GENERATION
# ──────────────────────────────────────────────

def generate_null_distributions(
    A: np.ndarray,
    W_quant: np.ndarray,
    null_fn: Callable[[np.ndarray], np.ndarray],
    n_reps: int = 500,
) -> Dict[str, List[float]]:
    """
    Run null_fn n_reps times; compute all metrics each time.
    Returns a dict of metric_name → list of values.
    """
    results: Dict[str, List[float]] = {f.name: [] for f in fields(Metrics)}
    for _ in range(n_reps):
        A_null = null_fn(A)
        m = compute_metrics(A_null, W_quant=W_quant)
        for f in fields(Metrics):
            results[f.name].append(getattr(m, f.name))
    return results


# ──────────────────────────────────────────────
# 5.  SUMMARY TABLE  (z-score, p-value)
# ──────────────────────────────────────────────

def build_summary_table(
    obs: Metrics,
    null_results: Dict[str, List[float]],
    null_label: str,
) -> pd.DataFrame:
    rows = []
    for f in fields(Metrics):
        name = f.name
        arr  = np.array(null_results[name], dtype=float)
        mu   = np.nanmean(arr)
        sd   = np.nanstd(arr, ddof=1)
        obs_val = getattr(obs, name)
        z = (obs_val - mu) / sd if sd > 0 else np.nan
        p_upper = (np.nansum(arr >= obs_val) + 1) / (len(arr) + 1)
        p_lower = (np.nansum(arr <= obs_val) + 1) / (len(arr) + 1)
        p_two   = 2 * min(p_upper, p_lower)
        rows.append({
            "metric":       name,
            "observed":     round(obs_val, 4),
            "null_mean":    round(mu,      4),
            "null_sd":      round(sd,      4),
            "z_score":      round(z,       3),
            "p_upper":      round(p_upper, 4),
            "p_lower":      round(p_lower, 4),
            "p_two_tail":   round(p_two,   4),
            "null_model":   null_label,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 6.  DISTRIBUTION PLOTS
# ──────────────────────────────────────────────

METRIC_LABELS = {
    "nestedness_nodf":      "Nestedness (NODF)",
    "h2_specialisation":    "H2′ Specialisation",
    "predator_degree_var":  "Predator Degree Variance",
    "prey_degree_var":      "Prey Degree Variance",
    "c_score":              "C-score",
    "robustness_r50":       "Robustness (R50)",
    "niche_overlap_mean":   "Niche Overlap (Pianka)",
}


def plot_null_distributions(
    obs: Metrics,
    null_results: Dict[str, List[float]],
    null_label: str,
    save_path: str | None = None,
) -> None:
    """
    One subplot per metric: histogram of null distribution with
    observed value marked and z-score annotated.
    """
    metric_names = [f.name for f in fields(Metrics)]
    n = len(metric_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(5 * ncols, 4 * nrows))
    fig.suptitle(f"Null distribution: {null_label}", fontsize=14, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.35)

    for idx, name in enumerate(metric_names):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        arr      = np.array(null_results[name], dtype=float)
        obs_val  = getattr(obs, name)

        mu = np.nanmean(arr)
        sd = np.nanstd(arr, ddof=1)
        z  = (obs_val - mu) / sd if sd > 0 else np.nan

        ax.hist(arr, bins=30, color="#5b9bd5", edgecolor="white", alpha=0.8)
        ax.axvline(obs_val, color="#e84040", linewidth=2.0, label="Observed")
        ax.axvline(mu,      color="#333333", linewidth=1.2, linestyle="--", label="Null mean")

        ax.set_title(METRIC_LABELS.get(name, name), fontsize=10)
        ax.set_xlabel("Value", fontsize=8)
        ax.set_ylabel("Count",  fontsize=8)
        ax.tick_params(labelsize=7)

        z_str = f"z = {z:.2f}" if not np.isnan(z) else "z = n/a"
        ax.text(0.97, 0.95, z_str, transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        ax.legend(fontsize=7, loc="upper left")

    # hide unused axes
    for idx in range(n, nrows * ncols):
        fig.add_subplot(gs[idx // ncols, idx % ncols]).set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → plot saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────
# 7.  ECOLOGICAL INTERPRETATION
# ──────────────────────────────────────────────

def interpret(obs: Metrics, summary: pd.DataFrame, null_label: str) -> str:
    """
    Produce a plain-text ecological interpretation of the results.
    """
    lines: List[str] = []
    lines.append(f"\n{'='*70}")
    lines.append(f"ECOLOGICAL INTERPRETATION  –  {null_label}")
    lines.append(f"{'='*70}")

    def sig(metric: str, direction: str = "upper") -> bool:
        row = summary[summary.metric == metric]
        if row.empty:
            return False
        p = row[f"p_{direction}"].values[0]
        return p < 0.05

    def z_val(metric: str) -> float:
        row = summary[summary.metric == metric]
        return row["z_score"].values[0] if not row.empty else np.nan

    # ── Nestedness ─────────────────────────────────────────────────────────
    z_nest = z_val("nestedness_nodf")
    if sig("nestedness_nodf", "upper"):
        lines.append(
            f"\n● NESTEDNESS (NODF = {obs.nestedness_nodf:.1f}, z = {z_nest:.2f}, p < 0.05):\n"
            "  The network is significantly MORE nested than expected by chance.\n"
            "  Specialist predators tend to consume subsets of the prey taken by\n"
            "  generalists — a pattern consistent with core–periphery habitat\n"
            "  structure or asymmetric predator body-size hierarchies."
        )
    elif sig("nestedness_nodf", "lower"):
        lines.append(
            f"\n● NESTEDNESS (NODF = {obs.nestedness_nodf:.1f}, z = {z_nest:.2f}, p < 0.05):\n"
            "  The network is significantly LESS nested than chance — consistent\n"
            "  with niche partitioning or competitive exclusion among predators."
        )
    else:
        lines.append(
            f"\n● NESTEDNESS (NODF = {obs.nestedness_nodf:.1f}, z = {z_nest:.2f}, n.s.):\n"
            "  Nestedness does not deviate from the null expectation."
        )

    # ── H2' specialisation ─────────────────────────────────────────────────
    z_h2 = z_val("h2_specialisation")
    if sig("h2_specialisation", "upper"):
        lines.append(
            f"\n● H2′ SPECIALISATION ({obs.h2_specialisation:.3f}, z = {z_h2:.2f}, p < 0.05):\n"
            "  Interactions are more specialised than random — predators use a\n"
            "  smaller subset of available prey than expected from their marginal\n"
            "  abundances.  Consider whether prey are genuinely rare or whether\n"
            "  dietary data under-samples rare links (sampling artefact)."
        )
    else:
        lines.append(
            f"\n● H2′ SPECIALISATION ({obs.h2_specialisation:.3f}, z = {z_h2:.2f}, n.s.):\n"
            "  Interaction diversity is consistent with random linking given\n"
            "  the observed marginal diet-breadths."
        )

    # ── Degree variance ────────────────────────────────────────────────────
    z_pdv = z_val("predator_degree_var")
    z_ydv = z_val("prey_degree_var")
    lines.append(
        f"\n● DEGREE VARIANCE:\n"
        f"  Predator deg. var = {obs.predator_degree_var:.2f}  (z = {z_pdv:.2f})\n"
        f"  Prey    deg. var = {obs.prey_degree_var:.2f}  (z = {z_ydv:.2f})\n"
        "  High predator degree variance indicates heterogeneous diet breadths\n"
        "  (a few generalists, many specialists), amplifying extinction risk\n"
        "  cascades if generalists are lost first."
    )

    # ── C-score ────────────────────────────────────────────────────────────
    z_cs = z_val("c_score")
    if sig("c_score", "upper"):
        lines.append(
            f"\n● C-SCORE ({obs.c_score:.2f}, z = {z_cs:.2f}, p < 0.05):\n"
            "  More checkerboard units than random → predators partition prey\n"
            "  space more than expected (competitive exclusion signal)."
        )
    elif sig("c_score", "lower"):
        lines.append(
            f"\n● C-SCORE ({obs.c_score:.2f}, z = {z_cs:.2f}, p < 0.05):\n"
            "  Fewer checkerboard units than random → predators co-occur on\n"
            "  prey more than expected (aggregation / facilitation signal)."
        )
    else:
        lines.append(
            f"\n● C-SCORE ({obs.c_score:.2f}, z = {z_cs:.2f}, n.s.):\n"
            "  Prey co-occurrence among predators is consistent with the null."
        )

    # ── Robustness ─────────────────────────────────────────────────────────
    z_rob = z_val("robustness_r50")
    lines.append(
        f"\n● ROBUSTNESS R50 ({obs.robustness_r50:.3f}, z = {z_rob:.2f}):\n"
        "  On average, {:.0f} % of prey species remain when half the predators\n"
        "  are removed.  Values above the null imply prey are well-buffered\n"
        "  by predator redundancy; values below suggest fragile dependencies.".format(
            obs.robustness_r50 * 100
        )
    )

    # ── Niche overlap ──────────────────────────────────────────────────────
    z_no = z_val("niche_overlap_mean")
    if sig("niche_overlap_mean", "upper"):
        lines.append(
            f"\n● NICHE OVERLAP ({obs.niche_overlap_mean:.3f}, z = {z_no:.2f}, p < 0.05):\n"
            "  Higher overlap than random → predators share diet more than\n"
            "  expected.  This may indicate shared habitat use, prey aggregation,\n"
            "  or under-sampling of rare prey items (apparent overlap artefact)."
        )
    else:
        lines.append(
            f"\n● NICHE OVERLAP ({obs.niche_overlap_mean:.3f}, z = {z_no:.2f}, n.s.):\n"
            "  Diet overlap is consistent with the null model."
        )

    # ── Sampling caveat ────────────────────────────────────────────────────
    lines.append(
        "\n── SAMPLING NOTE ─────────────────────────────────────────────────────\n"
        "  Diet studies based on stomach contents or DNA metabarcoding can\n"
        "  miss rare prey, inflating H2′ and niche overlap.  If n_samples\n"
        "  per predator is low (< 20), treat specialist / high-overlap signals\n"
        "  with caution before ecological inference.\n"
        "──────────────────────────────────────────────────────────────────────"
    )

    return "\n".join(lines)


# ──────────────────────────────────────────────
# 8.  MAIN PIPELINE
# ──────────────────────────────────────────────

if __name__ == "__main__":
    A = B          # binary structure
    n_reps = 500   # increase to 1000+ for publication

    # ── Step 1: observed metrics ──────────────────────────────────────────
    print("Computing observed metrics …")
    obs = compute_metrics(A, W_quant=W)
    print(obs)

    # ── Steps 2–4: one null model at a time ───────────────────────────────
    null_models = {
        "Shuffle (fixed connectance)":   lambda M: null_model_shuffle(M),
        "Swap (fixed marginals)":        lambda M: null_model_swap(M, n_swaps=10_000),
    }

    all_summaries: List[pd.DataFrame] = []

    for label, null_fn in null_models.items():
        print(f"\nRunning null model: {label}  ({n_reps} replicates) …")

        # Step 2: generate nulls & compute metrics
        null_results = generate_null_distributions(A, W, null_fn, n_reps=n_reps)

        # Step 3: summary table
        summary = build_summary_table(obs, null_results, null_label=label)
        all_summaries.append(summary)

        print(summary.to_string(index=False))

        # Step 4a: distribution plots
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
        plot_null_distributions(
            obs,
            null_results,
            null_label=label,
            save_path=f"figures/null_{safe_label}.png",
        )

        # Step 5: ecological interpretation
        interp = interpret(obs, summary, null_label=label)
        print(interp)

    # ── Combined summary table ─────────────────────────────────────────────
    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv("results/null_model_summary.csv", index=False)
    print("\nFull summary saved to results/null_model_summary.csv")