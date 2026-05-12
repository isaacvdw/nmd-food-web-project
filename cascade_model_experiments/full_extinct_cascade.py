"""
food_web_full_extinction.py
============================
Randomly removes species one at a time until all are gone (or all remaining
species are wiped out by cascades).  Applies the same cascade rule as the
simulation scripts:
  A species goes extinct if ALL of its prey have been removed / are extinct.

Outputs two figures:
  1. total_biomass_over_time.png  — total biomass across the whole web at
                                    each removal step
  2. tier_biomass_over_time.png   — one line per trophic tier showing how
                                    its biomass evolves over every step

Biomass model
  Every surviving species is assigned a constant biomass of 1.0, so the
  total biomass at any step equals the number of surviving species and the
  tier biomass equals the number of survivors in that tier.

Usage
  python full_extinct_cascade.py --csv data/FW_012_02.csv
  python full_extinct_cascade.py --csv data/FW_012_02.csv --seed 7
  python full_extinct_cascade.py --csv data/FW_012_02.csv --seed 7 --prefix results/run1
"""

import argparse
import random

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Graph helpers (self-contained copy)
# ---------------------------------------------------------------------------

def load_graph(csv_path: str) -> nx.DiGraph:
    df = pd.read_csv(csv_path, index_col=0)
    G = nx.DiGraph()
    prey_nodes = df.index.tolist()
    predator_nodes = df.columns.tolist()
    rows, cols = np.where(df.values > 0)
    G.add_edges_from(
        (prey_nodes[r], predator_nodes[c]) for r, c in zip(rows, cols)
    )
    return G


def compute_trophic_levels(G: nx.DiGraph) -> dict:
    G_clean = G.copy()
    G_clean.remove_edges_from(nx.selfloop_edges(G_clean))
    scc = list(nx.strongly_connected_components(G_clean))
    G_dag = nx.condensation(G_clean, scc)
    levels_dag = {}
    for node in nx.topological_sort(G_dag):
        preds = list(G_dag.predecessors(node))
        levels_dag[node] = (
            0 if not preds else max(levels_dag[p] for p in preds) + 1
        )
    levels = {}
    for i, comp in enumerate(scc):
        for node in comp:
            levels[node] = levels_dag[i]
    return levels


def cascade_extinctions(G_original: nx.DiGraph, removed: set) -> set:
    """Iteratively mark species extinct whose entire prey set is gone."""
    extinct = set(removed)
    changed = True
    while changed:
        changed = False
        for node in G_original.nodes():
            if node in extinct:
                continue
            prey = list(G_original.predecessors(node))
            if prey and all(p in extinct for p in prey):
                extinct.add(node)
                changed = True
    return extinct


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_full_extinction(csv_path: str, seed: int, output_prefix: str):
    random.seed(seed)
    np.random.seed(seed)

    print(f"Loading graph: {csv_path}")
    G = load_graph(csv_path)
    levels = compute_trophic_levels(G)
    nx.set_node_attributes(G, levels, "level")

    all_nodes = list(G.nodes())
    n_total = len(all_nodes)
    all_tiers = sorted(set(levels.values()))
    print(f"  {n_total} species across tiers {all_tiers}")

    # Shuffle removal order up front
    removal_order = all_nodes.copy()
    random.shuffle(removal_order)

    # -----------------------------------------------------------------------
    # Track state step by step
    # Each "step" corresponds to one deliberate removal.
    # After each removal we apply cascades, then record biomass.
    # Step 0 = pristine web (no removals yet).
    # -----------------------------------------------------------------------
    cumulative_extinct: set = set()

    # step_labels[i] = name of species deliberately removed at step i
    # (step 0 = baseline, no removal)
    step_labels = ["baseline"]

    # total_biomass[i] = number of surviving species after step i
    total_biomass = [n_total]

    # tier_biomass[tier][i] = surviving count in that tier after step i
    tier_biomass = {t: [sum(1 for n in all_nodes if levels[n] == t)]
                    for t in all_tiers}

    removal_idx = 0  # pointer into removal_order

    while removal_idx < len(removal_order):
        # Pick the next species that hasn't already gone extinct
        species = removal_order[removal_idx]
        removal_idx += 1

        if species in cumulative_extinct:
            # Already gone via cascade — skip without recording a step
            continue

        # Apply removal + cascades
        all_extinct = cascade_extinctions(G, cumulative_extinct | {species})
        cumulative_extinct = all_extinct

        surviving = set(all_nodes) - cumulative_extinct
        n_surviving = len(surviving)

        step_labels.append(species)
        total_biomass.append(n_surviving)
        for t in all_tiers:
            tier_biomass[t].append(
                sum(1 for n in surviving if levels[n] == t)
            )

        print(f"  Step {len(step_labels)-1:3d}: removed '{species}' from tier {levels[species]}"
              f"→ {n_surviving} survivors")

        if n_surviving == 0:
            break

    n_steps = len(total_biomass)
    x = list(range(n_steps))

    # -----------------------------------------------------------------------
    # Figure 1 — total biomass over time
    # -----------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(x, total_biomass, color="#2980B9", linewidth=2,
             marker="o", markersize=3, label="Total biomass")
    ax1.fill_between(x, total_biomass, alpha=0.15, color="#2980B9")

    ax1.set_xlabel("Removal step", fontsize=11)
    ax1.set_ylabel("Total biomass (surviving species count)", fontsize=11)
    ax1.set_title("Total Biomass Over Time\n"
                  "(each step = one deliberate random removal + cascades)",
                  fontsize=12)
    ax1.set_xlim(0, n_steps - 1)
    ax1.set_ylim(0, n_total * 1.05)
    ax1.grid(axis="y", alpha=0.3)

    # Annotate the largest single-step drop
    drops = [total_biomass[i] - total_biomass[i + 1]
             for i in range(len(total_biomass) - 1)]
    if drops:
        worst_step = int(np.argmax(drops)) + 1
        ax1.annotate(
            f"Largest drop\n({drops[worst_step-1]:.0f} species)\n"
            f"'{step_labels[worst_step]}'",
            xy=(worst_step, total_biomass[worst_step]),
            xytext=(worst_step + max(1, n_steps * 0.05),
                    total_biomass[worst_step] + n_total * 0.08),
            arrowprops=dict(arrowstyle="->", color="crimson"),
            fontsize=7, color="crimson"
        )

    fig1.tight_layout()
    out1 = f"{output_prefix}_total_biomass.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"\nSaved → {out1}")

    # -----------------------------------------------------------------------
    # Figure 2 — per-tier biomass over time
    # -----------------------------------------------------------------------
    # Use a colour palette with one colour per tier
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(all_tiers))
    tier_colors = {t: cmap(i) for i, t in enumerate(all_tiers)}

    fig2, ax2 = plt.subplots(figsize=(12, 5))

    for t in all_tiers:
        vals = tier_biomass[t]
        ax2.plot(x, vals, linewidth=2, marker="o", markersize=3,
                 color=tier_colors[t], label=f"Tier {t}")
        ax2.fill_between(x, vals, alpha=0.08, color=tier_colors[t])

    ax2.set_xlabel("Removal step", fontsize=11)
    ax2.set_ylabel("Biomass per tier (surviving species count)", fontsize=11)
    ax2.set_title("Biomass per Trophic Tier Over Time\n"
                  "(each step = one deliberate random removal + cascades)",
                  fontsize=12)
    ax2.set_xlim(0, n_steps - 1)
    ax2.set_ylim(0)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(title="Trophic tier", fontsize=9, title_fontsize=9,
               loc="upper right")

    fig2.tight_layout()
    out2 = f"{output_prefix}_tier_biomass.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved → {out2}")

    return out1, out2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Simulate complete random species removal and plot biomass "
            "trajectories (total and per tier)."
        )
    )
    parser.add_argument("--csv", required=True,
                        help="Path to adjacency matrix CSV "
                             "(e.g. data/FW_012_02.csv)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--prefix", default="extinction_run",
                        help="Output filename prefix "
                             "(default: extinction_run)")
    args = parser.parse_args()

    run_full_extinction(
        csv_path=args.csv,
        seed=args.seed,
        output_prefix=args.prefix,
    )