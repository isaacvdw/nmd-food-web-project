"""
food_web_tiered_removal.py
============================
Removes species from *specified tiers* in a given order.

Instead of picking species uniformly at random, this script accepts a
sequence of tier numbers.  At each step it picks a random *surviving*
species from the requested tier and removes it.

Produces the same visualisations as food_web_removal_simulation.py:
  - Tiered food web image per step (colour-coded nodes)
  - Biomass-per-tier bar chart embedded in the same figure

Usage examples
  # Remove one species from tier 2, then one from tier 1, then tier 3
  python simple_cascade_specific.py --csv ../data/FW_003_NAMED.csv --tiers 2 1 3

  # Remove from tier 0 three times in a row, then tier 1 twice
  python .python simple_cascade_specific.py --csv ../data/FW_003_NAMED.csv --tiers 0 0 0 1 1

  # With a fixed seed for reproducibility
  python simple_cascade_specific.py --csv ../data/FW_003_NAMED.csv --tiers 1 2 1 --seed 7
"""

import argparse
import random

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Shared helpers (copied from food_web_removal_simulation.py so each script
# is self-contained and can be run independently)
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


def hierarchical_pos(levels: dict) -> dict:
    nodes_by_level: dict = {}
    for node, lvl in levels.items():
        nodes_by_level.setdefault(lvl, []).append(node)
    pos = {}
    for lvl, nodes in nodes_by_level.items():
        n = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            pos[node] = np.array([(i - n / 2) / (n + 1), float(lvl)])
    return pos


def compute_biomass(G: nx.DiGraph, levels: dict,
                    base: float = 1.0, efficiency: float = 0.9) -> dict:
    G_clean = G.copy()
    G_clean.remove_edges_from(nx.selfloop_edges(G_clean))
    biomass = {}
    try:
        order = list(nx.topological_sort(G_clean))
    except nx.NetworkXUnfeasible:
        order = list(G_clean.nodes())
    for node in order:
        prey = list(G_clean.predecessors(node))
        if not prey or levels.get(node, 0) == 0:
            biomass[node] = base
        else:
            biomass[node] = np.mean([biomass.get(p, base) for p in prey]) * efficiency
    return biomass


def cascade_extinctions(G_original: nx.DiGraph, removed: set) -> set:
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


NODE_COLORS = {
    "surviving":  "#2ECC71",
    "removed":    "#E74C3C",
    "cascaded":   "#E67E22",
    "historical": "#95A5A6",
}

LEGEND_PATCHES = [
    mpatches.Patch(color=NODE_COLORS["surviving"],  label="Surviving"),
    mpatches.Patch(color=NODE_COLORS["removed"],    label="Removed (this step)"),
    mpatches.Patch(color=NODE_COLORS["cascaded"],   label="Cascade extinct (this step)"),
    mpatches.Patch(color=NODE_COLORS["historical"], label="Previously removed / extinct"),
]


def draw_food_web(ax, G_original, pos, levels,
                  newly_removed, newly_cascaded, historical,
                  step, total, tier_targeted):
    surviving = (
        set(G_original.nodes()) - newly_removed - newly_cascaded - historical
    )
    color_map = {}
    for n in G_original.nodes():
        if n in newly_removed:
            color_map[n] = NODE_COLORS["removed"]
        elif n in newly_cascaded:
            color_map[n] = NODE_COLORS["cascaded"]
        elif n in historical:
            color_map[n] = NODE_COLORS["historical"]
        else:
            color_map[n] = NODE_COLORS["surviving"]

    active = surviving | newly_removed | newly_cascaded
    edges_to_draw = [
        (u, v) for u, v in G_original.edges()
        if u in active and v in active
    ]

    nx.draw_networkx_nodes(
        G_original, pos, ax=ax,
        node_color=[color_map[n] for n in G_original.nodes()],
        node_size=120, alpha=0.9
    )
    nx.draw_networkx_edges(
        G_original, pos, edgelist=edges_to_draw, ax=ax,
        width=0.4, alpha=0.25, arrows=True,
        connectionstyle="arc3,rad=0.1", arrowsize=6
    )
    nx.draw_networkx_labels(G_original, pos, ax=ax,
                             font_size=4, font_color="black")

    max_lvl = max(levels.values())
    for lvl in range(max_lvl + 1):
        ax.axhline(y=lvl, color="gray", linestyle="--", alpha=0.15)
        ax.text(-0.65, lvl, f"Tier {lvl}", va="center",
                fontsize=7, color="gray")

    removed_name = ", ".join(sorted(newly_removed))
    ax.set_title(
        f"Step {step}/{total}  |  Target tier: {tier_targeted}  |  "
        f"Removed: {removed_name}  |  "
        f"Cascade extinctions: {len(newly_cascaded)}",
        fontsize=8, pad=4
    )
    ax.axis("off")
    ax.legend(handles=LEGEND_PATCHES, loc="lower right",
              fontsize=6, framealpha=0.7)


def draw_biomass_bar(ax, G_original, levels, surviving_nodes, step):
    G_sub = G_original.subgraph(surviving_nodes).copy()
    lvl_sub = {n: levels[n] for n in surviving_nodes}
    biomass = compute_biomass(G_sub, lvl_sub)

    max_lvl = max(levels.values())
    tiers = list(range(max_lvl + 1))
    tier_biomass = [
        sum(biomass.get(n, 0)
            for n in surviving_nodes if levels.get(n) == t)
        for t in tiers
    ]

    bars = ax.bar(
        [f"Tier {t}" for t in tiers], tier_biomass,
        color="#3498DB", alpha=0.8, edgecolor="white"
    )
    ax.set_title(f"Biomass per Trophic Tier  (step {step})",
                 fontsize=8, pad=4)
    ax.set_ylabel("Total Biomass (a.u.)", fontsize=7)
    ax.tick_params(axis="x", labelsize=6, rotation=30)
    ax.tick_params(axis="y", labelsize=6)
    max_val = max(tier_biomass) if tier_biomass else 1
    for bar, val in zip(bars, tier_biomass):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max_val,
                f"{val:.2f}", ha="center", va="bottom", fontsize=5
            )


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_tiered_simulation(csv_path: str, tier_sequence: list[int],
                           seed: int, output_prefix: str = "tiered_step"):
    random.seed(seed)
    np.random.seed(seed)

    print(f"Loading graph from: {csv_path}")
    G = load_graph(csv_path)
    print(f"  {G.number_of_nodes()} species, {G.number_of_edges()} interactions")

    levels = compute_trophic_levels(G)
    nx.set_node_attributes(G, levels, "level")
    pos = hierarchical_pos(levels)

    available_tiers = sorted(set(levels.values()))
    for t in tier_sequence:
        if t not in available_tiers:
            raise ValueError(
                f"Tier {t} not found in graph.  "
                f"Available tiers: {available_tiers}"
            )

    print(f"\nTier removal sequence: {tier_sequence}")

    cumulative_extinct: set = set()
    historical: set = set()
    saved_figs = []
    total = len(tier_sequence)

    for step, tier_targeted in enumerate(tier_sequence, 1):
        # Candidates: species in this tier that are still alive
        candidates = [
            n for n, lvl in levels.items()
            if lvl == tier_targeted and n not in cumulative_extinct
        ]

        if not candidates:
            print(f"\nStep {step}: Tier {tier_targeted} has no surviving "
                  f"species — skipping.")
            continue

        species = random.choice(candidates)
        newly_removed = {species}

        all_extinct_after = cascade_extinctions(
            G, cumulative_extinct | newly_removed
        )
        newly_cascaded = all_extinct_after - cumulative_extinct - newly_removed
        surviving = set(G.nodes()) - all_extinct_after

        print(f"\nStep {step}: tier {tier_targeted} → remove '{species}'")
        print(f"  Cascade extinctions: {sorted(newly_cascaded)}")
        print(f"  Surviving species:   {len(surviving)}/{G.number_of_nodes()}")

        fig, (ax_web, ax_bar) = plt.subplots(
            1, 2,
            figsize=(18, 9),
            gridspec_kw={"width_ratios": [3, 1]}
        )
        fig.subplots_adjust(wspace=0.05)

        draw_food_web(
            ax_web, G, pos, levels,
            newly_removed=newly_removed,
            newly_cascaded=newly_cascaded,
            historical=historical,
            step=step,
            total=total,
            tier_targeted=tier_targeted,
        )
        draw_biomass_bar(ax_bar, G, levels, surviving, step)

        fname = f"{output_prefix}_{step:02d}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_figs.append(fname)
        print(f"  Saved → {fname}")

        historical = all_extinct_after.copy()
        cumulative_extinct = all_extinct_after.copy()

    print(f"\nDone.  {len(saved_figs)} image(s) saved.")
    return saved_figs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Simulate tier-targeted species removals in a tiered food web.\n"
            "Each value in --tiers specifies which tier to remove a random "
            "species from at that step."
        )
    )
    parser.add_argument("--csv", required=True,
                        help="Path to the adjacency matrix CSV "
                             "(e.g. data/FW_012_02.csv)")
    parser.add_argument("--tiers", nargs="+", type=int, required=True,
                        metavar="T",
                        help="Sequence of tier numbers to target, e.g. "
                             "--tiers 2 1 3  or  --tiers 0 0 1 2")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--prefix", default="tiered_step",
                        help="Output filename prefix (default: tiered_step)")
    args = parser.parse_args()

    run_tiered_simulation(
        csv_path=args.csv,
        tier_sequence=args.tiers,
        seed=args.seed,
        output_prefix=args.prefix,
    )