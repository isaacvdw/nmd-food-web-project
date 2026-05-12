"""
food_web_removal_simulation.py
=================================
Randomly removes X species from the tiered food web one at a time.

For each removal step:
  - Draws the tiered food web with colour-coded nodes:
      GREEN  = surviving species
      RED    = directly removed species (this step)
      ORANGE = cascaded extinct (lost all food sources)
      GRAY   = previously removed / already extinct

  - Draws a bar chart of total biomass per trophic tier after that removal.

Cascade rule
  A species goes extinct if ALL of its prey (food sources) have been
  removed or are already extinct.  The cascade is iterative: a newly
  extinct species may trigger further extinctions up the chain.

Biomass model
  Tier-0 species have a base biomass of 1.0.
  Every other species has biomass = average biomass of its prey × 0.9
  (a simple 10% loss per trophic level).  After removals the surviving
  graph is used to recompute.

Usage
  python simple_cascade_removal.py --csv data/FW_012_02.csv --n 5
  python simple_cascade_removal.py --csv data/FW_012_02.csv --n 5 --seed 99
"""

import argparse
import random
import copy

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# 1.  Graph construction helpers
# ---------------------------------------------------------------------------

def load_graph(csv_path: str) -> nx.DiGraph:
    """Load adjacency matrix CSV and return a DiGraph (prey -> predator)."""
    df = pd.read_csv(csv_path, index_col=0)
    G = nx.DiGraph()
    prey_nodes = df.index.tolist()
    predator_nodes = df.columns.tolist()
    rows, cols = np.where(df.values > 0)
    G.add_edges_from((prey_nodes[r], predator_nodes[c]) for r, c in zip(rows, cols))
    return G


def compute_trophic_levels(G: nx.DiGraph) -> dict:
    """Assign integer trophic levels (tier 0 = basal, higher = apex)."""
    G_clean = G.copy()
    G_clean.remove_edges_from(nx.selfloop_edges(G_clean))

    scc = list(nx.strongly_connected_components(G_clean))
    G_dag = nx.condensation(G_clean, scc)

    levels_dag = {}
    for node in nx.topological_sort(G_dag):
        preds = list(G_dag.predecessors(node))
        levels_dag[node] = 0 if not preds else max(levels_dag[p] for p in preds) + 1

    levels = {}
    for i, comp in enumerate(scc):
        for node in comp:
            levels[node] = levels_dag[i]
    return levels


def hierarchical_pos(levels: dict) -> dict:
    """Return (x, y) positions for a tiered layout."""
    nodes_by_level: dict = {}
    for node, lvl in levels.items():
        nodes_by_level.setdefault(lvl, []).append(node)

    pos = {}
    for lvl, nodes in nodes_by_level.items():
        n = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            pos[node] = np.array([(i - n / 2) / (n + 1), float(lvl)])
    return pos


# ---------------------------------------------------------------------------
# 2.  Biomass model
# ---------------------------------------------------------------------------

# 

def compute_biomass(G: nx.DiGraph, levels: dict, base: float = 1.0) -> dict:
    """
    Modified biomass model: All surviving species are initialized 
    with a constant biomass of 1.0.
    """
    biomass = {}
    
    for node in G.nodes():
        biomass[node] = base

    return biomass


# ---------------------------------------------------------------------------
# 3.  Cascade extinction logic
# ---------------------------------------------------------------------------

def cascade_extinctions(G_original: nx.DiGraph, removed: set) -> set:
    """
    Given the full original graph and a set of already-removed/extinct
    nodes, return the *full* set of extinct nodes (including cascades).

    Rule: a species goes extinct if ALL its prey in the original graph
    are removed / extinct.  Iterates until stable.
    """
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
# 4.  Plotting helpers
# ---------------------------------------------------------------------------

NODE_COLORS = {
    "surviving":  "#2ECC71",   # green
    "removed":    "#E74C3C",   # red   (just removed this step)
    "cascaded":   "#E67E22",   # orange
    "historical": "#95A5A6",   # gray
}

LEGEND_PATCHES = [
    mpatches.Patch(color=NODE_COLORS["surviving"],  label="Surviving"),
    mpatches.Patch(color=NODE_COLORS["removed"],    label="Removed (this step)"),
    mpatches.Patch(color=NODE_COLORS["cascaded"],   label="Cascade extinct (this step)"),
    mpatches.Patch(color=NODE_COLORS["historical"], label="Previously removed / extinct"),
]


def draw_food_web(ax, G_original, pos, levels,
                  newly_removed: set, newly_cascaded: set,
                  historical: set, step: int, total: int):
    """Draw the food web on `ax` with coloured nodes."""
    surviving = (set(G_original.nodes())
                 - newly_removed - newly_cascaded - historical)

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

    node_colors = [color_map[n] for n in G_original.nodes()]

    # Draw edges only between surviving/newly-removed nodes for clarity
    active = surviving | newly_removed | newly_cascaded
    edges_to_draw = [(u, v) for u, v in G_original.edges()
                     if u in active and v in active]

    nx.draw_networkx_nodes(G_original, pos, ax=ax,
                           node_color=node_colors, node_size=120, alpha=0.9)
    nx.draw_networkx_edges(G_original, pos, edgelist=edges_to_draw, ax=ax,
                           width=0.4, alpha=0.25, arrows=True,
                           connectionstyle="arc3,rad=0.1",
                           arrowsize=6)
    nx.draw_networkx_labels(G_original, pos, ax=ax,
                            font_size=4, font_color="black")

    max_lvl = max(levels.values())
    for lvl in range(max_lvl + 1):
        ax.axhline(y=lvl, color="gray", linestyle="--", alpha=0.15)
        ax.text(-0.65, lvl, f"Tier {lvl}", va="center",
                fontsize=7, color="gray")

    removed_name = ", ".join(sorted(newly_removed))
    cascaded_n = len(newly_cascaded)
    ax.set_title(
        f"Step {step}/{total}  |  Removed: {removed_name}  |  "
        f"Cascade extinctions: {cascaded_n}",
        fontsize=8, pad=4
    )
    ax.axis("off")
    ax.legend(handles=LEGEND_PATCHES, loc="lower right",
              fontsize=6, framealpha=0.7)


def draw_biomass_bar(ax, G_surviving, levels_original, surviving_nodes,
                     step: int):
    """Bar chart of total biomass per trophic tier for surviving nodes."""
    G_sub = G_surviving.subgraph(surviving_nodes).copy()
    lvl_sub = {n: levels_original[n] for n in surviving_nodes}

    biomass = compute_biomass(G_sub, lvl_sub)

    max_lvl = max(levels_original.values())
    tiers = list(range(max_lvl + 1))
    tier_biomass = []
    for t in tiers:
        nodes_in_tier = [n for n in surviving_nodes
                         if levels_original.get(n) == t]
        tier_biomass.append(sum(biomass.get(n, 0) for n in nodes_in_tier))

    bars = ax.bar([f"Tier {t}" for t in tiers], tier_biomass,
                  color="#3498DB", alpha=0.8, edgecolor="white")
    ax.set_title(f"Biomass per Trophic Tier  (step {step})",
                 fontsize=8, pad=4)
    ax.set_ylabel("Total Biomass (a.u.)", fontsize=7)
    ax.tick_params(axis="x", labelsize=6, rotation=30)
    ax.tick_params(axis="y", labelsize=6)

    for bar, val in zip(bars, tier_biomass):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(tier_biomass, default=1),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=5)


# ---------------------------------------------------------------------------
# 5.  Main simulation
# ---------------------------------------------------------------------------

def run_simulation(csv_path: str, n_removals: int, seed: int,
                   output_prefix: str = "removal_step"):
    random.seed(seed)
    np.random.seed(seed)

    print(f"Loading graph from: {csv_path}")
    G = load_graph(csv_path)
    print(f"  {G.number_of_nodes()} species, {G.number_of_edges()} interactions")

    levels = compute_trophic_levels(G)
    nx.set_node_attributes(G, levels, "level")
    pos = hierarchical_pos(levels)

    all_nodes = list(G.nodes())
    if n_removals > len(all_nodes):
        raise ValueError(f"Cannot remove {n_removals} species from a web "
                         f"with only {len(all_nodes)} species.")

    candidates = all_nodes.copy()
    random.shuffle(candidates)
    species_to_remove = candidates[:n_removals]

    print(f"\nSpecies selected for removal (in order):")
    for i, s in enumerate(species_to_remove, 1):
        print(f"  {i}. {s}  (Tier {levels[s]})")

    historical: set = set()        # all extinct/removed so far *before* this step
    cumulative_extinct: set = set()

    saved_figs = []

    for step, species in enumerate(species_to_remove, 1):
        newly_removed = {species}

        # Run cascade from the full current extinction set + new removal
        all_extinct_after = cascade_extinctions(
            G, cumulative_extinct | newly_removed
        )
        newly_cascaded = all_extinct_after - cumulative_extinct - newly_removed

        surviving = set(G.nodes()) - all_extinct_after

        print(f"\nStep {step}: remove '{species}'  "
              f"(Tier {levels[species]})")
        print(f"  Cascade extinctions: {sorted(newly_cascaded)}")
        print(f"  Surviving species:   {len(surviving)}/{G.number_of_nodes()}")

        # ---- Figure layout: food web (left) + biomass bar (right) ----
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
            total=n_removals
        )

        draw_biomass_bar(ax_bar, G, levels, surviving, step)

        fname = f"{output_prefix}_{step:02d}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_figs.append(fname)
        print(f"  Saved → {fname}")

        # Update state for the next step
        historical = all_extinct_after.copy()
        cumulative_extinct = all_extinct_after.copy()

    print(f"\nDone.  {len(saved_figs)} image(s) saved.")
    return saved_figs


# ---------------------------------------------------------------------------
# 6.  CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate random species removals in a tiered food web."
    )
    parser.add_argument("--csv",    required=True,
                        help="Path to the adjacency matrix CSV "
                             "(e.g. data/FW_012_02.csv)")
    parser.add_argument("--n",      type=int, default=5,
                        help="Number of species to remove (default: 5)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--prefix", default="removal_step",
                        help="Output filename prefix (default: removal_step)")
    args = parser.parse_args()

    run_simulation(
        csv_path=args.csv,
        n_removals=args.n,
        seed=args.seed,
        output_prefix=args.prefix,
    )