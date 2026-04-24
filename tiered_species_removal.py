"""
glv_tiered_removal.py
=====================
GLV (Generalised Lotka-Volterra) food-web simulation with:
  - Tiered (trophic-level) network layout
  - Single-species removal experiments
  - Cascading multi-species removal (sequential, each building on the last)

OUTPUT
------
  output_images/
      baseline_tiered.png
      removal_<SpeciesName>.png          (single removal)
      cascade_removal_step_<N>_<Name>.png  (one per step in cascade)
  impact_ranking.csv

DATA FILES
----------
  data/FW_008.csv          diet/interaction matrix  (prey = rows, predator = cols)
  data/FW_008-species.csv  species attributes; must contain 'Specie' and 'Kingdom' columns
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.integrate import solve_ivp
from collections import defaultdict


# =============================================================================
#  SECTION 1 — PATHS & CONFIGURATION
# =============================================================================

CSV_PATH     = 'data/FW_003_NAMED.csv'
SPECIES_PATH = 'data/FW_003-species.csv'
OUTPUT_DIR   = 'output_images_FW_003'

# Single-species removal: set to a species name or None to pick randomly
SINGLE_REMOVAL_TARGET = None

# Cascading removal: how many species to remove sequentially
N_CASCADE_REMOVALS = 3

# Set a random seed for reproducibility (None = fully random)
RANDOM_SEED = 42


# =============================================================================
#  SECTION 2 — MODEL PARAMETERS
# =============================================================================

R_BASAL              = 2.0
K_CARRYING           = 1.0
D_DEATH              = 0.005
E_EFFICIENCY         = 0.2
H_HANDLING           = 0.2
NORMALISE            = False
T_END                = 500
N_TIMEPOINTS         = 2000
EXTINCTION_THRESHOLD = 1e-6


# =============================================================================
#  SECTION 3 — TROPHIC TIER COMPUTATION
# =============================================================================

def compute_trophic_levels(G):
    """
    Assign each node a trophic tier via longest-path layering on a DAG.

    Self-loops are removed first, then strongly connected components are
    collapsed so the remaining graph is acyclic. Tier 0 = basal producers;
    higher tiers = higher predators.
    """
    G_clean = G.copy()
    G_clean.remove_edges_from(nx.selfloop_edges(G_clean))

    scc      = list(nx.strongly_connected_components(G_clean))
    G_dag    = nx.condensation(G_clean, scc)
    levels_dag = {}

    for node in nx.topological_sort(G_dag):
        preds = list(G_dag.predecessors(node))
        levels_dag[node] = 0 if not preds else max(levels_dag[p] for p in preds) + 1

    levels = {}
    for i, comp in enumerate(scc):
        for node in comp:
            levels[node] = levels_dag[i]

    return levels


def get_tiered_pos(G, levels, x_spread=1.0):
    """Return a deterministic {node: (x, y)} layout with tiers on the y-axis."""
    nodes_by_level = defaultdict(list)
    for node, lvl in levels.items():
        nodes_by_level[lvl].append(node)

    pos = {}
    for lvl, nodes in nodes_by_level.items():
        nodes_sorted = sorted(nodes)
        width = len(nodes_sorted)
        for i, node in enumerate(nodes_sorted):
            x = x_spread * (i - (width - 1) / 2) / max(width, 1)
            pos[node] = np.array([x, float(lvl)])
    return pos


# =============================================================================
#  SECTION 4 — DATA LOADING
# =============================================================================

def load_food_web(csv_path, species_csv_path):
    """
    Load the diet matrix and species attributes.

    Returns
    -------
    G           : nx.DiGraph  (prey → predator, weighted)
    species     : list[str]
    W           : np.ndarray  shape (n, n), W[prey, predator]
    kingdom_map : dict        species → Kingdom string
    levels      : dict        species → trophic tier
    """
    diet_df = pd.read_csv(csv_path, index_col=0)
    attr_df = pd.read_csv(species_csv_path)

    prey_names = diet_df.index.tolist()
    pred_names = diet_df.columns.tolist()
    species    = list(dict.fromkeys(prey_names + pred_names))
    n          = len(species)
    idx        = {sp: i for i, sp in enumerate(species)}

    W = np.zeros((n, n))
    for r, prey in enumerate(prey_names):
        for c, pred in enumerate(pred_names):
            val = diet_df.iloc[r, c]
            if val > 0:
                W[idx[prey], idx[pred]] = float(val)

    kingdom_map = dict(zip(attr_df['Specie'], attr_df['Kingdom']))

    G = nx.DiGraph()
    G.add_nodes_from(species)
    nx.set_node_attributes(G, {sp: kingdom_map.get(sp, 'Unknown') for sp in species}, 'Kingdom')

    for i, sp_i in enumerate(species):
        for j, sp_j in enumerate(species):
            if W[i, j] > 0:
                G.add_edge(sp_i, sp_j, weight=W[i, j])

    levels = compute_trophic_levels(G)
    nx.set_node_attributes(G, levels, 'level')

    return G, species, W, kingdom_map, levels


# =============================================================================
#  SECTION 5 — MODEL SETUP
# =============================================================================

def get_basal_species(G, species):
    return np.array([G.in_degree(sp) == 0 for sp in species], dtype=bool)


def build_attack_matrix(W):
    A = W.copy()
    if NORMALISE:
        col_sums = A.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        A = A / col_sums[np.newaxis, :]
    return A


def build_params(species, is_basal):
    n = len(species)
    return {
        'r'  : np.where(is_basal, R_BASAL,     0.0),
        'K'  : np.full(n, K_CARRYING),
        'd'  : np.where(is_basal, 0.0,         D_DEATH),
        'e'  : E_EFFICIENCY,
        'h'  : H_HANDLING,
        'B0' : np.where(is_basal, 1.0,         0.5),
        'extinction_threshold': EXTINCTION_THRESHOLD,
    }


# =============================================================================
#  SECTION 6 — ODE (Type II GLV)
# =============================================================================

def make_rhs(A, is_basal, params, forced_extinct=None):
    r, K, d = params['r'], params['K'], params['d']
    e, h    = params['e'], params['h']
    thresh  = params['extinction_threshold']
    forced  = set() if forced_extinct is None else forced_extinct

    def rhs(t, B):
        B = B.copy()
        B[B < thresh] = 0.0
        for fi in forced:
            B[fi] = 0.0

        dB    = np.zeros_like(B)
        denom = 1.0 + h * (A * B[:, np.newaxis]).sum(axis=0)

        for i in range(len(B)):
            if B[i] == 0.0:
                continue
            if is_basal[i]:
                dB[i] += r[i] * B[i] * (1.0 - B[i] / K[i])
            else:
                dB[i] -= d[i] * B[i]
                for j in range(len(B)):
                    if A[j, i] > 0 and B[j] > 0:
                        dB[i] += e * A[j, i] * B[i] * B[j] / denom[i]
            for k in range(len(B)):
                if A[i, k] > 0 and B[k] > 0:
                    dB[i] -= A[i, k] * B[k] * B[i] / denom[k]
        return dB

    return rhs


def run_simulation(A, is_basal, params, forced_extinct=None, initial_B=None):
    """
    Run the GLV ODE.

    Parameters
    ----------
    forced_extinct : set of int indices forced to zero throughout
    initial_B      : override starting biomass vector (used for cascades)
    """
    B0 = initial_B.copy() if initial_B is not None else params['B0'].copy()
    if forced_extinct:
        for fi in forced_extinct:
            B0[fi] = 0.0

    rhs    = make_rhs(A, is_basal, params, forced_extinct)
    t_eval = np.linspace(0, T_END, N_TIMEPOINTS)
    sol    = solve_ivp(rhs, (0, T_END), B0,
                       method='RK45', t_eval=t_eval,
                       rtol=1e-6, atol=1e-9)
    B = sol.y
    B[B < EXTINCTION_THRESHOLD] = 0.0
    return sol.t, B


def who_survived(B, params):
    return B[:, -1] > params['extinction_threshold']


# =============================================================================
#  SECTION 7 — REMOVAL EXPERIMENTS
# =============================================================================

def remove_species(species, target, A, is_basal, params, initial_B=None):
    """
    Remove a single species and return the resulting simulation.

    Parameters
    ----------
    initial_B : optional starting biomass vector (for cascading removals)

    Returns
    -------
    t, B, n_secondary_extinctions
    """
    idx = species.index(target)
    t, B = run_simulation(A, is_basal, params,
                          forced_extinct={idx}, initial_B=initial_B)

    alive_start = (initial_B if initial_B is not None else params['B0']) > EXTINCTION_THRESHOLD
    alive_end   = who_survived(B, params)
    n_secondary = int(np.sum(alive_start & ~alive_end)) - 1
    return t, B, max(n_secondary, 0)


def cascade_removals(species, targets, A, is_basal, params):
    """
    Remove species one at a time, each simulation starting from the
    steady-state biomass left by the previous removal.

    Parameters
    ----------
    targets : list[str]  species to remove in order

    Returns
    -------
    steps : list of dicts, one per removal step, each containing:
        'target'      : removed species name
        'step'        : 1-based step number
        't'           : time array
        'B'           : biomass matrix
        'n_secondary' : secondary extinctions at this step
        'cumulative'  : total species lost so far (including removals)
    """
    steps        = []
    current_B    = None   # None → use params['B0'] on first step
    total_lost   = 0

    for step_num, target in enumerate(targets, start=1):
        print(f'  Step {step_num}: removing {target} …')
        t, B, n_sec = remove_species(species, target, A, is_basal, params,
                                     initial_B=current_B)
        total_lost += 1 + n_sec
        steps.append({
            'target'     : target,
            'step'       : step_num,
            't'          : t,
            'B'          : B,
            'n_secondary': n_sec,
            'cumulative' : total_lost,
        })
        # Carry the end-state biomass into the next step
        current_B = B[:, -1].copy()

    return steps


def rank_all_removals(species, A, is_basal, params):
    """Return a DataFrame of all species ranked by secondary extinctions caused."""
    n       = len(species)
    results = []
    for sp in species:
        _, _, n_sec = remove_species(species, sp, A, is_basal, params)
        results.append({
            'Species removed'       : sp,
            'Secondary extinctions' : n_sec,
            'Total lost'            : n_sec + 1,
            'Fraction of web lost'  : round((n_sec + 1) / n, 3),
        })
    return (pd.DataFrame(results)
              .sort_values('Secondary extinctions', ascending=False)
              .reset_index(drop=True))


# =============================================================================
#  SECTION 8 — COLOUR HELPERS
# =============================================================================

COL_REMOVED  = '#e74c3c'
COL_EXTINCT  = '#e67e22'
COL_SURVIVED = '#2ecc71'


def node_colors(G, species, B, params, removed_set):
    """
    Assign display colour to each node given a set of removed species.

    removed_set : set of species names that were explicitly removed
    """
    sp_idx    = {sp: i for i, sp in enumerate(species)}
    alive_end = who_survived(B, params)
    colors    = []
    for nd in G.nodes():
        if nd in removed_set:
            colors.append(COL_REMOVED)
        elif sp_idx.get(nd, -1) >= 0 and not alive_end[sp_idx[nd]]:
            colors.append(COL_EXTINCT)
        else:
            colors.append(COL_SURVIVED)
    return colors


# =============================================================================
#  SECTION 9 — VISUALISATION
# =============================================================================

def _draw_tiered_network(ax, G, pos, colors, levels, title, legend_handles):
    nx.draw_networkx_nodes(G, pos, node_color=colors, alpha=0.88, ax=ax)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3,
                           arrows=True, arrowsize=8,
                           connectionstyle='arc3,rad=0.1', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=5, ax=ax)

    for lvl in range(max(levels.values()) + 1):
        ax.axhline(y=lvl, color='gray', linestyle='--', alpha=0.15, lw=0.8)
        ax.text(-0.7, lvl, f'Tier {lvl}', va='center', fontsize=8, color='gray')

    ax.legend(handles=legend_handles, loc='upper right', fontsize=8)
    ax.set_title(title, fontsize=12)
    ax.axis('off')


def _draw_trajectories(ax, G, species, t, B, params, focal_species, title):
    sp_idx    = {sp: i for i, sp in enumerate(species)}
    alive_end = who_survived(B, params)
    neighbours = list(dict.fromkeys(
        list(G.predecessors(focal_species)) + list(G.successors(focal_species))
    ))

    cmap = plt.cm.tab20
    if neighbours:
        for k, sp in enumerate(neighbours[:20]):
            i = sp_idx.get(sp, -1)
            if i < 0:
                continue
            ax.plot(t, B[i],
                    color=cmap(k % 20),
                    lw=1.8 if alive_end[i] else 0.8,
                    ls='-'  if alive_end[i] else '--',
                    label=sp + ('' if alive_end[i] else ' ✗'))
    else:
        ax.text(0.5, 0.5, 'No direct neighbours',
                ha='center', va='center', transform=ax.transAxes)

    ax.set_xlabel('Time (arbitrary units)', fontsize=9)
    ax.set_ylabel('Biomass (normalised)',    fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc='upper left')


def plot_baseline_network(G, species, B, params, levels, out_dir):
    """Save a baseline tiered network (no removal)."""
    pos       = get_tiered_pos(G, levels)
    sp_idx    = {sp: i for i, sp in enumerate(species)}
    alive_end = who_survived(B, params)
    colors    = ['#2ecc71' if alive_end[sp_idx[nd]] else '#95a5a6' for nd in G.nodes()]

    fig, ax = plt.subplots(figsize=(14, 10))
    legend  = [Patch(color='#2ecc71', label='Survived'),
               Patch(color='#95a5a6', label='Naturally extinct')]
    _draw_tiered_network(ax, G, pos, colors, levels,
                         'Baseline food web — trophic tiers (no removal)', legend)
    plt.tight_layout()
    _save(fig, out_dir, 'baseline_tiered.png')


def plot_removal(G, species, t, B, params, target, levels, out_dir, prefix='removal'):
    """Save a two-panel figure for a single species removal."""
    pos       = get_tiered_pos(G, levels)
    colors    = node_colors(G, species, B, params, removed_set={target})
    alive_end = who_survived(B, params)
    sp_idx    = {sp: i for i, sp in enumerate(species)}

    n_ext = sum(1 for nd in G.nodes()
                if nd != target
                and sp_idx.get(nd, -1) >= 0
                and not alive_end[sp_idx[nd]])

    fig = plt.figure(figsize=(18, 9))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1], figure=fig)

    legend_net = [
        Patch(color=COL_REMOVED,  label=f'Removed: {target}'),
        Patch(color=COL_EXTINCT,  label=f'Extinct after removal ({n_ext})'),
        Patch(color=COL_SURVIVED, label=f'Survived ({G.number_of_nodes() - n_ext - 1})'),
    ]
    _draw_tiered_network(fig.add_subplot(gs[0]), G, pos, colors, levels,
                         f'Trophic-tier network after removing: {target}', legend_net)
    _draw_trajectories(fig.add_subplot(gs[1]), G, species, t, B, params, target,
                       f'Neighbour trajectories\n(prey & predators of {target})')

    plt.suptitle(f'GLV species-removal — {target}', fontsize=14, y=1.01)
    plt.tight_layout()
    safe = target.replace(' ', '_').replace('/', '_')
    _save(fig, out_dir, f'{prefix}_{safe}.png')


def plot_cascade_step(G, species, t, B, params, step, levels, out_dir,
                      all_removed, n_secondary):
    """Save a two-panel figure for one step of a cascading removal."""
    pos    = get_tiered_pos(G, levels)
    target = step['target']
    colors = node_colors(G, species, B, params, removed_set=set(all_removed))

    n_ext = sum(1 for nd in G.nodes()
                if nd not in all_removed
                and (sp_idx := {sp: i for i, sp in enumerate(species)}).get(nd, -1) >= 0
                and not who_survived(B, params)[sp_idx[nd]])

    fig = plt.figure(figsize=(18, 9))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1], figure=fig)

    legend_net = [
        Patch(color=COL_REMOVED,  label=f'Removed so far ({len(all_removed)})'),
        Patch(color=COL_EXTINCT,  label=f'Secondary extinct ({n_ext})'),
        Patch(color=COL_SURVIVED, label=f'Survived ({G.number_of_nodes() - n_ext - len(all_removed)})'),
    ]
    _draw_tiered_network(
        fig.add_subplot(gs[0]), G, pos, colors, levels,
        f'Cascade step {step["step"]}: removed {target}\n'
        f'(cumulative lost: {step["cumulative"]})',
        legend_net,
    )
    _draw_trajectories(
        fig.add_subplot(gs[1]), G, species, t, B, params, target,
        f'Neighbour trajectories after step {step["step"]}\n'
        f'(prey & predators of {target})',
    )

    plt.suptitle(
        f'GLV cascading removal — step {step["step"]} of {step["step"] + len(all_removed) - 1}: {target}',
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    safe = target.replace(' ', '_').replace('/', '_')
    _save(fig, out_dir, f'cascade_step{step["step"]:02d}_{safe}.png')


def _save(fig, out_dir, filename):
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == '__main__':
    rng = np.random.default_rng(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    print('Loading food web …')
    G, species, W, kingdom_map, levels = load_food_web(CSV_PATH, SPECIES_PATH)
    is_basal = get_basal_species(G, species)
    A        = build_attack_matrix(W)
    params   = build_params(species, is_basal)

    print(f'  Species         : {len(species)}')
    print(f'  Interactions    : {G.number_of_edges()}')
    print(f'  Basal producers : {is_basal.sum()}')
    print(f'  Trophic tiers   : {min(levels.values())} – {max(levels.values())}')

    # ── Baseline ─────────────────────────────────────────────────────────────
    print('\nRunning baseline simulation …')
    t_base, B_base = run_simulation(A, is_basal, params)
    print(f'  Survivors: {who_survived(B_base, params).sum()} / {len(species)}')
    plot_baseline_network(G, species, B_base, params, levels, OUTPUT_DIR)

    # ── Single-species removal ────────────────────────────────────────────────
    target = SINGLE_REMOVAL_TARGET or rng.choice(species)
    print(f'\nSingle removal: {target}')
    t_r, B_r, n_sec = remove_species(species, target, A, is_basal, params)
    plot_removal(G, species, t_r, B_r, params, target, levels, OUTPUT_DIR)
    print(f'  Secondary extinctions: {n_sec}')

    # ── Cascading multi-species removal ───────────────────────────────────────
    n_cascade = min(N_CASCADE_REMOVALS, len(species))
    cascade_targets = list(rng.choice(species, size=n_cascade, replace=False))
    print(f'\nCascading removal of {n_cascade} species: {cascade_targets}')
    steps = cascade_removals(species, cascade_targets, A, is_basal, params)

    removed_so_far = []
    for step in steps:
        removed_so_far.append(step['target'])
        plot_cascade_step(
            G, species, step['t'], step['B'], params,
            step, levels, OUTPUT_DIR,
            all_removed=list(removed_so_far),
            n_secondary=step['n_secondary'],
        )
        print(f'  Step {step["step"]} ({step["target"]}): '
              f'{step["n_secondary"]} secondary extinctions, '
              f'{step["cumulative"]} total lost so far')

    # ── Impact ranking ────────────────────────────────────────────────────────
    print('\nComputing full impact ranking …')
    ranking = rank_all_removals(species, A, is_basal, params)
    ranking.to_csv('impact_ranking.csv', index=False)
    print('Top 10:')
    print(ranking.head(10).to_string(index=False))
    print('\nFull ranking saved to: impact_ranking.csv')
    print(f'All images saved to:   {OUTPUT_DIR}/')