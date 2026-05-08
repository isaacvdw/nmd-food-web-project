"""
combined_tiered_glv.py
======================
Combines tiered food web visualization with GLV dynamics and timed species removal.

Features:
  - Tiered food web layout (trophic levels on y-axis)
  - Snapshots every N time units (default 100)
  - Left panel: Network with node colors (alive, removed, secondary extinct, natural extinct)
  - Right panel: Total biomass per tier over time
  - Command-line arguments for removal specifications
  
Usage:
  # Baseline only
  python combined_tiered_glv.py
  
  # Remove 2 species from tier 0 at t=100, then 1 from tier 1 at t=200
  python combined_tiered_glv.py --removals "2,0,100 1,1,200"
  
  # Custom parameters
  python combined_tiered_glv.py --removals "1,0,150" --t-end 400 --interval 50
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.integrate import solve_ivp
from collections import defaultdict


# =============================================================================
#  SECTION 1 — PARAMETERS & PATHS
# =============================================================================

# Model parameters
R_BASAL = 2.0
K_CARRYING = 1.0
D_DEATH = 0.005
E_EFFICIENCY = 0.2
H_HANDLING = 0.2
ATTACK_SCALING = 0.15


NORMALISE = False
N_TIMEPOINTS = 2000
EXTINCTION_THRESHOLD = 1e-10

CSV_PATH = 'data/FW_003_NAMED.csv'
SPECIES_PATH = 'data/FW_003-species.csv'
OUTPUT_DIR = 'output_combined_glv_FW003'

# Visualization parameters
SNAPSHOT_INTERVAL = 100  # Default interval (can be overridden)

# Color scheme
COL_SURVIVED = '#2ecc71'      # green
COL_REMOVED = '#e74c3c'       # red
COL_SECONDARY_EXT = '#e67e22' # orange (secondary extinction from removal)
COL_NATURAL_EXT = '#95a5a6'   # grey (naturally extinct during spin-up)


# =============================================================================
#  SECTION 2 — COMMAND-LINE ARGUMENT PARSING
# =============================================================================

def parse_removal_specifications(removal_specs_str):
    """
    Parse removal specifications from command-line argument.
    
    Format: "num_species,tier,time num_species,tier,time ..."
    Example: "2,0,100 1,1,200" means:
      - At t=100: remove 2 species from tier 0
      - At t=200: remove 1 species from tier 1
    
    Returns
    -------
    removals : list of dicts
        [{'num': 2, 'tier': 0, 'time': 100}, {'num': 1, 'tier': 1, 'time': 200}]
    """
    if not removal_specs_str or removal_specs_str.strip() == '':
        return []
    
    removals = []
    for spec in removal_specs_str.strip().split():
        parts = spec.split(',')
        if len(parts) != 3:
            print(f"Warning: invalid removal spec '{spec}', skipping")
            continue
        try:
            num_sp = int(parts[0])
            tier = int(parts[1])
            time = float(parts[2])
            removals.append({'num': num_sp, 'tier': tier, 'time': time})
        except ValueError as e:
            print(f"Warning: could not parse removal spec '{spec}': {e}")
    
    return sorted(removals, key=lambda x: x['time'])


# =============================================================================
#  SECTION 3 — TROPHIC LEVEL COMPUTATION
# =============================================================================

def compute_trophic_levels(G):
    """
    Assign each node a trophic tier via longest-path layering on a DAG.
    Self-loops removed first; SCCs collapsed so graph is acyclic.
    Tier 0 = basal producers; higher tiers = higher predators.
    """
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
    species = list(dict.fromkeys(prey_names + pred_names))
    n = len(species)
    idx = {sp: i for i, sp in enumerate(species)}

    W = np.zeros((n, n))
    for r, prey in enumerate(prey_names):
        for c, pred in enumerate(pred_names):
            val = diet_df.iloc[r, c]
            if val > 0:
                W[idx[prey], idx[pred]] = float(val)

    kingdom_map = dict(zip(attr_df['Specie'], attr_df['Kingdom']))

    G = nx.DiGraph()
    G.add_nodes_from(species)
    nx.set_node_attributes(
        G, {sp: kingdom_map.get(sp, 'Unknown') for sp in species}, 'Kingdom'
    )

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
    return W * ATTACK_SCALING


def build_params(species, is_basal):
    n = len(species)
    return {
        'r': np.where(is_basal, R_BASAL, 0.0),
        'K': np.full(n, K_CARRYING),
        'd': np.where(is_basal, 0.0, D_DEATH),
        'e': E_EFFICIENCY,
        'h': H_HANDLING,
        'B0': np.where(is_basal, 1.0, 0.5),
        'extinction_threshold': EXTINCTION_THRESHOLD
    }


# =============================================================================
#  SECTION 6 — ODE (Type II GLV)
# =============================================================================

def make_rhs(A, is_basal, params, forced_extinct=None):
    r, K, d = params['r'], params['K'], params['d']
    e, h = params['e'], params['h']
    thresh = params['extinction_threshold']
    forced = set() if forced_extinct is None else forced_extinct

    def rhs(t, B):
        B = B.copy()
        B[B < thresh] = 0.0
        for fi in forced:
            B[fi] = 0.0

        dB = np.zeros_like(B)
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


def run_simulation(A, is_basal, params, removal_events, t_end):
    """
    Run GLV simulation with timed species removals.
    
    Parameters
    ----------
    removal_events : list of dicts
        Each dict: {'time': float, 'species_indices': set of int, 'names': list}
    """
    B0 = params['B0'].copy()
    currently_removed = set()
    
    def rhs_with_removals(t, B):
        B = B.copy()
        B[B < EXTINCTION_THRESHOLD] = 0.0
        
        # Check if any removal events occur at this time
        for event in removal_events:
            if abs(t - event['time']) < 1e-3:  # close enough to event time
                currently_removed.update(event['species_indices'])
                print(f"    [t={t:.2f}] Removing species: {event['names']}")
        
        # Force removed species to zero
        for ri in currently_removed:
            B[ri] = 0.0
        
        return make_rhs(A, is_basal, params, forced_extinct=currently_removed)(t, B)
    
    t_eval = np.linspace(0, t_end, N_TIMEPOINTS)
    sol = solve_ivp(
        rhs_with_removals, (0, t_end), B0,
        method='RK45', t_eval=t_eval, rtol=1e-6, atol=1e-9,
    )
    B = sol.y
    B[B < EXTINCTION_THRESHOLD] = 0.0
    return sol.t, B, currently_removed


def who_survived(B, params):
    """Boolean mask: True for species alive at the end of the simulation."""
    return B[:, -1] > params['extinction_threshold']


# =============================================================================
#  SECTION 7 — BIOMASS ANALYSIS
# =============================================================================

def compute_tier_biomass(B, species, levels):
    """
    Compute total biomass for each trophic tier at each time point.
    
    Parameters
    ----------
    B       : np.ndarray, shape (n_species, n_timepoints)
    species : list[str]
    levels  : dict, species → tier
    
    Returns
    -------
    tier_biomass : dict
        tier_id → np.ndarray of biomass at each timepoint
    """
    tier_biomass = defaultdict(lambda: np.zeros(B.shape[1]))
    
    for i, sp in enumerate(species):
        tier = levels[sp]
        tier_biomass[tier] += B[i, :]
    
    return dict(tier_biomass)


# =============================================================================
#  SECTION 8 — VISUALIZATION
# =============================================================================

def plot_snapshot(G, species, B_snapshot, B_full, t_snapshot, t_full, params, 
                  levels, natural_ext_set, removed_species_set, 
                  out_dir, snapshot_num):
    """
    Create a two-panel figure showing network state and tier biomass.
    
    Parameters
    ----------
    B_snapshot      : np.ndarray, biomass at snapshot time
    B_full          : np.ndarray, full biomass trajectory
    t_snapshot      : float, current snapshot time
    t_full          : np.ndarray, full time array
    removed_species_set : set of species indices that were explicitly removed
    """
    sp_idx = {sp: i for i, sp in enumerate(species)}
    pos = get_tiered_pos(G, levels)
    natural_ext = natural_ext_set or set()
    
    # Determine status at snapshot time
    alive_now = B_snapshot > EXTINCTION_THRESHOLD
    
    # Determine node colors
    colors = []
    for nd in G.nodes():
        if nd in natural_ext:
            colors.append(COL_NATURAL_EXT)
        elif sp_idx.get(nd, -1) in removed_species_set:
            colors.append(COL_REMOVED)
        else:
            idx = sp_idx.get(nd, -1)
            if idx >= 0 and alive_now[idx]:
                colors.append(COL_SURVIVED)
            else:
                colors.append(COL_SECONDARY_EXT)
    
    # Count for legend
    n_alive = sum(1 for nd in G.nodes() 
                  if nd not in natural_ext 
                  and sp_idx.get(nd, -1) not in removed_species_set
                  and sp_idx.get(nd, -1) >= 0 
                  and alive_now[sp_idx[nd]])
    n_secondary_ext = sum(1 for nd in G.nodes()
                         if nd not in natural_ext
                         and sp_idx.get(nd, -1) not in removed_species_set
                         and (sp_idx.get(nd, -1) < 0 or not alive_now[sp_idx[nd]]))
    n_removed = len(removed_species_set)
    n_natural = len(natural_ext)
    
    legend_items = [
        Patch(color=COL_SURVIVED, label=f'Alive ({n_alive})'),
        Patch(color=COL_SECONDARY_EXT, label=f'Secondary extinct ({n_secondary_ext})'),
        Patch(color=COL_REMOVED, label=f'Removed ({n_removed})'),
        Patch(color=COL_NATURAL_EXT, label=f'Naturally extinct ({n_natural})'),
    ]
    
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1], figure=fig)
    
    # Left panel: network
    ax_net = fig.add_subplot(gs[0])
    nx.draw_networkx_nodes(G, pos, node_color=colors, alpha=0.88, ax=ax_net)
    nx.draw_networkx_edges(
        G, pos, width=0.5, alpha=0.3,
        arrows=True, arrowsize=8,
        connectionstyle='arc3,rad=0.1', ax=ax_net,
    )
    nx.draw_networkx_labels(G, pos, font_size=5, ax=ax_net)
    
    # Tier lines and labels
    for lvl in range(max(levels.values()) + 1):
        ax_net.axhline(y=lvl, color='gray', linestyle='--', alpha=0.15, lw=0.8)
        ax_net.text(-0.7, lvl, f'Tier {lvl}', va='center', fontsize=8, color='gray')
    
    ax_net.legend(handles=legend_items, loc='upper right', fontsize=9)
    ax_net.set_title(f'Food web state at t = {t_snapshot:.1f}', fontsize=12)
    ax_net.axis('off')
    
    # Right panel: tier biomass trajectory
    ax_biomass = fig.add_subplot(gs[1])
    tier_biomass = compute_tier_biomass(B_full, species, levels)
    
    cmap = plt.cm.tab20
    for tier_id in sorted(tier_biomass.keys()):
        biomass = tier_biomass[tier_id]
        ax_biomass.plot(
            t_full, biomass,
            color=cmap(tier_id % 20),
            lw=2,
            label=f'Tier {tier_id}',
        )
    
    # Mark current time
    ax_biomass.axvline(x=t_snapshot, color='red', linestyle='--', alpha=0.5, lw=2)
    
    ax_biomass.set_xlabel('Time', fontsize=11)
    ax_biomass.set_ylabel('Total tier biomass', fontsize=11)
    ax_biomass.set_title('Tier biomass over time', fontsize=12)
    ax_biomass.legend(fontsize=9, loc='best')
    ax_biomass.grid(True, alpha=0.3)
    ax_biomass.set_xlim(left=0)
    
    plt.suptitle(f'Snapshot {snapshot_num} (t = {t_snapshot:.1f})', 
                 fontsize=14, y=1.00)
    plt.tight_layout()
    
    # Save
    path = os.path.join(out_dir, f'snapshot_{snapshot_num:03d}_t{t_snapshot:.0f}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='GLV food-web with tiered visualization and timed removals.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Baseline only (no removals)
  python combined_tiered_glv.py
  
  # Remove 2 species from tier 0 at t=100
  python combined_tiered_glv.py --removals "2,0,100"
  
  # Remove at t=100 and t=200
  python combined_tiered_glv.py --removals "2,0,100 1,1,200"
  
  # Custom time and interval
  python combined_tiered_glv.py --removals "1,0,150" --t-end 400 --interval 50
        ''',
    )
    parser.add_argument(
        '--removals',
        type=str,
        default='',
        help='Removal specs: "num,tier,time num,tier,time ..." (e.g., "2,0,100")',
    )
    parser.add_argument(
        '--t-end',
        type=float,
        default=500,
        help='Total simulation time',
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=100,
        help='Snapshot interval (every N time units)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=OUTPUT_DIR,
        help='Output directory',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    
    args = parser.parse_args()
    
    rng = np.random.default_rng(args.seed)
    out_dir = args.output_dir
    t_end = args.t_end
    snapshot_interval = args.interval
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Parse removal specifications
    removal_specs = parse_removal_specifications(args.removals)
    
    # Load food web
    print('Loading food web …')
    G, species, W, kingdom_map, levels = load_food_web(CSV_PATH, SPECIES_PATH)
    is_basal = get_basal_species(G, species)
    A = build_attack_matrix(W)
    params = build_params(species, is_basal)
    
    print(f'  Species         : {len(species)}')
    print(f'  Interactions    : {G.number_of_edges()}')
    print(f'  Basal producers : {is_basal.sum()}')
    print(f'  Trophic tiers   : {min(levels.values())} – {max(levels.values())}')
    
    # Spin-up to steady state
    print(f'\nSpinning up to steady state …')
    t_spinup, B_spinup, _ = run_simulation(A, is_basal, params, [], 200)
    params['B0'] = B_spinup[:, -1]
    baseline_alive = who_survived(B_spinup, params)
    
    natural_ext_set = {
        species[i] for i, sp in enumerate(species)
        if not baseline_alive[i]
    }
    
    print(f'  Alive at steady state: {baseline_alive.sum()} / {len(species)}')
    if natural_ext_set:
        print(f'  Naturally extinct: {sorted(natural_ext_set)}')
    
    # Convert removal specs to removal events
    removal_events = []
    sp_idx = {sp: i for i, sp in enumerate(species)}
    removed_species_indices = set()
    
    if removal_specs:
        print(f'\nPlanning {len(removal_specs)} removal event(s) …')
        for spec in removal_specs:
            tier = spec['tier']
            n_to_remove = spec['num']
            time = spec['time']
            
            # Get alive species in this tier
            tier_species = [
                (sp, sp_idx[sp]) for sp in species
                if levels[sp] == tier and baseline_alive[sp_idx[sp]]
            ]
            
            if len(tier_species) < n_to_remove:
                print(f'  Warning: only {len(tier_species)} alive in tier {tier}, '
                      f'requested {n_to_remove}')
                n_to_remove = len(tier_species)
            
            # Select species
            selected_idx = rng.choice(len(tier_species), size=n_to_remove, replace=False)
            selected_species = [tier_species[i] for i in selected_idx]
            species_indices = {sp_i for sp, sp_i in selected_species}
            species_names = [sp for sp, sp_i in selected_species]
            
            removed_species_indices.update(species_indices)
            
            removal_events.append({
                'time': time,
                'species_indices': species_indices,
                'names': species_names,
            })
            
            print(f'  t={time}: remove {n_to_remove} from tier {tier}: {species_names}')
    
    # Run simulation
    print(f'\nRunning {t_end}-unit simulation …')
    t_sim, B_sim, removed_at_end = run_simulation(A, is_basal, params, removal_events, t_end)
    
    # Generate snapshots
    print('\nGenerating snapshots …')
    snapshot_times = np.arange(0, t_end + snapshot_interval, snapshot_interval)
    snapshot_num = 0
    
    for snap_time in snapshot_times:
        if snap_time > t_end:
            break
        
        # Find closest timepoint
        idx = np.argmin(np.abs(t_sim - snap_time))
        B_snap = B_sim[:, idx]
        
        snapshot_num += 1
        plot_snapshot(
            G, species, B_snap, B_sim, t_sim[idx], t_sim,
            params, levels, natural_ext_set, removed_species_indices,
            out_dir, snapshot_num
        )
    
    print(f'\n✓ Simulation complete!')
    print(f'  Output directory: {out_dir}/')
    print(f'  Total snapshots: {snapshot_num}')
