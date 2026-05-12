"""
GLV food-web model with bounded parameter sweep and species-removal analysis.

Loads a directed food web (prey -> predator) from CSV, fits a generalized
Lotka-Volterra model with a Type II functional response, then ranks species
by their cascade impact when removed.

Pipeline
--------
1. load_food_web()              build species list and interaction matrix W
2. basal_mask / trophic_tiers   ecological structure
3. param_sweep()                grid search over plausible GLV parameters
4. simulate()                   baseline biomass trajectory
5. removal_sweep()              per-species secondary-extinction count
6. plot_*                       baseline dynamics, impact bars, tier network
"""

import os
import itertools

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp


# =============================================================================
# Configuration
# =============================================================================

# Input files
CSV_PATH     = 'data/FW_006_NAMED.csv'
SPECIES_PATH = 'data/FW_006-species.csv'

# Run mode
REMOVAL_MODE    = 'all'      # 'all' | 'basal' | 'consumers'
RUN_PARAM_SWEEP = True

# Output
PLOT_DIR = 'plots'
PLOT_DPI = 150

# Default parameters (used only when RUN_PARAM_SWEEP = False).
# Paste the best parameters reported by the sweep here.
ATTACK_SCALING       = 0.6
D_DEATH              = 0.05
E_EFFICIENCY         = 0.30
H_HANDLING           = 0.40
CONSUMER_COMPETITION = 0.05

# Fixed dynamic parameters
R_BASAL    = 1.0     # intrinsic growth rate of basal species
K_CARRYING = 1.0     # carrying capacity of basal species
NORMALISE  = False   # column-normalise interaction matrix

# Tiering: if True, a predator can share a tier with one of its prey
# (omnivory-friendly); if False, every predator is strictly above its prey.
ALLOW_SAME_LEVEL = False

# Integration
T_END        = 500
N_TIMEPOINTS = 2000

# Thresholds
EXTINCTION_THRESHOLD = 0.005   # final biomass below this counts as extinct
NUMERICAL_FLOOR      = 1e-12   # floor used inside RHS to keep B >= 0
MAX_BIOMASS          = 8.0     # cap above which scoring penalises blow-up


# =============================================================================
# Data loading
# =============================================================================

def load_food_web(csv_path, species_path):
    """Read an interaction matrix and species table into a DiGraph (prey -> predator).

    Returns
    -------
    G        : networkx.DiGraph with 'Kingdom' on every node and 'weight' on edges
    species  : ordered list of node names
    W        : (n, n) array. W[i, j] is interaction strength from prey i to predator j
    """
    M = pd.read_csv(csv_path, index_col=0)
    species_table = pd.read_csv(species_path)

    # Preserve appearance order: rows first, then any new columns.
    species = list(dict.fromkeys(M.index.tolist() + M.columns.tolist()))
    idx = {s: i for i, s in enumerate(species)}

    n = len(species)
    W = np.zeros((n, n))
    for prey in M.index:
        for predator in M.columns:
            value = M.loc[prey, predator]
            if value > 0:
                W[idx[prey], idx[predator]] = float(value)

    kingdom = dict(zip(species_table['Specie'], species_table['Kingdom']))

    G = nx.DiGraph()
    G.add_nodes_from(species)
    for s in species:
        G.nodes[s]['Kingdom'] = kingdom.get(s, 'Unknown')
    for i, prey in enumerate(species):
        for j, pred in enumerate(species):
            if W[i, j] > 0:
                G.add_edge(prey, pred, weight=W[i, j])

    return G, species, W


# =============================================================================
# Ecological structure
# =============================================================================

def basal_mask(G, species):
    """Boolean array: True where the species has no incoming edges (no prey)."""
    return np.array([G.in_degree(s) == 0 for s in species])


def top_mask(G, species, basal):
    """Boolean array: True for non-basal species with no outgoing edges."""
    return np.array([
        (not basal[i]) and G.out_degree(s) == 0
        for i, s in enumerate(species)
    ])


def trophic_tiers(G, species, basal, allow_same_level=ALLOW_SAME_LEVEL):
    """Trophic tier by SCC condensation + longest-path on the DAG.

    Tier 0 = basal level.
    Higher tiers are placed above lower tiers.

    Cycles are collapsed into a single component before layering.
    Self-loops are removed before computing tiers.
    """
    G_clean = G.copy()
    G_clean.remove_edges_from(nx.selfloop_edges(G_clean))

    sccs = list(nx.strongly_connected_components(G_clean))
    G_dag = nx.condensation(G_clean, sccs)

    levels_dag = {}

    for node in nx.topological_sort(G_dag):
        preds = list(G_dag.predecessors(node))

        if not preds:
            levels_dag[node] = 0
        elif allow_same_level:
            levels_dag[node] = max(levels_dag[p] for p in preds)
        else:
            levels_dag[node] = max(levels_dag[p] for p in preds) + 1

    tier = {}
    for i, comp in enumerate(sccs):
        for node in comp:
            tier[node] = levels_dag[i]

    return tier


# =============================================================================
# Parameters & dynamics
# =============================================================================

def attack_matrix(W, scale):
    """Scale the raw interaction matrix; optionally column-normalise."""
    A = W.copy()
    if NORMALISE:
        col_sums = A.sum(axis=0)
        col_sums[col_sums == 0] = 1
        A = A / col_sums[None, :]
    return scale * A


def make_parameters(species, basal, tier, d, e, h, c):
    """Assemble the GLV parameter dictionary with constant per-capita rates.

    Basal species start at biomass 1.0.
    All non-basal species start at biomass 0.5.

    Dynamics themselves treat every consumer the same.
    The `tier` parameter is kept for API compatibility but is no longer used
    for setting initial biomass.
    """
    n = len(species)

    B0 = np.full(n, 0.5)
    B0[basal] = 1.0

    return dict(
        r=np.where(basal, R_BASAL, 0.0),
        K=np.full(n, K_CARRYING),
        d=np.where(basal, 0.0, d),
        e=e,
        h=h,
        c=np.where(basal, 0.0, c),
        B0=B0,
    )

def make_rhs(A, basal, p, forced_extinct=()):
    """Return dB/dt(t, B) for solve_ivp.

    Type II functional response:
        F[i, j] = a_ij B_i B_j / (1 + h * sum_k a_kj B_k)
      - prey i loses sum_j F[i, j]
      - predator j gains e * sum_i F[i, j]
    """
    forced = set(forced_extinct)

    def rhs(_t, B):
        B = np.maximum(B, NUMERICAL_FLOOR)
        for i in forced:
            B[i] = 0.0

        # Saturating denominator per predator column, shape (n,).
        denom = 1.0 + p['h'] * (A * B[:, None]).sum(axis=0)
        # Per-pair flow: F[i, j] is flow from prey i to predator j.
        F = A * B[:, None] * B[None, :] / denom[None, :]

        loss_per_prey = F.sum(axis=1)
        gain_per_pred = F.sum(axis=0)

        dB = np.zeros_like(B)
        cons = ~basal
        # Basal: logistic growth - predation loss
        dB[basal] = (
            p['r'][basal] * B[basal] * (1 - B[basal] / p['K'][basal])
            - loss_per_prey[basal]
        )
        # Consumers: -mortality - self-limitation + assimilated gain - predation loss
        dB[cons] = (
            -p['d'][cons] * B[cons]
            - p['c'][cons] * B[cons] ** 2
            + p['e'] * gain_per_pred[cons]
            - loss_per_prey[cons]
        )
        for i in forced:
            dB[i] = 0.0
        return dB

    return rhs


def simulate(A, basal, p, forced_extinct=(), t_end=T_END, n_tp=N_TIMEPOINTS):
    """Integrate the GLV system from B0; return (t, B) with B shape (n_species, n_tp)."""
    B0 = p['B0'].copy()
    for i in forced_extinct:
        B0[i] = 0.0

    sol = solve_ivp(
        make_rhs(A, basal, p, forced_extinct),
        (0, t_end), B0,
        t_eval=np.linspace(0, t_end, n_tp),
        rtol=1e-6, atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    B = sol.y
    B[B < 0] = 0
    return sol.t, B


def alive_at_end(B):
    """Boolean array: species with biomass >= EXTINCTION_THRESHOLD at the final timestep."""
    return B[:, -1] >= EXTINCTION_THRESHOLD


def stability_cv(B):
    """Mean coefficient of variation over the last 20% of the trajectory."""
    tail = B[:, int(0.8 * B.shape[1]):]
    cv = tail.std(axis=1) / (tail.mean(axis=1) + 1e-12)
    return float(np.nanmean(cv))


# =============================================================================
# Parameter sweep
# =============================================================================

def trial_diagnostics(B, basal):
    """Return diagnostics for a parameter trial.

    No score is calculated. Parameter trials are compared by:
    1. highest number of surviving species
    2. lowest maximum final biomass if survivor count is tied
    """
    survivors = alive_at_end(B)
    final_B = B[:, -1]
    cv = stability_cv(B)

    diag = dict(
        surv=int(survivors.sum()),
        total=float(survivors.mean()),
        maxB=float(final_B.max()),
        stab=cv,
    )

    return diag


def param_sweep(species, W, basal, tier):
    """Grid search over ecologically plausible GLV parameter ranges.

    The best parameter set is selected by:
    1. highest overall number of surviving species
    2. lowest maximum final biomass if survivor count is tied

    No combined scoring function is used.
    """
    grid = dict(
        attack_scaling=[0.3, 0.6, 1.0],
        d=[0.01, 0.02, 0.05],
        e=[0.20, 0.30, 0.40],
        h=[0.30, 0.50, 0.70],
        c=[0.01, 0.03, 0.05],
    )

    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    print(f"Parameter sweep: {len(combos)} combinations on a survival-friendly realistic grid")

    best_trial, best_diag = None, None

    for k, vals in enumerate(combos, 1):
        trial = dict(zip(keys, vals))

        A = attack_matrix(W, trial['attack_scaling'])
        p = make_parameters(
            species,
            basal,
            tier,
            trial['d'],
            trial['e'],
            trial['h'],
            trial['c'],
        )

        try:
            _, B = simulate(A, basal, p, t_end=300, n_tp=700)
            diag = trial_diagnostics(B, basal)
        except Exception:
            continue

        # Reject numerical failures and obvious biomass blow-up.
        valid = (
            np.all(np.isfinite(B))
            and diag['maxB'] <= MAX_BIOMASS
        )

        if not valid:
            continue

        better = (
            best_trial is None
            or diag['surv'] > best_diag['surv']
            or (
                diag['surv'] == best_diag['surv']
                and diag['maxB'] < best_diag['maxB']
            )
        )

        if better:
            best_trial, best_diag = trial, diag
            print(
                f"  [{k:4d}/{len(combos)}] "
                f"surv={diag['surv']:2d}/{len(species)} "
                f"alive={diag['total']:.2f} stab={diag['stab']:.2f} "
                f"maxB={diag['maxB']:.2f} {trial}"
            )

    if best_trial is None:
        raise RuntimeError("No parameter set worked.")

    print("BEST PARAMETERS")
    for key in keys:
        print(f"  {key:<16} = {best_trial[key]}")

    print(
        f"  survival={best_diag['surv']}/{len(species)}, "
        f"alive={best_diag['total']:.2f}, "
        f"stab={best_diag['stab']:.2f}, maxB={best_diag['maxB']:.2f}"
    )

    return best_trial


# =============================================================================
# Removal sweep
# =============================================================================

def removal_sweep(species, candidates, A, basal, p, base_alive, tier):
    """Force-extinct each candidate species in turn; count secondary extinctions."""
    rows = []
    for n, s in enumerate(candidates, 1):
        print(f"  [{n:3d}/{len(candidates)}] {s:<30}", end='')
        i = species.index(s)
        _, B = simulate(A, basal, p, forced_extinct={i})

        secondary = base_alive & ~alive_at_end(B)
        secondary[i] = False  # the removed species itself doesn't count
        n_sec = int(secondary.sum())
        n_total = n_sec + 1   # secondary + the removed species

        rows.append({
            'Species': s,
            'Tier': tier[s],
            'Type': 'Basal' if basal[i] else 'Consumer',
            'Secondary extinctions': n_sec,
            'Total lost': n_total,
            'Fraction lost': round(n_total / max(base_alive.sum(), 1), 3),
        })
    print()
    return (pd.DataFrame(rows)
              .sort_values('Secondary extinctions', ascending=False)
              .reset_index(drop=True))


def print_table(df, n=5):
    """Print the top-n / bottom-n species per tier by secondary-extinction count."""
    print("SPECIES IMPORTANCE BY TROPHIC TIER")
    for t in sorted(df.Tier.unique()):
        sub = df[df.Tier == t]
        shown = pd.concat([sub.head(n), sub.tail(n)]).drop_duplicates()
        print(f"Tier {t} ({len(sub)} species)")
        print(f"  {'Species':<28} {'Type':<10} {'Sec.ext':>7} {'Frac':>7}")
        for _, r in shown.iterrows():
            print(f"  {r.Species:<28} {r.Type:<10} "
                  f"{r['Secondary extinctions']:>7} {r['Fraction lost']:>7.3f}")


# =============================================================================
# Plotting
# =============================================================================

def hierarchical_pos(species, tier):
    """Create a simple tiered layout with tier 0 at the bottom."""
    pos = {}
    nodes_by_tier = {}

    for s in species:
        nodes_by_tier.setdefault(tier[s], []).append(s)

    for t, nodes in nodes_by_tier.items():
        nodes = sorted(nodes)
        width = len(nodes)

        for i, s in enumerate(nodes):
            x = (i - width / 2) / max(width + 1, 1)
            y = t
            pos[s] = np.array([x, y])

    return pos

# Visual style for trajectory categories
LINE_STYLE = {
    'removed':       {'color': '#CC1111', 'lw': 1.8, 'ls': '--', 'alpha': 0.85},
    'newly_extinct': {'color': '#FF8C00', 'lw': 1.5, 'ls': '-',  'alpha': 0.85},
    'extinct':       {'color': '#AAAAAA', 'lw': 0.8, 'ls': '--', 'alpha': 0.30},
    'basal':         {'color': '#1a7a2e', 'lw': 1.6, 'ls': '-',  'alpha': 0.85},
    'consumer':      {'color': '#1a3a9e', 'lw': 1.6, 'ls': '-',  'alpha': 0.85},
}


def _save(fig, path):
    if path:
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"  Saved → {path}")


def _classify(B, basal, target=None, base_alive=None):
    """Return one of {removed, extinct, basal, consumer} per species at end of run."""
    end_alive = alive_at_end(B)
    out = []
    for i, ok in enumerate(end_alive):
        if i == target:
            out.append('removed')
        elif (not ok) or (base_alive is not None and not base_alive[i]):
            out.append('extinct')
        elif basal[i]:
            out.append('basal')
        else:
            out.append('consumer')
    return out


def plot_baseline(t, B, species, basal, path=None):
    """Time-series of biomass, coloured by basal / consumer / extinct."""
    cat = _classify(B, basal)
    fig, ax = plt.subplots(figsize=(14, 6))

    for i in range(len(species)):
        ax.plot(t, B[i], **LINE_STYLE[cat[i]])

    legend_items = []
    label_map = {
        'basal':    'Basal surviving',
        'consumer': 'Consumer surviving',
        'extinct':  f'Extinct < {EXTINCTION_THRESHOLD}',
    }
    for k, label in label_map.items():
        if k in cat:
            s = LINE_STYLE[k]
            legend_items.append(Line2D(
                [0], [0],
                color=s['color'], lw=2, ls=s['ls'], alpha=s['alpha'],
                label=label,
            ))

    ax.legend(handles=legend_items, loc='upper right')
    ax.set(xlabel='Time', ylabel='Biomass', title='Baseline GLV dynamics')
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    _save(fig, path)
    plt.show()

    print(f"  Baseline: {cat.count('basal')} basal + "
          f"{cat.count('consumer')} consumers surviving, "
          f"{cat.count('extinct')} extinct")


def plot_impact(df, path=None):
    """Horizontal bar chart of secondary extinctions, faceted by trophic tier."""
    df = df[df['Secondary extinctions'] > 0]
    if df.empty:
        print("  No secondary extinctions — skipping impact plot.")
        return

    tiers = sorted(df.Tier.unique())
    fig, axes = plt.subplots(1, len(tiers),
                             figsize=(max(5 * len(tiers), 8), 6))
    axes = np.atleast_1d(axes)
    global_max = df['Secondary extinctions'].max()

    for ax, t in zip(axes, tiers):
        sub = df[df.Tier == t].sort_values('Secondary extinctions')
        colors = ['#1a7a2e' if x == 'Basal' else '#1a3a9e' for x in sub.Type]
        bars = ax.barh(sub.Species, sub['Secondary extinctions'],
                       color=colors, alpha=0.85)
        ax.bar_label(bars, fontsize=8)
        ax.set_title(f'Tier {t}')
        ax.set_xlabel('Secondary extinctions')
        ax.set_xlim(right=global_max * 1.2)

    plt.tight_layout()
    _save(fig, path)
    plt.show()


def plot_removal_tier_graph(G, species, tier, basal, base_alive, B_removed,
                            target_name, path=None):
    """Tiered network after one removal.

    Tier 0 is shown at the bottom.
    Colours indicate removed, newly extinct, already extinct, basal, and consumer species.
    """
    target = species.index(target_name)
    end_alive = alive_at_end(B_removed)

    newly_extinct = base_alive & ~end_alive
    newly_extinct[target] = False

    cat = []
    for i in range(len(species)):
        if i == target:
            cat.append('removed')
        elif newly_extinct[i]:
            cat.append('newly_extinct')
        elif not base_alive[i]:
            cat.append('extinct')
        elif basal[i]:
            cat.append('basal')
        else:
            cat.append('consumer')

    pos = hierarchical_pos(species, tier)
    tiers = sorted(set(tier.values()))

    fig, ax = plt.subplots(figsize=(12, 10))

    # Tier lines
    for t in tiers:
        ax.axhline(y=t, color='gray', linestyle='--', alpha=0.2)
        ax.text(
            -0.65,
            t,
            f"Tier {t}",
            va='center',
            fontsize=10,
            color='gray',
        )

    # Edges
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=0.5,
        alpha=0.3,
        arrows=True,
        connectionstyle="arc3,rad=0.1",
        edge_color="#555",
    )

    # Nodes
    for i, s in enumerate(species):
        style = LINE_STYLE[cat[i]]

        ax.scatter(
            pos[s][0],
            pos[s][1],
            s=(G.degree(s) + 1) * 45,
            color=style['color'],
            alpha=style['alpha'],
            edgecolors='white',
            linewidths=0.5,
            zorder=3,
        )

        ax.text(
            pos[s][0],
            pos[s][1] - 0.08,
            s,
            ha='center',
            va='top',
            fontsize=6.5,
            alpha=max(style['alpha'], 0.35),
        )

    legend_items = []
    label_map = {
        'removed': f'Removed: {target_name}',
        'newly_extinct': f'Newly extinct ({int(newly_extinct.sum())})',
        'basal': 'Basal surviving',
        'consumer': 'Consumer surviving',
        'extinct': 'Already extinct in baseline',
    }

    for k, label in label_map.items():
        if k in cat:
            s = LINE_STYLE[k]
            legend_items.append(Line2D(
                [0], [0],
                marker='o',
                color='none',
                markerfacecolor=s['color'],
                markeredgecolor='white',
                markersize=9,
                alpha=s['alpha'],
                label=label,
            ))

    ax.legend(handles=legend_items, loc='upper right', fontsize=9)

    ax.set_title(f'Tiered cascade graph after removing {target_name}')
    ax.set_yticks(tiers)
    ax.set_yticklabels([f'Tier {t}' for t in tiers])
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)

    ax.set_ylim(min(tiers) - 0.5, max(tiers) + 0.5)
    ax.axis('off')

    plt.tight_layout()
    _save(fig, path)
    plt.show()

def plot_top_removal_impacts(species, A, basal, p, base_alive, ranking,
                             top_n=3, path_prefix=None,
                             G=None, tier=None, graph_prefix=None):
    """Trajectory plots and matching tiered cascade graphs for top-N removals."""
    impactful = ranking[ranking['Secondary extinctions'] > 0].head(top_n)
    if impactful.empty:
        print("  No removals caused secondary extinctions — skipping trajectory plots.")
        return

    z_priority = {
        'extinct': 0,
        'basal': 1,
        'consumer': 1,
        'newly_extinct': 2,
        'removed': 3,
    }

    for _, row in impactful.iterrows():
        target_name = row['Species']
        i = species.index(target_name)
        n_sec = int(row['Secondary extinctions'])

        t, B = simulate(A, basal, p, forced_extinct={i})

        end_alive = alive_at_end(B)
        newly_extinct = base_alive & ~end_alive
        newly_extinct[i] = False

        cat = []
        for j in range(len(species)):
            if j == i:
                cat.append('removed')
            elif newly_extinct[j]:
                cat.append('newly_extinct')
            elif not base_alive[j]:
                cat.append('extinct')
            elif basal[j]:
                cat.append('basal')
            else:
                cat.append('consumer')

        order = sorted(range(len(species)), key=lambda j: z_priority[cat[j]])

        fig, ax = plt.subplots(figsize=(14, 6))
        for j in order:
            ax.plot(t, B[j], **LINE_STYLE[cat[j]])

        legend_items = []
        label_map = {
            'basal': 'Basal surviving',
            'consumer': 'Consumer surviving',
            'extinct': 'Already extinct in baseline',
            'newly_extinct': f'Newly extinct after removal ({n_sec})',
            'removed': f'Removed: {target_name}',
        }

        for k, label in label_map.items():
            if k in cat:
                s = LINE_STYLE[k]
                legend_items.append(Line2D(
                    [0], [0],
                    color=s['color'],
                    lw=2,
                    ls=s['ls'],
                    alpha=s['alpha'],
                    label=label,
                ))

        ax.legend(handles=legend_items, loc='upper right', fontsize=9)
        ax.set(
            xlabel='Time',
            ylabel='Biomass',
            title=f'Removal of {target_name}: {n_sec} secondary extinction(s)',
        )
        ax.set_ylim(bottom=0)
        plt.tight_layout()

        if path_prefix is not None:
            safe_name = target_name.replace(' ', '_').replace('/', '_')
            _save(fig, f'{path_prefix}_{safe_name}.png')
        plt.show()

        if G is not None and tier is not None and graph_prefix is not None:
            safe_name = target_name.replace(' ', '_').replace('/', '_')
            plot_removal_tier_graph(
                G,
                species,
                tier,
                basal,
                base_alive,
                B,
                target_name,
                path=f'{graph_prefix}_{safe_name}.png',
            )


def tier_network(G, species, tier, basal, base_alive, path=None):
    """Static network drawing arranged by trophic tier.

    Tier 0 is shown at the bottom.
    Colours still indicate basal / consumer / already extinct status.
    """
    pos = hierarchical_pos(species, tier)
    tiers = sorted(set(tier.values()))
    idx = {s: i for i, s in enumerate(species)}

    fig, ax = plt.subplots(figsize=(12, 10))

    # Tier lines
    for t in tiers:
        ax.axhline(y=t, color='gray', linestyle='--', alpha=0.2)
        ax.text(
            -0.65,
            t,
            f"Tier {t}",
            va='center',
            fontsize=10,
            color='gray',
        )

    # Edges
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=0.5,
        alpha=0.3,
        arrows=True,
        connectionstyle="arc3,rad=0.1",
        edge_color="#555",
    )

    # Nodes
    for s in species:
        i = idx[s]

        if not base_alive[i]:
            color = LINE_STYLE['extinct']['color']
            alpha = LINE_STYLE['extinct']['alpha']
        elif basal[i]:
            color = LINE_STYLE['basal']['color']
            alpha = LINE_STYLE['basal']['alpha']
        else:
            color = LINE_STYLE['consumer']['color']
            alpha = LINE_STYLE['consumer']['alpha']

        ax.scatter(
            pos[s][0],
            pos[s][1],
            s=(G.degree(s) + 1) * 45,
            color=color,
            alpha=alpha,
            edgecolors='white',
            linewidths=0.5,
            zorder=3,
        )

        ax.text(
            pos[s][0],
            pos[s][1] - 0.08,
            s,
            ha='center',
            va='top',
            fontsize=6.5,
            alpha=max(alpha, 0.35),
        )

    legend_items = [
        Line2D(
            [0], [0],
            marker='o',
            color='none',
            markerfacecolor=LINE_STYLE['basal']['color'],
            markeredgecolor='white',
            markersize=9,
            alpha=LINE_STYLE['basal']['alpha'],
            label='Basal surviving',
        ),
        Line2D(
            [0], [0],
            marker='o',
            color='none',
            markerfacecolor=LINE_STYLE['consumer']['color'],
            markeredgecolor='white',
            markersize=9,
            alpha=LINE_STYLE['consumer']['alpha'],
            label='Consumer surviving',
        ),
        Line2D(
            [0], [0],
            marker='o',
            color='none',
            markerfacecolor=LINE_STYLE['extinct']['color'],
            markeredgecolor='white',
            markersize=9,
            alpha=LINE_STYLE['extinct']['alpha'],
            label='Already extinct in baseline',
        ),
    ]

    ax.legend(handles=legend_items, loc='upper right', fontsize=9)

    ax.set_title('Food web by trophic tier')
    ax.set_yticks(tiers)
    ax.set_yticklabels([f'Tier {t}' for t in tiers])
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)

    ax.set_ylim(min(tiers) - 0.5, max(tiers) + 0.5)
    ax.axis('off')

    plt.tight_layout()
    _save(fig, path)
    plt.show()

# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- Load and structure --------------------------------------------------
    G, species, W = load_food_web(CSV_PATH, SPECIES_PATH)
    basal = basal_mask(G, species)
    tier  = trophic_tiers(G, species, basal)
    top   = top_mask(G, species, basal)

    print(f"Loaded {len(species)} species, {G.number_of_edges()} interactions")
    print(f"Basal: {basal.sum()} | Consumers: {(~basal).sum()} | "
          f"Top predators: {[s for s, b in zip(species, top) if b]}")
    for t in sorted(set(tier.values())):
        members = [s for s in species if tier[s] == t]
        print(f"Tier {t}: {members}")

    # --- Parameter selection -------------------------------------------------
    if RUN_PARAM_SWEEP:
        best = param_sweep(species, W, basal, tier)
        scale = best['attack_scaling']
        d, e, h, c = best['d'], best['e'], best['h'], best['c']
    else:
        scale = ATTACK_SCALING
        d, e = D_DEATH, E_EFFICIENCY
        h, c = H_HANDLING, CONSUMER_COMPETITION

    A = attack_matrix(W, scale)
    p = make_parameters(species, basal, tier, d, e, h, c)
    print(f"Using: attack={scale}, d={d}, e={e}, h={h}, c={c}, "
          f"threshold={EXTINCTION_THRESHOLD}")

    # --- Baseline ------------------------------------------------------------
    print("Running baseline …")
    t, B = simulate(A, basal, p)
    base_alive = alive_at_end(B)
    print(f"Survivors: {base_alive.sum()}/{len(species)} | "
          f"Top surviving: {(base_alive & top).sum()}/{top.sum()}")

    plot_baseline(t, B, species, basal, f'{PLOT_DIR}/baseline.png')
    tier_network(G, species, tier, basal, base_alive,
                 f'{PLOT_DIR}/tier_network_baseline.png')

    # --- Removal sweep -------------------------------------------------------
    if REMOVAL_MODE == 'basal':
        candidates = [s for s, b in zip(species, basal) if b]
    elif REMOVAL_MODE == 'consumers':
        candidates = [s for s, b in zip(species, basal) if not b]
    else:
        candidates = list(species)
    candidates = [s for s in candidates if base_alive[species.index(s)]]

    print(f"Running {REMOVAL_MODE} removal sweep ({len(candidates)} species) …")
    ranking = removal_sweep(species, candidates, A, basal, p, base_alive, tier)
    ranking.to_csv(f'impact_ranking_{REMOVAL_MODE}.csv', index=False)
    print_table(ranking)
    plot_impact(ranking, f'{PLOT_DIR}/impact_by_tier_{REMOVAL_MODE}.png')

    # --- Trajectory and tiered graph plots for the most impactful removals ---
    print("Plotting trajectories and tiered graphs for the top-3 most-impactful removals …")
    plot_top_removal_impacts(
        species, A, basal, p, base_alive, ranking,
        top_n=3,
        path_prefix=f'{PLOT_DIR}/removal_impact_{REMOVAL_MODE}',
        G=G,
        tier=tier,
        graph_prefix=f'{PLOT_DIR}/removal_graph_{REMOVAL_MODE}',
    )


if __name__ == '__main__':
    main()
