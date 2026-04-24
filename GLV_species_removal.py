# glv_FW006_full_fixed_with_plots.py

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.integrate import solve_ivp

# =============================================================================
# PARAMETERS (TUNED BALANCED VERSION)
# =============================================================================
R_BASAL = 1.2
K_CARRYING = 1.0
D_DEATH = 0.03
E_EFFICIENCY = 0.3
H_HANDLING = 0.6
ATTACK_SCALING = 0.15

NORMALISE = False

T_END = 500
N_TIMEPOINTS = 2000
EXTINCTION_THRESHOLD = 1e-6

CSV_PATH = 'data/FW_006_NAMED.csv'
SPECIES_PATH = 'data/FW_006-species.csv'
REMOVAL_TARGET = 'Wigeongrass'

# =============================================================================
# LOAD
# =============================================================================
def load_food_web(csv_path, species_csv_path):
    diet_df = pd.read_csv(csv_path, index_col=0)
    attr_df = pd.read_csv(species_csv_path)

    prey_names = diet_df.index.tolist()
    pred_names = diet_df.columns.tolist()

    species = list(dict.fromkeys(prey_names + pred_names))
    idx = {sp: i for i, sp in enumerate(species)}

    W = np.zeros((len(species), len(species)))

    for r, prey in enumerate(prey_names):
        for c, pred in enumerate(pred_names):
            val = diet_df.iloc[r, c]
            if val > 0:
                W[idx[prey], idx[pred]] = float(val)

    G = nx.DiGraph()
    G.add_nodes_from(species)

    kingdom_map = dict(zip(attr_df['Specie'], attr_df['Kingdom']))
    for sp in species:
        G.nodes[sp]['Kingdom'] = kingdom_map.get(sp, 'Unknown')

    for i in range(len(species)):
        for j in range(len(species)):
            if W[i, j] > 0:
                G.add_edge(species[i], species[j], weight=W[i, j])

    return G, species, W, kingdom_map

# =============================================================================
# BUILD
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
# MODEL
# =============================================================================
def make_rhs(A, is_basal, params, forced_extinct=None):

    r, K, d, e, h = params['r'], params['K'], params['d'], params['e'], params['h']
    thresh = params['extinction_threshold']
    forced = set() if forced_extinct is None else forced_extinct

    def rhs(t, B):
        B = B.copy()
        B[B < thresh] = 0.0

        for fi in forced:
            B[fi] = 0.0

        dB = np.zeros_like(B)

        denom = 1.0 + h * (A * B[:, None]).sum(axis=0)

        for i in range(len(B)):
            if B[i] == 0:
                continue

            if is_basal[i]:
                dB[i] += r[i] * B[i] * (1 - B[i] / K[i])
            else:
                dB[i] -= d[i] * B[i]

                for j in range(len(B)):
                    if A[j, i] > 0 and B[j] > 0:
                        feeding = A[j, i] * B[i] * B[j] / denom[i]
                        dB[i] += e * feeding

            for k in range(len(B)):
                if A[i, k] > 0 and B[k] > 0:
                    feeding = A[i, k] * B[k] * B[i] / denom[k]
                    dB[i] -= feeding

        return dB

    return rhs

def run_simulation(A, is_basal, params, forced_extinct=None):

    B0 = params['B0'].copy()
    if forced_extinct:
        for fi in forced_extinct:
            B0[fi] = 0.0

    sol = solve_ivp(
        make_rhs(A, is_basal, params, forced_extinct),
        (0, T_END),
        B0,
        t_eval=np.linspace(0, T_END, N_TIMEPOINTS)
    )

    B = sol.y
    B[B < EXTINCTION_THRESHOLD] = 0
    return sol.t, B

def who_survived(B, params):
    return B[:, -1] > params['extinction_threshold']

# =============================================================================
# REMOVAL (FIXED)
# =============================================================================
def remove_species(species, target, A, is_basal, params, baseline_alive):

    idx = species.index(target)
    t, B = run_simulation(A, is_basal, params, {idx})

    alive_end = who_survived(B, params)
    secondary = baseline_alive & ~alive_end
    secondary[idx] = False

    return t, B, int(np.sum(secondary))

# =============================================================================
# PLOTS (RESTORED)
# =============================================================================
def plot_baseline(t, B, species, params):
    alive = who_survived(B, params)

    plt.figure(figsize=(12, 5))
    for i, sp in enumerate(species):
        if alive[i]:
            plt.plot(t, B[i], lw=1.5)
        else:
            plt.plot(t, B[i], '--', alpha=0.3)

    plt.title("Baseline dynamics")
    plt.xlabel("Time")
    plt.ylabel("Biomass")
    plt.tight_layout()
    plt.show()

def plot_after_removal(t, B, species, params, target):
    alive = who_survived(B, params)

    plt.figure(figsize=(12, 5))
    for i, sp in enumerate(species):
        if sp == target:
            continue
        if alive[i]:
            plt.plot(t, B[i], lw=1.5)
        else:
            plt.plot(t, B[i], '--', alpha=0.3)

    plt.title(f"After removing {target}")
    plt.tight_layout()
    plt.show()

def plot_network(G, species, B, params, target):
    alive = who_survived(B, params)
    idx = {sp: i for i, sp in enumerate(species)}

    colors = []
    for sp in G.nodes():
        if sp == target:
            colors.append('red')
        elif not alive[idx[sp]]:
            colors.append('orange')
        else:
            colors.append('green')

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_color=colors, with_labels=True, node_size=300)
    plt.title("Network after removal")
    plt.show()

def plot_impact_ranking(df):
    top = df.head(10)

    plt.figure(figsize=(8, 5))
    plt.barh(top['Species removed'], top['Secondary extinctions'])
    plt.gca().invert_yaxis()
    plt.title("Impact ranking")
    plt.tight_layout()
    plt.show()

def rank_all_removals(species, A, is_basal, params, baseline_alive):
    """
    Remove each species one at a time and count true secondary extinctions
    relative to the baseline steady state.
    """
    results = []
    n = len(species)

    for sp in species:
        _, _, n_sec = remove_species(
            species, sp, A, is_basal, params, baseline_alive
        )

        results.append({
            'Species removed': sp,
            'Secondary extinctions': n_sec,
            'Total lost': n_sec + 1,
            'Fraction of web lost': round((n_sec + 1) / n, 3),
        })

    return (pd.DataFrame(results)
              .sort_values('Secondary extinctions', ascending=False)
              .reset_index(drop=True))

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    G, species, W, kingdom_map = load_food_web(CSV_PATH, SPECIES_PATH)
    is_basal = get_basal_species(G, species)

    A = build_attack_matrix(W)
    params = build_params(species, is_basal)

    print("\nRunning baseline…")
    t_base, B_base = run_simulation(A, is_basal, params)

    baseline_alive = who_survived(B_base, params)
    print("Baseline survivors:", baseline_alive.sum(), "/", len(species))

    plot_baseline(t_base, B_base, species, params)

    target = REMOVAL_TARGET

    print("\nRunning removal:", target)
    t_pert, B_pert, n_sec = remove_species(
        species, target, A, is_basal, params, baseline_alive
    )

    print("Secondary extinctions:", n_sec)

    plot_after_removal(t_pert, B_pert, species, params, target)
    plot_network(G, species, B_pert, params, target)

    print("\nRanking…")
    ranking = rank_all_removals(species, A, is_basal, params, baseline_alive)
    print(ranking.head(10))

    plot_impact_ranking(ranking)