"""
glv_FW006_simple.py
===================
Generalized Lotka-Volterra (GLV) model with Type II functional response
configured for food web FW_006:

    Laguna Alvarado, western Gulf of Mexico
    30 species/groups, 229 interactions
    Source: Cruz-Escalona et al. (2007), Estuarine, Coastal and Shelf Science

HOW TO USE
----------
1.  Put your two CSV files in the same folder as this script:
        FW_006_02.csv           ← the diet/interaction matrix
        FW_006_02-species.csv   ← the species list with Kingdom column

2.  Run the script:
        python glv_FW006_simple.py

3.  To remove a different species, change REMOVAL_TARGET near the bottom.

WHAT THE FILES MUST LOOK LIKE
------------------------------
Diet matrix (FW_006_02.csv):
    - First column  = prey names  (row labels)
    - First row     = predator names  (column headers)
    - Each cell     = diet fraction of that prey in that predator's diet
                      (values between 0 and 1, or raw counts — see NORMALISE)
    - 0 means "this predator does not eat this prey"
    - Canibalism (predator eating itself) is allowed; the matrix handles it.

Species list (FW_006_02-species.csv):
    - Must have at minimum two columns: 'Specie' and 'Kingdom'
    - Kingdom is used only for colouring network plots
    - Optional column 'BodyMass_kg' enables allometric scaling (see below)
"""

# ── Standard library imports ─────────────────────────────────────────────────
import numpy as np            # arrays and maths
import pandas as pd           # CSV reading and DataFrames
import networkx as nx         # network / graph operations
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.integrate import solve_ivp   # ODE solver


# =============================================================================
#  SECTION 1 — FILE PATHS  (edit these to match your filenames)
# =============================================================================

CSV_PATH     = 'data/FW_006_NAMED.csv'          # diet matrix
SPECIES_PATH = 'data/FW_006-species.csv'  # species attributes

# The species whose removal we want to study in Section 4
REMOVAL_TARGET = 'Snooks'   # ← replace None with e.g. 'Zooplankton' (a string)
#   If you leave it as None the script will pick the first species automatically.


# =============================================================================
#  SECTION 2 — KEY MODEL PARAMETERS  (plain-English guide below)
# =============================================================================

# ── Intrinsic growth rate of basal producers ─────────────────────────────────
#   Basal species (those with no prey in the network — e.g. phytoplankton,
#   detritus) grow on their own following logistic growth.  A value of 1.0 is
#   a convenient normalised rate; the actual time unit is arbitrary.
#   Larger r → faster growth.
R_BASAL = 2.0

# ── Carrying capacity ─────────────────────────────────────────────────────────
#   Maximum normalised biomass a basal species can reach on its own (without
#   being grazed).  Set to 1.0 = "one unit of biomass".  If you have field
#   estimates of standing stock, you can replace this.
K_CARRYING = 1.0

# ── Metabolic (natural) death rate for consumers ─────────────────────────────
#   In every time step a consumer loses this fraction of its biomass to
#   respiration / natural mortality regardless of predation.
#   0.1 = 10 % per time unit.  Higher d → consumers need more prey to persist.
D_DEATH = 0.05

# ── Trophic (assimilation) efficiency ─────────────────────────────────────────
#   Of all the biomass a consumer ingests, only this fraction is converted into
#   its own biomass.  The rest is lost as respiration or faeces.
#   0.1 (10 %) is the classic Lindeman (1942) "10 % rule".
#   Realistic range: 0.05 – 0.20.
E_EFFICIENCY = 0.2

# ── Handling time (Type II functional response) ───────────────────────────────
#   Controls predator saturation:
#     h = 0   → Type I  (linear, no saturation — predators eat as fast as
#                         prey are available, no matter how abundant)
#     h > 0   → Type II (predators get "full"; feeding rate levels off at
#                         high prey density — more realistic for most animals)
#   h = 0.1 means a predator spends 10 % of a time unit handling each prey
#   item; this limits how fast it can eat even when prey are very abundant.
H_HANDLING = 0.2

# ── Normalise the diet matrix? ────────────────────────────────────────────────
#   FW_006 cell values are "relative frequency of prey in diet" — they are
#   already diet fractions that sum to ≤ 1 per predator column.
#   Set NORMALISE = False because the data is already proportional.
#   Set NORMALISE = True only if your CSV contains raw counts or biomass flows
#   that need to be rescaled so each predator's diet sums to 1.
NORMALISE = False

# ── Simulation time ───────────────────────────────────────────────────────────
#   Run the ODE from t=0 to T_END.  500 time units is usually enough for the
#   system to reach a steady state (equilibrium).
T_END        = 500
N_TIMEPOINTS = 2000   # resolution of the output trajectory

# ── Extinction threshold ───────────────────────────────────────────────────────
#   If a species' biomass drops below this value it is treated as extinct and
#   set to 0.  This prevents the ODE solver from fighting tiny negative numbers.
EXTINCTION_THRESHOLD = 1e-6


# =============================================================================
#  SECTION 3 — LOAD DATA
# =============================================================================

def load_food_web(csv_path, species_csv_path):
    """
    Read the diet matrix and species list; return everything needed downstream.

    ASSUMPTION: The CSV row labels are prey species; column labels are
    predators.  An entry W[prey, predator] > 0 means the predator eats the
    prey.  This is the standard Web-of-Life / EcoPath layout.
    """
    diet_df  = pd.read_csv(csv_path, index_col=0)
    attr_df  = pd.read_csv(species_csv_path)

    # Collect all unique species names, preserving order
    prey_names = diet_df.index.tolist()
    pred_names = diet_df.columns.tolist()
    species    = list(dict.fromkeys(prey_names + pred_names))
    n          = len(species)
    idx        = {sp: i for i, sp in enumerate(species)}

    # Build the weight matrix W  (n × n)
    # W[i, j] = interaction weight from species[i] (prey) to species[j] (predator)
    W = np.zeros((n, n))
    for r, prey in enumerate(prey_names):
        for c, pred in enumerate(pred_names):
            val = diet_df.iloc[r, c]
            if val > 0:
                W[idx[prey], idx[pred]] = float(val)

    # Build a directed graph (prey → predator edges)
    G = nx.DiGraph()
    G.add_nodes_from(species)
    kingdom_map = dict(zip(attr_df['Specie'], attr_df['Kingdom']))
    for sp in species:
        G.nodes[sp]['Kingdom'] = kingdom_map.get(sp, 'Unknown')
    for i, sp_i in enumerate(species):
        for j, sp_j in enumerate(species):
            if W[i, j] > 0:
                G.add_edge(sp_i, sp_j, weight=W[i, j])

    return G, species, W, kingdom_map, attr_df


# =============================================================================
#  SECTION 4 — BUILD PARAMETERS
# =============================================================================

def get_basal_species(G, species):
    """
    A basal species is one that has NO prey in the network (in-degree = 0).
    In Laguna Alvarado these would be primary producers (phytoplankton,
    macroalgae) and detritus.

    ASSUMPTION: if a species appears only as a prey item and never predates
    anything, it is treated as a basal producer with logistic growth.
    """
    return np.array([G.in_degree(sp) == 0 for sp in species], dtype=bool)


def build_attack_matrix(W, normalise):
    """
    Convert the raw weight matrix into 'attack rates' a[prey, predator].

    If NORMALISE=True each predator's column is divided by its sum so that
    diet fractions add to 1.  For FW_006 the values are already relative
    frequencies, so NORMALISE=False and we use them directly as attack rates.

    ASSUMPTION: attack rate is proportional to diet fraction.  A predator
    that gets 50 % of its diet from species X attacks species X twice as
    often as one for which X is only 25 % of the diet.
    """
    A = W.copy()
    if normalise:
        col_sums = A.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        A = A / col_sums[np.newaxis, :]
    return A


def build_params(species, G, is_basal):
    """
    Assemble all model parameters into one dictionary.

    Parameters set here (see Section 2 for plain-English descriptions):
        r  – intrinsic growth rates  (only >0 for basal species)
        K  – carrying capacities
        d  – metabolic death rates   (only >0 for consumers)
        e  – assimilation efficiency
        h  – handling time
        B0 – initial biomass of every species
    """
    n  = len(species)
    r  = np.where(is_basal, R_BASAL,      0.0)
    K  = np.full(n,          K_CARRYING)
    d  = np.where(is_basal,  0.0,         D_DEATH)
    B0 = np.where(is_basal,  1.0,         0.5)  # producers start at full K,
                                                  # consumers start at half

    return {
        'r' : r,
        'K' : K,
        'd' : d,
        'e' : E_EFFICIENCY,
        'h' : H_HANDLING,
        'B0': B0,
        'extinction_threshold': EXTINCTION_THRESHOLD,
    }


# =============================================================================
#  SECTION 5 — THE ODE (the actual model equations)
# =============================================================================

def make_rhs(A, is_basal, params, forced_extinct=None):
    """
    This function defines how fast each species' biomass changes at any moment.

    ── For a BASAL PRODUCER (e.g. phytoplankton) ────────────────────────────
        dB/dt = (logistic growth)  −  (biomass eaten by all predators)

        Logistic growth = r × B × (1 − B/K)
            The species grows fast when rare, slows as it approaches K,
            and would not grow at all if B = K (population at carrying capacity).

    ── For a CONSUMER (e.g. fish, shrimp) ───────────────────────────────────
        dB/dt = −(metabolic loss)
                + (energy gained from eating prey)
                − (biomass lost to its own predators)

        Metabolic loss    = d × B
        Energy gained     = e × Σ(feeding rates on each prey)
        Predation loss    = Σ(feeding rates by each predator on this species)

    ── FEEDING RATE (Type II functional response) ────────────────────────────
        Feeding of predator j on prey i:

                    a[i,j] × B_j × B_i
        F(i→j) = ─────────────────────────────────────
                  1  +  h × Σ_k( a[k,j] × B_k )

        Numerator:  proportional to both predator and prey abundance,
                    scaled by how much predator j likes prey i (attack rate).
        Denominator: the "1 + handling" term — when prey are abundant the
                    predator spends time handling them, which limits the
                    feeding rate.  Without this (h=0) feeding is unbounded.

    ASSUMPTION — shared handling time: all prey of a given predator share the
    same handling time parameter h.  A more complex model could assign
    different h values to each predator–prey pair.
    """
    r      = params['r']
    K      = params['K']
    d      = params['d']
    e      = params['e']
    h      = params['h']
    thresh = params['extinction_threshold']
    forced = set() if forced_extinct is None else forced_extinct

    def rhs(t, B):
        B = B.copy()
        B[B < thresh] = 0.0           # snap near-zero values to 0
        for fi in forced:
            B[fi] = 0.0               # keep removed species at 0

        dB    = np.zeros_like(B)
        n     = len(B)

        # Handling-time denominator for each predator j:
        # denom[j] = 1 + h × (sum of a[k,j]*B[k] for all prey k of j)
        denom = 1.0 + h * (A * B[:, np.newaxis]).sum(axis=0)

        for i in range(n):
            if B[i] == 0.0:
                continue

            if is_basal[i]:
                # Logistic growth for producers
                dB[i] += r[i] * B[i] * (1.0 - B[i] / K[i])
            else:
                # Metabolic loss for consumers
                dB[i] -= d[i] * B[i]
                # Assimilation gain from eating each prey j
                for j in range(n):
                    if A[j, i] > 0 and B[j] > 0:
                        feeding = A[j, i] * B[i] * B[j] / denom[i]
                        dB[i] += e * feeding

            # Predation loss to every predator k
            for k in range(n):
                if A[i, k] > 0 and B[k] > 0:
                    feeding = A[i, k] * B[k] * B[i] / denom[k]
                    dB[i] -= feeding

        return dB

    return rhs


# =============================================================================
#  SECTION 6 — RUN THE SIMULATION
# =============================================================================

def run_simulation(A, is_basal, params, forced_extinct=None):
    """
    Feed the equations into scipy's ODE solver (Runge-Kutta 4/5).
    Returns the time axis and a (n_species × n_timepoints) biomass array.
    """
    B0 = params['B0'].copy()
    if forced_extinct:
        for fi in forced_extinct:
            B0[fi] = 0.0

    rhs    = make_rhs(A, is_basal, params, forced_extinct)
    t_eval = np.linspace(0, T_END, N_TIMEPOINTS)

    sol = solve_ivp(
        rhs,
        (0, T_END),
        B0,
        method   = 'RK45',
        t_eval   = t_eval,
        rtol     = 1e-6,
        atol     = 1e-9,
    )

    B = sol.y
    B[B < EXTINCTION_THRESHOLD] = 0.0
    return sol.t, B


def who_survived(B, params):
    """Boolean array: True if species has biomass > threshold at end of run."""
    return B[:, -1] > params['extinction_threshold']


# =============================================================================
#  SECTION 7 — SINGLE SPECIES REMOVAL (perturbation experiment)
# =============================================================================

def remove_species(species, target, A, is_basal, params):
    """
    Set one species to 0 and re-run the simulation.
    Returns time, biomass trajectories, and count of secondary extinctions.

    Secondary extinction = a species that was alive at the start but is dead
    at the end, caused indirectly by the removal of the target.
    """
    idx = species.index(target)
    t, B = run_simulation(A, is_basal, params, forced_extinct={idx})

    alive_start = params['B0'] > params['extinction_threshold']
    alive_end   = who_survived(B, params)
    n_secondary = int(np.sum(alive_start & ~alive_end)) - 1  # minus the target
    return t, B, max(n_secondary, 0)


def print_removal_report(species, target, B, params):
    idx         = species.index(target)
    alive_start = params['B0'] > params['extinction_threshold']
    alive_end   = who_survived(B, params)

    lost = [sp for i, sp in enumerate(species)
            if alive_start[i] and not alive_end[i] and sp != target]

    print("\n" + "=" * 55)
    print(f"  REMOVAL EXPERIMENT: '{target}'")
    print(f"  Secondary extinctions ({len(lost)}): {lost if lost else 'none'}")
    print(f"  Total species lost : {len(lost) + 1} / {len(species)}")
    print(f"  Surviving species  : {len(species) - len(lost) - 1}")
    print("=" * 55 + "\n")


# =============================================================================
#  SECTION 8 — RANK ALL SPECIES BY IMPACT
# =============================================================================

def rank_all_removals(species, A, is_basal, params):
    """
    Remove each species one at a time, record how many others go extinct,
    and return a ranked table.  This answers: "which species is the most
    important for the network's stability?"
    """
    results = []
    n = len(species)
    for sp in species:
        _, _, n_sec = remove_species(species, sp, A, is_basal, params)
        results.append({
            'Species removed'      : sp,
            'Secondary extinctions': n_sec,
            'Total lost'           : n_sec + 1,
            'Fraction of web lost' : round((n_sec + 1) / n, 3),
        })
    return (pd.DataFrame(results)
              .sort_values('Secondary extinctions', ascending=False)
              .reset_index(drop=True))


# =============================================================================
#  SECTION 9 — PLOTS
# =============================================================================

def plot_baseline(t, B, species, params):
    """
    Shows how all 30 species' biomasses change over time with no disturbance.
    Solid lines = species that reach a positive steady state.
    Dashed grey  = species that go extinct even without disturbance.
    """
    alive_end = who_survived(B, params)
    fig, ax   = plt.subplots(figsize=(13, 5))
    cmap      = plt.cm.tab20

    for i, sp in enumerate(species):
        color = cmap(i % 20)
        if alive_end[i]:
            ax.plot(t, B[i], color=color, lw=1.5, label=sp)
        else:
            ax.plot(t, B[i], color='grey', lw=0.7, ls='--', alpha=0.4)

    ax.set_xlabel('Time (arbitrary units)')
    ax.set_ylabel('Biomass (normalised)')
    ax.set_title('Laguna Alvarado — baseline GLV dynamics (no perturbation)')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)
    plt.tight_layout()
    plt.savefig('baseline_trajectories.png', dpi=150)
    plt.show()
    print("Figure saved: baseline_trajectories.png")


def plot_after_removal(t, B, species, params, target):
    """
    Same trajectory plot but after removing the target species.
    The removed species is skipped entirely.
    """
    alive_end = who_survived(B, params)
    fig, ax   = plt.subplots(figsize=(13, 5))
    cmap      = plt.cm.tab20

    for i, sp in enumerate(species):
        if sp == target:
            continue
        color = cmap(i % 20)
        if alive_end[i]:
            ax.plot(t, B[i], color=color, lw=1.5, label=sp)
        else:
            ax.plot(t, B[i], color='grey', lw=0.7, ls='--', alpha=0.4)

    ax.set_xlabel('Time (arbitrary units)')
    ax.set_ylabel('Biomass (normalised)')
    ax.set_title(f'GLV dynamics after removing: {target}')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)
    plt.tight_layout()
    fname = f'after_removal_{target.replace(" ", "_")}.png'
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Figure saved: {fname}")


def plot_network(G, species, B, params, target, kingdom_map):
    """
    Network diagram coloured by outcome:
        Red    = removed species
        Orange = went extinct after the removal
        Teal   = survived
    Node size = number of connections (degree).
    """
    alive_end = who_survived(B, params)
    sp_idx    = {sp: i for i, sp in enumerate(species)}
    pos        = nx.spring_layout(G, k=0.6, seed=42)
    node_sizes = [(G.degree(nd) + 1) * 120 for nd in G.nodes()]

    colors = []
    for nd in G.nodes():
        i = sp_idx.get(nd, -1)
        if nd == target:
            colors.append('#e74c3c')
        elif i >= 0 and not alive_end[i]:
            colors.append('#e67e22')
        else:
            colors.append('#2ecc71')

    n_sec = sum(1 for nd in G.nodes()
                if nd != target and sp_idx.get(nd, -1) >= 0
                and not alive_end[sp_idx[nd]])

    plt.figure(figsize=(11, 8))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=colors, alpha=0.88)
    nx.draw_networkx_edges(G, pos, width=0.8, alpha=0.35,
                           arrows=True, arrowsize=12)
    nx.draw_networkx_labels(G, pos, font_size=7)

    legend = [
        Patch(color='#e74c3c', label=f'Removed: {target}'),
        Patch(color='#e67e22', label=f'Extinct after removal ({n_sec})'),
        Patch(color='#2ecc71', label=f'Surviving ({G.number_of_nodes() - n_sec - 1})'),
    ]
    plt.legend(handles=legend, loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f'Laguna Alvarado — network after removing {target}')
    plt.axis('off')
    plt.tight_layout()
    fname = f'network_{target.replace(" ", "_")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure saved: {fname}")


def plot_impact_ranking(ranking_df):
    """
    Horizontal bar chart: how many species go extinct when each species
    is removed.  The higher the bar, the more 'keystone' the species is.
    """
    top = ranking_df.head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top['Species removed'][::-1],
            top['Secondary extinctions'][::-1],
            color='#3498db', edgecolor='white', lw=0.5)
    ax.set_xlabel('Number of secondary extinctions')
    ax.set_title('Laguna Alvarado — most impactful species removals')
    plt.tight_layout()
    plt.savefig('impact_ranking.png', dpi=150)
    plt.show()
    print("Figure saved: impact_ranking.png")


# =============================================================================
#  MAIN — runs everything in order
# =============================================================================

if __name__ == '__main__':

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print("Loading food web data …")
    G, species, W, kingdom_map, attr_df = load_food_web(CSV_PATH, SPECIES_PATH)
    is_basal = get_basal_species(G, species)
    A        = build_attack_matrix(W, normalise=NORMALISE)
    params   = build_params(species, G, is_basal)

    print(f"\nNetwork loaded:")
    print(f"  Species           : {len(species)}")
    print(f"  Interactions      : {G.number_of_edges()}")
    print(f"  Basal producers   : {is_basal.sum()}")
    print(f"  Consumers         : {(~is_basal).sum()}")
    print(f"  Basal species     : {[sp for sp, b in zip(species, is_basal) if b]}")

    # ── Step 2: Baseline — no disturbance ────────────────────────────────────
    print("\nRunning baseline simulation …")
    t_base, B_base = run_simulation(A, is_basal, params)
    n_alive = who_survived(B_base, params).sum()
    print(f"  Survivors at steady state: {n_alive} / {len(species)}")
    print(f"  Surviving species: {', '.join([str(sp) for i, sp in enumerate(species) if who_survived(B_base, params)[i]])}")    
    plot_baseline(t_base, B_base, species, params)

    # ── Step 3: Remove one species ───────────────────────────────────────────
    target = REMOVAL_TARGET if REMOVAL_TARGET else species[0]
    print(f"\nRunning removal experiment: '{target}' …")
    t_pert, B_pert, n_sec = remove_species(species, target, A, is_basal, params)
    print_removal_report(species, target, B_pert, params)
    plot_after_removal(t_pert, B_pert, species, params, target)
    plot_network(G, species, B_pert, params, target, kingdom_map)

    # ── Step 4: Rank every species by how impactful its removal is ───────────
    print("\nRanking all species by removal impact (this may take a minute) …")
    ranking = rank_all_removals(species, A, is_basal, params)
    print("\nTop 10 most impactful removals:")
    print(ranking.head(10).to_string(index=False))
    ranking.to_csv('impact_ranking.csv', index=False)
    print("\nFull ranking saved to: impact_ranking.csv")
    plot_impact_ranking(ranking)