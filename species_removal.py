import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def build_graph(csv_path, species_csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    attributes_df = pd.read_csv(species_csv_path)

    G = nx.DiGraph()
    adj_matrix = df.values
    prey_nodes = df.index.tolist()
    predator_nodes = df.columns.tolist()

    rows, cols = np.where(adj_matrix == 1)
    edges = [(prey_nodes[r], predator_nodes[c]) for r, c in zip(rows, cols)]
    G.add_edges_from(edges)

    kingdom_map = dict(zip(attributes_df['Specie'], attributes_df['Kingdom']))
    for node in G.nodes():
        G.nodes[node]['Kingdom'] = kingdom_map.get(node, 'Unknown')

    return G, kingdom_map


def cascade_removal(G, removed_species):
    """
    Simple topological cascade model.
    Rule: if all prey of a predator go extinct, the predator goes extinct too.

    Parameters
    ----------
    G : nx.DiGraph
        Food web with edges prey -> predator.
    removed_species : str
        The species to remove as the primary extinction event.

    Returns
    -------
    extinct_set : set
        All extinct species including the removed one.
    extinction_order : list
        Sequence of extinctions in cascade order.
    """
    extinct = {removed_species}
    queue = [removed_species]
    extinction_order = [removed_species]

    while queue:
        just_removed = queue.pop(0)
        for predator in G.successors(just_removed):
            if predator in extinct:
                continue
            all_prey = list(G.predecessors(predator))
            if all(prey in extinct for prey in all_prey):
                extinct.add(predator)
                queue.append(predator)
                extinction_order.append(predator)

    return extinct, extinction_order


def print_cascade_summary(G, removed_species, extinct_set, extinction_order):
    secondary = [s for s in extinction_order if s != removed_species]
    print("=" * 50)
    print(f"  Primary extinction : {removed_species}")
    print(f"  Secondary extinctions ({len(secondary)}): {secondary if secondary else 'none'}")
    print(f"  Total lost : {len(extinct_set)} / {G.number_of_nodes()} species")
    print(f"  Surviving  : {G.number_of_nodes() - len(extinct_set)} species")
    print("=" * 50)


def plot_cascade(G, pos, node_sizes, removed_species, extinct_set):
    node_colors = []
    for node in G.nodes():
        if node == removed_species:
            node_colors.append('red')
        elif node in extinct_set:
            node_colors.append('orange')
        else:
            node_colors.append('steelblue')

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.85)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.4, arrows=True, arrowsize=15)
    nx.draw_networkx_labels(G, pos, font_size=8)

    legend = [
        Patch(color='red',       label=f'Removed: {removed_species}'),
        Patch(color='orange',    label=f'Cascaded extinct ({len(extinct_set) - 1})'),
        Patch(color='steelblue', label=f'Surviving ({G.number_of_nodes() - len(extinct_set)})'),
    ]
    plt.legend(handles=legend, loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f"Cascade after removing: {removed_species}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def run_all_removals(G):
    """
    Run cascade removal for every species and return a summary DataFrame
    sorted by number of secondary extinctions triggered.
    """
    results = []
    for species in G.nodes():
        extinct_set, _ = cascade_removal(G, species)
        results.append({
            'removed': species,
            'secondary_extinctions': len(extinct_set) - 1,
            'total_lost': len(extinct_set),
            'fraction_lost': len(extinct_set) / G.number_of_nodes(),
        })
    return pd.DataFrame(results).sort_values('secondary_extinctions', ascending=False).reset_index(drop=True)


if __name__ == '__main__':
    # --- Build graph from your CSV files ---
    G, kingdom_map = build_graph('data/FW_012_02.csv', 'data/FW_012_02-species.csv')
    pos = nx.spring_layout(G, k=0.5, seed=42)
    node_sizes = [(G.in_degree(node) + 1) * 500 for node in G.nodes()]

    # --- Single removal: pick any species by name ---
    target = list(G.nodes())[0]   # replace with e.g. 'Phytoplankton'
    extinct_set, extinction_order = cascade_removal(G, target)
    print_cascade_summary(G, target, extinct_set, extinction_order)
    plot_cascade(G, pos, node_sizes, target, extinct_set)

    # --- Sweep: most impactful species to remove ---
    print("\nTop 10 most impactful removals:")
    summary = run_all_removals(G)
    print(summary.head(10).to_string(index=False))