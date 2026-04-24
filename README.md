# nmd-food-web-project

file structure
|- /data
    |- FW_012_02-species.csv (contains the species names and classifications for North Carolina)
    |- FW_012_02.csv (contains the adjacency matrix of the predators and prey for North         Carolina)
    |- FW_008.csv (contains the species names and classifications for Carribean)
    |- FW_008-species.csv (contains the adjacency matrix of the predators and prey for Carribean)
    |- FW_003_NAMED.csv (contains the species names and classifications for Angola)
    |- FW_003-species.csv (contains the adjacency matrix of the predators and prey for Angola)
|- food_web.ipynb (analysis using normal directed graph)
|- food_web_tiered.ipynb (Analysis using tiered graphs)
|- food_web_tiered - Copy.ipynb (Analysis using tiered graphs + 3D visualisation)
|- research.py 
    (null model research comparing small number of parameters: "connectance",
        "mean_predator_degree", "mean_prey_degree", "predator_degree_var", "prey_degree_var")
|- research_2.py --- most updated for the null model research
    (null model research comparing the parameters: 
        nestedness_nodf: nestedness (0–100)
        h2_specialisation: H2' (0 = no specialisation, 1 = perfect)
        predator_degree_var: variance of predator (column) degrees
        prey_degree_var: variance of prey (row) degrees
        c_score: Stone & Roberts C-score
        robustness_r50: prey fraction remaining when 50 % predators lost
        niche_overlap_mean: mean Pianka overlap across predator pairs
    )
|- tiered_species_removal.py (species removal based on GLV and also visualisation using 
   tiered network)

dependencies needed:
- pandas
- networkx
- matplotlib
- numpy
- seaborn


FoodWebs chosen for comparison:
- FW008
- FW003

TODO:
- fix the logic that all species are stable (no extinctions in the initial foodweb)
- do the analysis for the null model research
- make it clearer as to what tiers (basal and apex predators) are affected after a species is removed
- can change color scheme to species extinct because of removal and species that have already been extinct