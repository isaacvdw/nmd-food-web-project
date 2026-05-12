# nmd-food-web-project

file structure
|- /data
    |- FW_012_02-species.csv (contains the species names and classifications for North Carolina)
    |- FW_012_02.csv (contains the adjacency matrix of the predators and prey for North         Carolina)
    |- FW_008.csv (contains the species names and classifications for Carribean)
    |- FW_008-species.csv (contains the adjacency matrix of the predators and prey for Carribean)
    |- FW_003_NAMED.csv (contains the species names and classifications for Angola)
    |- FW_003-species.csv (contains the adjacency matrix of the predators and prey for Angola)
    |- FW_006.csv (contains the species names and classifications for gulf of Mexico)
|- /cascade_model_experiments (simple cascade model)
    |- full_extinct_cascade.py (simulates the removal of all species randomly)
    |- simple_cascade_removal.py (simulates removal of random x number of species with x as input)
    |- simple_cascade_specific.py (simulates removal of species from specific tiers as input)
|- /null_models
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
|- food_web.ipynb (analysis using normal directed graph)
|- food_web_tiered.ipynb (Analysis using tiered graphs)
|- food_web_tiered - Copy.ipynb (Analysis using tiered graphs + 3D visualisation)
|- tiered_species_removal.py (species removal based on GLV and also visualisation using 
   tiered network)
|- GLV_species_removal.oy (simulates GLV species removal)

dependencies needed:
- pandas
- networkx
- matplotlib
- numpy
- seaborn
