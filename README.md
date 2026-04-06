# nmd-food-web-project

file structure
|- /data
    |- FW_012_02-species.csv (contains the species names and classifications for North Carolina)
    |- FW_012_02.csv (contains the adjacency matrix of the predators and prey for North         Carolina)
    |- FW_008.csv (contains the species names and classifications for Carribean)
    |- FW_008-species.csv (contains the adjacency matrix of the predators and prey for Carribean)
|- food_web.ipynb (analysis using normal directed graph)
|- food_web_tiered.ipynb (Analysis using tiered graphs) 

dependencies needed:
- pandas
- networkx
- matplotlib
- numpy
- seaborn


TODO:
- include analysis on the weights of the graph