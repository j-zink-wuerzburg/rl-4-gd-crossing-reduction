# RL 4 Crossing Reduction

This repository provides the supplemental material for the paper "Using Reinforcement Learning to Optimize the Global and Local Crossing Number" by Timo Brand, Henry Förster, Stephen Kobourov, Daniel Kohrt, Robin Schukrafft, Markus Wallinger, and Johannes Zink, which appears at the 34th International Symposium on Graph Drawing and Network Visualization (GD 2026).

## Organization of the repository

> code-and-graphs

Contains the python requirements in *requirements.txt* and the python source code in the folder *src*. The configuration files are stored in *configs* and our best fully trained models for the global and local crossing are directly available in the folder *models*. The folder *graphs* contains the data sets with graphs for training and testing.

> evaluation

Within the *evaluation* directory, the folder *results* contains for each algorithm two csv files: one with results on the testing graphs for the Rome data set, one for the extended Barabási-Albert graphs data set. These are the results presented in the paper. The directory contains some more python scripts and csv files to summarize the data from the results folder.

