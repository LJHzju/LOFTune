# LOFTune

This repository contains the source code for our paper: **LOFTune: A Low-overhead and Flexible Approach for Spark SQL Configuration Tuning**.

# Requirements
***
- tokenizers 0.11.4
- optuna 3.5.0
- quantile-forest 1.1.3
- scikit-learn 1.0.2
- torch 1.12.1
- tree-sitter 0.20.1
- sqlglot 20.7.1
***

# Datasets
***
- [TPCDS(100G and 300G)](https://www.tpc.org/tpcds/)
- [TPCH(100G)](https://www.tpc.org/tpch/)
- [IMDB](http://homepages.cwi.nl/~boncz/job/imdb.tgz)

# Structure
***
- config: The parameters of the algorithm and model.
- data: Part of datasets used in the experiments.
- modules: Knowledge Base Updater, Configuration Recommender, Controller and some helper functions.
- sql_encoder: Convert sql to vector, i.e. Multi-task SQL Representation Learning.
- main.py: A complete function entrance, including all callable related interfaces.
- run_tests.sh: A shell test script that can be run directly.
- scripts and utils.py: Some commonly used helper functions.
***

# Usage
***
1. Download datasets
2. Set mode and workloads in run_tests.sh
3. Execute run_tests.sh
