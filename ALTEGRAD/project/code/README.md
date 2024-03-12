# ALTeGraD-2023 Data Challenge

## Kaggle Team: CAC

Borachhun YOU [borachhun.you@ens-paris-saclay.fr](borachhun.you@ens-paris-saclay.fr)
Haocheng LIU [haocheng.liu@polytechnique.edu](haocheng.liu@polytechnique.edu)
Ly An CHHAY [ly-an.chhay@polytechnique.edu](ly-an.chhay@polytechnique.edu)

## Overview

This folder contains the necessary Jupyter notebooks to train and infer our model. We have extended the baseline model by incorporating a Graph Attention Network (GAT) module and introducing residual connections. You can find detailed comments within the notebook for a deeper understanding of the implementation. The model's performance can be further enhanced by substituting in more sophisticated language models such as SciBert.

## Structure

- `DistilBERT_or_SciBert_GAT10_50ep.ipynb`: This Jupyter notebook contains the code for training and inference of our model. We build upon a baseline architecture and enhance it with a GAT module, alongside residual connections to improve learning efficacy. For clarity and further information on the model's architecture and functionality, please refer to the inline comments throughout the notebook. The performence is further improved by replacing the language model with [SciBert](https://github.com/allenai/scibert).
- `Further_Study/`: The directory named `Further_Study` includes our exploratory attempts. Here, we adopt an adversarial architecture akin to GANs, utilizing a Critic to estimate the Wasserstein distance between encoded embeddings. This is aimed at bolstering the model's performance and facilitating more efficient learning. Though it dose not outperform a SciBert-based model as above, we consider it an interesting attempt and believe it to be a promising method for the future.
- `EDA.ipynb`: This Jupyter notebook is dedicated to the Exploratory Data Analysis (EDA) of molecular graph structures. It showcases our approach to visualizing molecular graphs and computing statistical features like degree distribution, clustering coefficient, and distribution of shortest paths.