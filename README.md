# Copula-Based Normalizing Flows
This repository contains the code for reproducing the experiments in the paper:

> M. Laszkiewicz, J. Lederer, A. Fischer, _Copula-Based Normalizing Flows_, INNF+ 2021. </br>

## How to run the code
1. Plotting the training and test loss over 100 trails:
```
python 2d_estimation.py
```
This will generate Figure 1 based on pre-trained models. To retrain and then generate Figure 1, please set the flag ``--compute True``. 

2. Computing the Q-plots:
```
python quantile_functions.py
```
This reproduces Figures 2, 8, and 9.

3. Computing the Lipschitz surfaces: 
```
python lipschitz_surface.py --base_dist exactMaginals
```
This reproduces Figure 3 and 10. Possible options for the flag ``--base_dist`` are ``normal, heavierTails, correctFamily, exactMaginals``. 

4. Visualizing the Copulae:
```
jupyter lab visualizations_copula.ipynb
```
This opens the corresponding Jupyter notebook. 

## Access the data
All plots are saved in ``/plots``. 
``/utils`` contains some helper functions. 
We save the ``.csv`` file computed by ``2d_estimation.py`` in ``/data``. 