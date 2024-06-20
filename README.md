|**[Introduction](#introduction)**
|**[Dependencies](#dependencies)**
|**[Data](#data)**
|**[Models](#models)**
|**[Configuration](#configuration)**
|**[Evaulation](#evaluation)**
|**[Notebooks](#notebooks)**
|**[Citation](#citation)**

**Introduction**
In the drug discovery process, where experiments can be costly and time-consuming, computational models that predict drug-target interactions are valuable tools to accelerate the development of new therapeutic agents.
Estimating the uncertainty inherent in these neural network predictions provides valuable information that facilitates optimal decision-making when risk assessment is crucial.
However, such models can be poorly calibrated, which results in unreliable uncertainty estimates that do not reflect the true predictive uncertainty.
In this study, we compare different metrics, including accuracy and calibration scores, used for model hyperparameter tuning to investigate which model selection strategy achieves well-calibrated models.
Furthermore, we propose to use a computationally efficient Bayesian uncertainty estimation method named Bayesian Linear Probing (BLP), which generates Hamiltonian Monte Carlo (HMC) trajectories to obtain samples for the parameters of a Bayesian Logistic Regression fitted to the hidden layer of the baseline neural network.
We report that BLP improves model calibration and achieves the performance of common uncertainty quantification methods by combining the benefits of uncertainty estimation and probability calibration methods.
Finally, we show that combining post hoc calibration method with well-performing uncertainty quantification approaches can boost model accuracy and calibration.

**Dependencies**

Main requirements:
- Python >= 3.9.7
- Cuda >= 12.4
- Pytorch >= 2.0.1

**Logging** is supported with: wandb >= 0.15.10

Install full conda environment with

```bash
$ conda env create -f config/environment.yaml
```


**Data**
## ChEMBL Targets [[1]](#1):

- ChEMBL1951: Monoamine oxidase A
- ChEMBL340: Cytochrome P450 3A4
- ChEMBL240: hERG


Download Data into data/:

```bash
$ cd data/
#TODO: Add Datasets
```

**Models**
Following Models are supported:

- Baseline (MLP)
- Platt - scaled MLP (MLP + P)
- Ensemble Model (MLP-E)
- Platt - scaled Ensemble Model (MLP-E + P)
- MC dropout Model (MLP-D)
- Bayesian linear probing (MLP-BLP)
- Platt - scaled Bayesian linear probing (MLP-BLP + P)

**Configuration**
# Tuning Hyperparameters of MLP using wandb

Start Hyperparameter Sweep with

```bash
$ wandb sweep config/sweep_baseline.yaml 

```

Copy Sweep ID and start agent with 
```bash
$ wandb agent sweep_id 

```

Other setting options (e.g. TargetID, Number of Model Repetitions) can be chosen in sweep_baseline.yaml. 
In this study, hyperparameters were tuned optimizing 4 different metrics:

- Accuracy (acc)
- AUC (rocauc)
- BCE loss (bce)
- Adaptive Calibration Error [[2]](#2) (ace)

**Evaluation**

# Baseline Model (MLP)/ Ensemble Model (MLP-E) 

Train baseline MLP with 

```bash
$ python train_baseline.py \
--targetid 240 \
--hidden_sizes 4 \
--dropout 0.6 \
--weight_decay 0.0001 \
--learning_rate 0.01 \
--hp_metric acc \
```
Choose Hyperparameters for hidden_size, dropout, weight_decay and learning_rate according to reults in hyperparameter sweep. Specify metric that was chosen to optimize Hyperparameters (```--hp-metrics```).

Number of model repetitions can be specified with ```--nr_models``` and number of base estimators used for generating ensemble models can be specified with ```--nr_ensemble_estimators```. If ```--nr_ensemble_estimators```> 1, ensemble models are generated in addition.

# MC Dropout (MLP-D)
Generate MC dropout models from baseline models with

```bash
$ python train_mc.py \
--targetid 240 \
--hidden_sizes 4 \
--dropout 0.6 \
--weight_decay 0.0001 \
--learning_rate 0.01 \
--hp_metric acc \
```

Number of forward passes can be specified with ```--nr_mc_estimators```.

# Bayesian Linear Probing (MLP-BLP) [[3]](#3) [[4]](#4) 
# TODO!!!!!!!!!!!!!!!!!

# Platt Scaling (MLP + P, MLP-E + P, MLP-BLP + P)

Generate platt-scaled predictions with:
```bash
$ python train_platt.py \
--targetid 240 \
--hp_metric acc \
```

To apply Platt-Scaling to MLP-E (or to MLP-BLP), set ```--from_ensemble``` (or ```--from_BLP```) to True.


**Notebooks**

Results are visualized in CalibrationStudy.ipynb.

**Citation**

Cite this repository: Awaiting Review...

**References**

<a id="1">[1]</a>  Mendez, D., Gaulton, A., Bento, A.P., Chambers, J., De Veij, M., F´elix, E., Magari˜nos, M., Mosquera, J., Mutowo, P., Nowotka, M., Gordillo-Mara˜n´on, M.,
Hunter, F., Junco, L., Mugumbate, G., Rodriguez-Lopez, M., Atkinson, F., Bosc, N., Radoux, C., Segura-Cabrera, A., Hersey, A., Leach, A.: ChEMBL: towards
direct deposition of bioassay data. Nucleic Acids Research 47(D1), 930–940 (2018) https://doi.org/10.1093/nar/gky1075

<a id="2">[2]</a>   Nixon, J., Dusenberry, M.W., Zhang, L., Jerfel, G., Tran, D.: Measuring calibration in deep learning. In: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR) Workshops (2019)

<a id="3">[3]</a>   Cobb, A.D., Baydin, A.G., Jalaian, B.: hamiltorch. GitHub (2023). https://github.com/AdamCobb/hamiltorch

<a id="4">[4]</a>   Cobb, A.D., Jalaian, B.: Scaling hamiltonian monte carlo inference for bayesian neural networks with symmetric splitting. In: Uncertainty in Artificial  Intelligence, pp. 675–685 (2021). PMLR







