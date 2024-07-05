|**[Dependencies](#dependencies)**
|**[Data](#data)**
|**[Supported Models](#supported-models)**
|**[Configuration](#configuration)**
|**[Model Generation](#model-generation)**
|**[Model Evaluation](#model-evaluation)**
|**[Citation](#citation)**

# **Introduction**
In the drug discovery process, where experiments can be costly and time-consuming, computational models that predict drug-target interactions are valuable tools to accelerate the development of new therapeutic agents.
Estimating the uncertainty inherent in these neural network predictions provides valuable information that facilitates optimal decision-making when risk assessment is crucial.
However, such models can be poorly calibrated, which results in unreliable uncertainty estimates that do not reflect the true predictive uncertainty.
In this study, we compare different metrics, including accuracy and calibration scores, used for model hyperparameter tuning to investigate which model selection strategy achieves well-calibrated models.
Furthermore, we propose to use a computationally efficient Bayesian uncertainty estimation method named Bayesian Linear Probing (BLP), which generates Hamiltonian Monte Carlo (HMC) trajectories to obtain samples for the parameters of a Bayesian Logistic Regression fitted to the hidden layer of the baseline neural network.
We report that BLP improves model calibration and achieves the performance of common uncertainty quantification methods by combining the benefits of uncertainty estimation and probability calibration methods.
Finally, we show that combining post hoc calibration method with well-performing uncertainty quantification approaches can boost model accuracy and calibration.

# **Dependencies**

**Hyperparameter Tuning** is supported with wandb sweeps: wandb >= 0.15.10

Install full conda environment with

```bash
$ conda env create -f config/environment.yaml
```


# **Data**
## ChEMBL Targets [[1]](#1):

- ChEMBL1951: Monoamine oxidase A
- ChEMBL340: Cytochrome P450 3A4
- ChEMBL240: hERG


The data can be downloaded [here](https://zenodo.org/records/12663462?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjI4ZGZmMTkzLWFkYzAtNGYyMC1iNmU1LTIwN2JiNmI5ZjdmZSIsImRhdGEiOnt9LCJyYW5kb20iOiI5ZTM3ZGIwOTU0Yjg2MjNjOTI1ZWE0MjA3YWQ1MzcwZSJ9.ceb44yTkp-0ZDUf2dk6J_dRz2EVreqBXiqUL7LIkLhnFfpUfpqMtqa37CN6xunh917kI0l9TEVQWLNelvZdJIA).


# **Supported Models**
Following models are supported:

- Baseline (MLP)
- Platt - scaled MLP (MLP + P)
- Ensemble Model (MLP-E)
- Platt - scaled Ensemble Model (MLP-E + P)
- MC dropout Model (MLP-D)
- Bayesian linear probing (MLP-BLP)
- Platt - scaled Bayesian linear probing (MLP-BLP + P)

# **Configuration**
## Tuning Hyperparameters of MLP using wandb [[2]](#2)

Start hyperparameter sweep with

```bash
$ wandb sweep config/sweep_baseline.yaml 

```

Copy Sweep-ID and start agent with 
```bash
$ wandb agent sweep_id 

```

Other setting options (e.g. TargetID, Number of model repetitions) can be chosen in sweep_baseline.yaml. 
In this study, hyperparameters were tuned optimizing 4 different metrics:

- Accuracy (acc)
- AUC (rocauc)
- BCE loss (bce)
- Adaptive Calibration Error [[3]](#3) (ace)

# **Model Generation**

## Baseline Model (MLP)/ Ensemble Model (MLP-E) 

Train baseline MLP with 

```bash
$ python train_baseline.py \
--targetid 340 \
--hidden_sizes 20 \
--dropout 0.6 \
--weight_decay 0.01 \
--learning_rate 0.001 \
--hp_metric bce \
```
Choose hyperparameters for hidden_size, dropout, weight_decay and learning_rate according to results in hyperparameter sweep. Specify metric that was chosen to optimize hyperparameters (```--hp-metrics```).

Number of model repetitions can be specified with ```--nr_models``` and number of base estimators used for generating ensemble models can be specified with ```--nr_ensemble_estimators```. If ```--nr_ensemble_estimators```> 1, ensemble models are generated in addition to the baseline models.

## MC Dropout (MLP-D)
> [!WARNING]
> Needs predictions from baseline MLPs (run train_baseline.py)!

Generate MC dropout models from baseline models with

```bash
$ python train_mc.py \
--targetid 340 \
--hidden_sizes 20 \
--dropout 0.6 \
--weight_decay 0.01 \
--learning_rate 0.001 \
--hp_metric bce \
```

Number of forward passes can be specified with ```--nr_mc_estimators```.

## Bayesian Linear Probing (MLP-BLP) [[4]](#4) [[5]](#5) 
1. Hyperparameter tuning of MLP-BLP model:
> [!WARNING]
> Needs hidden layer from baseline MLPs (run train_baseline.py)!

Tune precision of prior distribution (tau_in) with:
```bash
$ wandb sweep config/sweep_BLP.yaml 
```

Copy Sweep-ID and start agent with 
```bash
$ wandb agent sweep_id
```
We recommend a subsequent second sweep for a finer search around the optimum of the first sweep.

2.  MLP-BLP model training:
> [!WARNING]
> Needs hidden layer from baseline MLPs (run train_baseline.py)!

Generate MLP-BLP models with

```bash
$ python train_BLP.py \
--targetid 340 \
--hidden_sizes 20 \
--tau_in 200 \
--hp_metric bce \
```

Choose hyperparameters for tau_in according to results in hyperparameter sweep. Specify metric that was chosen to optimize hyperparameters (```--hp-metrics```).

Other parameters, like number of model repetitions, stepsize and number of generated samples can be specified with ```--nr_models```, ```--step_size``` and ```--num_samples```. 


## Platt Scaling (MLP + P, MLP-E + P, MLP-BLP + P)
> [!WARNING]
> Needs predictions from baseline MLPs (run train_baseline.py)/ ensemble models (run train_baseline.py)/ BLP models (#TODO)!

Generate platt-scaled predictions with:
```bash
$ python train_platt.py \
--targetid 340 \
--hp_metric bce \
```

To apply Platt-Scaling to MLP-E (or to MLP-BLP), set ```--from_ensemble``` (or ```--from_BLP```) to True.

# **Model Evaluation**
To calculate the accuracy, the auc, the calibration error and the Brier score using the model predictions, run:
```bash
$ python evaluation.py /
--targets 240 340 1951 /
--hp_metrics acc auc bce ace
```
The resuts are stored in ```predictions/results.csv```.

The number of generated models, as well as the numberd of base estimators used for generating the ensemble models and the number of forward passes used for MC sopout can be specified with ```--nr_models```, ```--nr_ensemble_estimators```, and ```--nr_mc_estimators'```.

# **Citation**

Cite this repository: Awaiting Review...

# **References**

<a id="1">[1]</a>  Mendez, D., Gaulton, A., Bento, A.P., Chambers, J., De Veij, M., F´elix, E., Magari˜nos, M., Mosquera, J., Mutowo, P., Nowotka, M., Gordillo-Mara˜n´on, M.,
Hunter, F., Junco, L., Mugumbate, G., Rodriguez-Lopez, M., Atkinson, F., Bosc, N., Radoux, C., Segura-Cabrera, A., Hersey, A., Leach, A.: ChEMBL: towards
direct deposition of bioassay data. Nucleic Acids Research 47(D1), 930–940 (2018) https://doi.org/10.1093/nar/gky1075

<a id="2">[2]</a> Biewald, L.: Experiment Tracking with Weights and Biases. Software available
from wandb.com (2020). https://www.wandb.com/

<a id="3">[3]</a>   Nixon, J., Dusenberry, M.W., Zhang, L., Jerfel, G., Tran, D.: Measuring calibration in deep learning. In: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR) Workshops (2019)

<a id="4">[4]</a>   Cobb, A.D., Baydin, A.G., Jalaian, B.: hamiltorch. GitHub (2023). https://github.com/AdamCobb/hamiltorch

<a id="5">[5]</a>   Cobb, A.D., Jalaian, B.: Scaling hamiltonian monte carlo inference for bayesian neural networks with symmetric splitting. In: Uncertainty in Artificial  Intelligence, pp. 675–685 (2021). PMLR







