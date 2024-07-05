import torch
import hamiltorch
import numpy as np

import torch.nn as nn
import matplotlib.pyplot as plt

import wandb
from torchmetrics.classification import Accuracy, AUROC, CalibrationError
from calibration_study.utils import get_predictions, generate_file_path, calcCalibrationErrors
from calibration_study.models import Logistic
import calibration_study.metrics as metrics
import sklearn.linear_model as lm
import argparse


parser  =  argparse.ArgumentParser(description = 'Training a single-task classification model. Save MLP predictions and Hidden Layer for MLP model.')
parser.add_argument('--targetid', type = int, help = 'ChEMBL-ID')
parser.add_argument('--hp_metric', type = str, required=True, help="Optimization metric used for learning the original NN", choices=["bce","rocauc","ace","acc"])
parser.add_argument('--hidden_sizes', type = int)
parser.add_argument('--nr_models', type = int, default = 10, help = 'Nr of model repeats')
parser.add_argument('--step_size', default = 0.1, type = float)
parser.add_argument('--num_samples', type = int, default = 1000)
parser.add_argument('--burnin', type = int, default = 500)
parser.add_argument('--num_steps_per_samples', type  =  int, default = 2100) 
parser.add_argument('--tau_in', required=True, help="Prior precision", type=float)
args = parser.parse_args()

targetid=args.targetid
hp_metric = args.hp_metric
hidden_sizes = args.hidden_sizes
nr_models = args.nr_models
step_size = args.step_size
num_samples = args.num_samples
burnin = args.burnin
num_steps_per_sample = args.num_steps_per_samples
tau_in= args.tau_in

Accuracy = Accuracy(task="binary")
AUROC = AUROC(task="binary")
ECE =  metrics.ECE(10)
ACE = metrics.ACE(10)
Brier = metrics.BrierScore()


def autocorr(x,t):
    if t==0:
        return 1.0
    return np.corrcoef(x[:-t], x[t:])[0,1]

def ipse(x):
    "Initial Positive Sequence Estimator"
    max_steps = x.shape[0]//2
    res = -1
    pair = 0
    i = 0
    while (pair >= 0):
        res += pair
        pair = 2*(autocorr(x,2*i) + autocorr(x, 2*i+1))
        i += 1
    return x.shape[0]/res



data_path = f'data/CHEMBL{targetid}'
Y = torch.from_numpy(np.load(f'{data_path}_Y_train.npy', allow_pickle = True))

y_val = torch.from_numpy(np.load(f'{data_path}_Y_val.npy', allow_pickle = True))
y_val_int = y_val.type(torch.int64)
y_test = torch.from_numpy(np.load(f'{data_path}_Y_test.npy', allow_pickle = True))

net = Logistic(hidden_sizes) 

params_init = hamiltorch.util.flatten(net).clone()
tau_list = torch.ones_like(params_init) * tau_in 
inv_mass = torch.ones_like(tau_list) * 100

report_inv_mass = "Hessian based SAI"
tau_out = 1.

for model_idx in range(nr_models):
    print("=============================")
    print(f"       Model_idx {model_idx}")
    print("=============================")

#Data loading
    hidden_file_test = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_test_hidden')
    hidden_file_val = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_val_hidden')
    hidden_file_train = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_train_hidden')
    
    X = torch.Tensor(np.load(hidden_file_train))
    x_val = torch.Tensor(np.load(hidden_file_val))    
    x_test = torch.Tensor(np.load(hidden_file_test))

    net = Logistic(hidden_sizes) 
    
    #Performing Laplace approximation to initialize the mass matrix ...
    eps = 100.0 #due to the scale of Hessian is large
    scale = 100.0
    LR = lm.LogisticRegression().fit(X, Y)
    res = LR.predict_proba(X)

    #Nonregularized Hessian:
    H =  (X.T @ torch.Tensor(np.diag(res[:,0]*res[:,1])) @ X)
    H = H + eps * np.eye(X.shape[1]) #Scaled and regularized
    
    #Regularized Hessian:
    diagonal_elements = np.diag(H) / (H**2).sum(0) * scale
    #Scaled preconditioning diagonal:
    inv_mass[:-1] = diagonal_elements
    inv_mass[-1] = 1/eps



    params_hmc = hamiltorch.sample_model(model = net,
                                                x = X, 
                                                y = Y,
                                            burn = burnin,
                                    params_init = params_init, 
                                    num_samples = num_samples, 
                                    step_size   = step_size, 
                        num_steps_per_sample   = num_steps_per_sample, 
                                    tau_out     = tau_out, 
                                    model_loss  = 'binary_class_linear_output', 
                                    tau_list    = tau_list,
                                    inv_mass    = inv_mass)


     #Calculating metrics on val
    #===========================
    pred_val, list_log_prob = hamiltorch.predict_model(model = net,
                                                        x = x_val, 
                                                        y = y_val,
                                                samples  = params_hmc,
                                                model_loss = 'binary_class_linear_output', 
                                                tau_out  = tau_out,
                                                tau_list = tau_list)
    #Calculating metrics on Test
    avgpred_val = torch.nn.functional.sigmoid(pred_val).mean(0)
      
    #Calculating metrics on test
    #===========================
    pred_test, list_log_prob = hamiltorch.predict_model(model = net,
                                                        x = x_test, 
                                                        y = y_test,
                                                samples  = params_hmc,
                                                model_loss = 'binary_class_linear_output', 
                                                tau_out  = tau_out,
                                                tau_list = tau_list)
    
    #Calculating metrics on Test
    avgpred_test = torch.nn.functional.sigmoid(pred_test).mean(0)

    file_path_blp_val = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_val_BLP')
    file_path_blp_test = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_test_BLP')
    np.save(file_path_blp_val, avgpred_val)
    np.save(file_path_blp_test, avgpred_test)
