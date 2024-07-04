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
import os
import yaml

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

def train(targetid, hp_metric, hidden_sizes, nr_models, step_size, num_samples, burnin, num_steps_per_sample, tau_in):

    data_path = f'data/CHEMBL{targetid}'
    Y = torch.from_numpy((np.load(f'{data_path}_Y_train.npy', allow_pickle = True) + 1)/2)

    #Validation Data
    y_val = torch.from_numpy(np.load(f'{data_path}_Y_val.npy', allow_pickle = True))
    y_val_int = y_val.type(torch.int64)

    net = Logistic(hidden_sizes) 

    params_init = hamiltorch.util.flatten(net).clone()
    tau_list = torch.ones_like(params_init) * tau_in 
    inv_mass = torch.ones_like(tau_list) * 100

    report_inv_mass = "Hessian based SAI"
    tau_out = 1.
    

    #Weghts and biasses loging
    wandb.init(project = f'JCIM_BLP_{targetid}_{hp_metric}')

    #wandb.init(project=proj_name)
    wandb.config.update({"step_size" : step_size,
                "L" : num_steps_per_sample,
                "total_samples" : num_samples,
                "burnin" : burnin,
                "tau_out" : tau_out,
                "tau_in" : tau_list[0],
                "inverse_masses": report_inv_mass})

    accumulator = {"ESS_cutoff_0.05": 0.0,
                "ESS_cutoff_0.005" : 0.0,
                "ESS_cutoff_sc_0.05": 0.0,
                "ESS_cutoff_sc_0.005" : 0.0,
                "ESS_IPSE" : 0.0,
                "ESS_err_cutoff_0.05": 0.0,
                "ESS_err_cutoff_0.005" : 0.0,
                "ESS_err_cutoff_sc_0.05": 0.0,
                "ESS_err_cutoff_sc_0.005" : 0.0,
                "ESS_err_IPSE" : 0.0,
                "val_acc":0.0,
                "val_ECE" : 0.0,
                "val_AUC" :0.0,
                "val_ACE": 0.0,
                "val_Brier": 0.0,
                "tr_acc":0.0,
                "tr_ECE" : 0.0,
                "tr_AUC" :0.0,
                "val_logloss" :0.0,
                "tr_logloss" :0.0,
                "test_acc":0.0,
                "test_ECE" : 0.0,
                "test_AUC" :0.0,
                "test_ACE": 0.0,
                "test_Brier": 0.0,
                "test_logloss" :0.0,
                }
    for model_idx in range(nr_models):
        print("=============================")
        print(f"       Model_idx {model_idx}")
        print("=============================")

    #Data loading
        hidden_file_test = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_test_hidden')
        hidden_file_val = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_val_hidden')
        hidden_file_train = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_train_hidden')
        
        
        print("Opening training X file: ", hidden_file_train)
        X = torch.Tensor(np.load(hidden_file_train))
        print("Opening validation X file: ", hidden_file_val)
        x_val = torch.Tensor(np.load(hidden_file_val))

        net = Logistic(hidden_sizes) 
       
        print("Performing Laplace approximation to initialize the mass matrix ...")
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


        #Compute IPSE on all parameters and take the minimum
        params = torch.stack(params_hmc)
        ESSs = np.zeros(params.shape[1])
        for i in range(params.shape[1]):
            ESSs[i] = ipse(params[:,i])
        ess_ipse_parmin = np.min(ESSs)
        wandb.summary[f"ESS_IPSE_param_min_{model_idx}"] = ess_ipse_parmin

        #Calculating metrics on validation
        #=================================
        pred, list_log_prob = hamiltorch.predict_model(model = net,
                                                        x = x_val, 
                                                        y = y_val,
                                                    samples  = params_hmc,
                                                model_loss = 'binary_class_linear_output', 
                                                    tau_out  = tau_out,
                                                    tau_list = tau_list)


        #Mixing and Convergence diagnostics of a single prediction
        max_autocorr = num_samples - burnin
        scaling = 1 - np.linspace(1,max_autocorr-1,max_autocorr-1)/(num_samples - burnin)
        
        
        
        xidx = 40
        acorr = [autocorr(pred[:,xidx,0],t) for t in range(1,max_autocorr)] #0

        ess_ipse = ipse(pred[:,xidx,0])
        wandb.summary[f"ESS_IPSE_{model_idx}"] = ess_ipse
        accumulator[f"ESS_IPSE"] += ess_ipse


        #Mixing and Convergence on the error (data likelihood)
        err = [((pred[i,:,0]-y_val[:,0])**2).sum() for i in range(pred.shape[0])]
        acorr = [autocorr(err,t) for t in range(1,max_autocorr)] 

        ess_err_ipse = ipse(torch.tensor(err))
        wandb.summary[f"ESS_err_IPSE_{model_idx}"] = ess_err_ipse
        accumulator[f"ESS_err_IPSE"] += ess_err_ipse

        #Calculating metrics on Validation
        avgpred = torch.nn.functional.sigmoid(pred).mean(0)
        
        acc_val = Accuracy((avgpred>0)*1.0, y_val_int)
        

        wandb.summary[f"val_acc_{model_idx}"] = acc_val
        accumulator[f"val_acc"] += acc_val
        Accuracy.reset()
        auc_val = AUROC(avgpred, y_val_int)

        wandb.summary[f"val_AUC_{model_idx}"] = auc_val 
        accumulator[f"val_AUC"] += auc_val
        AUROC.reset()
        ECE_val = ECE(avgpred[:,0], y_val_int[:,0])

        wandb.summary[f"val_ECE_{model_idx}"] = ECE_val
        accumulator[f"val_ECE"] += ECE_val
        ECE.reset()
        ACE_val = ACE(avgpred[:,0], y_val_int[:,0])

        wandb.summary[f"val_ACE_{model_idx}"] = ACE_val
        accumulator[f"val_ACE"] += ACE_val
        ACE.reset()
        Brier_val = Brier(avgpred[:,0], y_val_int[:,0])

        wandb.summary[f"val_Brier_{model_idx}"] = Brier_val
        accumulator[f"val_Brier"] += Brier_val
        Brier.reset()

        nll_val = torch.stack(list_log_prob).mean()
        wandb.summary[f"val_logloss_{model_idx}"] = nll_val
        accumulator[f"val_logloss"] += nll_val

    for key in accumulator.keys():
        wandb.summary[key+"_TOT"] = accumulator[key]/model_idx

def main():
    with open("config/sweep_baseline.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader) 
    wandb.init(config=config) 

    id = wandb.config.targetid
    hp_metric = wandb.config.hp_metric
    hidden_sizes = wandb.config.hidden_sizes
    nr_models = wandb.config.nr_models
    step_size = wandb.config.step_size
    num_samples = wandb.config.num_samples
    burnin = wandb.config.burnin
    num_steps = wandb.config.num_steps_per_sample
    tau_in = wandb.config.tau_in

    train(targetid = id, hp_metric = hp_metric, hidden_sizes = hidden_sizes, nr_models = nr_models, step_size = step_size, num_samples = num_samples, burnin = burnin, num_steps_per_sample = num_steps, tau_in = tau_in)
    wandb.finish()

main()