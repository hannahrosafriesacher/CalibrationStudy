import os 
import numpy as np
import torch

import torch.optim as optim
from calibration_study.baseline import Baseline

from calibration_study.utils import calcCalibrationErrors, calcBrier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

import pandas as pd
import wandb

import yaml
import warnings


warnings.filterwarnings('ignore')
wandb.login()

#Comment out when running on VSC
os.environ['CUDA_VISIBLE_DEVICES']='1'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(targetid, epoch_number, batch_size, nr_models, hidden_sizes, dropout, weight_decay, learning_rate):

    data_path = f'data/CHEMBL{targetid}'

    #to Torch
    X_train = np.load(f'{data_path}/X_train.npy', allow_pickle = True)
    Y_train = np.load(f'{data_path}/Y_train.npy', allow_pickle = True)
    X_val = np.load(f'{data_path}/X_val.npy', allow_pickle = True)
    Y_val = np.load(f'{data_path}/Y_val.npy', allow_pickle = True)
    X_test = np.load(f'{data_path}/X_test.npy', allow_pickle = True)
    Y_test = np.load(f'{data_path}/Y_test.npy', allow_pickle = True)

    X_train_torch = torch.from_numpy(X_train).float().to(device)
    Y_train_torch = torch.from_numpy(Y_train).to(device)
    X_val_torch = torch.from_numpy(X_val).float().to(device)
    Y_val_torch = torch.from_numpy(Y_val).to(device)
    X_test_torch = torch.from_numpy(X_test).float().to(device)
    num_input_features = X_train.shape[1]


    metric_sums = {
    'loss_train': 0, 'acc_train': 0, 'auc_roc_train': 0, 'auc_pr_train': 0, 'ECE_train': 0, 'ACE_train': 0, 'brier_train': 0,
    'loss_val': 0, 'acc_val': 0, 'auc_roc_val': 0, 'auc_pr_val': 0, 'ECE_val': 0, 'ACE_val': 0, 'brier_val': 0,
    }   
    
    for model_idx in range(nr_models):  
        net = Baseline(hidden_sizes=hidden_sizes, input_features=num_input_features, output_features=1, dropout=dropout).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Initialize best metric values
        best_metrics = {'loss_train': 1, 'acc_train': 0, 'auc_roc_train': 0, 'auc_pr_train': 0, 'ECE_train': 1,
                        'ACE_train': 1, 'brier_train': 1, 'loss_val': 1, 'acc_val': 0, 'auc_roc_val': 0,
                        'auc_pr_val': 0, 'ECE_val': 1, 'ACE_val': 1, 'brier_val': 1
                        }

        # Define comparison functions
        comparison_functions = {
            'loss_train': min, 'acc_train': max, 'auc_roc_train': max, 'auc_pr_train': max, 'ECE_train': min,
            'ACE_train': min, 'brier_train': min, 'loss_val': min, 'acc_val': max,
            'auc_roc_val': max, 'auc_pr_val': max, 'ECE_val': min, 'ACE_val': min, 'brier_val': min
        }       

        #training loop
        for epoch in range(epoch_number):  # loop over the dataset multiple times 
            permutation = torch.randperm(X_train_torch.size()[0])
            
            for i in range(0,X_train.size()[0], batch_size):
                optimizer.zero_grad()
                # get the inputs; data is a list of [inputs, labels]
                indices = permutation[i:i+batch_size]
                inputs, labels = X_train_torch[indices], Y_train_torch[indices]        
                # forward + backward + optimizer
                net.eval()
                outputs = net(inputs)
                net.train()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            #get loss of each epoch for plotting convergence
            net.eval()
            
            #predict Training and Validation Dataset
            pred_train_logits = net(X_train_torch, return_hidden  =  0).detach()
            pred_val_logits = net(X_val_torch, return_hidden  =  0).detach()
            pred_train = torch.special.expit(pred_train_logits).cpu().numpy()
            pred_val = torch.special.expit(pred_val_logits).cpu().numpy()
            pred_train_labels = np.where(pred_train>0.5,1.0,0.0)
            pred_val_labels = np.where(pred_val>0.5,1.0,0.0)
            

            pred_train_logits_cpu = pred_train_logits.cpu().numpy()
            pred_val_logits_cpu = pred_val_logits.cpu().numpy()
            
            #Scores Training Dataset
            loss_train = criterion(net(X_train_torch, return_hidden = 0), Y_train_torch).detach().item()
            acc_train = accuracy_score(Y_train, pred_train_labels)
            auc_roc_train = roc_auc_score(Y_train, pred_train)
            precision_train, recall_train, _  =  precision_recall_curve(Y_train, pred_train)
            auc_pr_train  =  auc(recall_train, precision_train)
            ECE_train = calcCalibrationErrors(Y_train, pred_train_logits_cpu,10)[0]
            ACE_train = calcCalibrationErrors(Y_train, pred_train_logits_cpu,10)[1]
            brier_train = calcBrier(Y_train, pred_train_logits_cpu)

            #Scores Validation Dataset
            loss_val = criterion(net(X_val_torch, return_hidden  =  0), Y_val+torch).detach().item()
            acc_val = accuracy_score(Y_val, pred_val_labels)
            auc_roc_val = roc_auc_score(Y_val, pred_val)
            precision_val, recall_val, _  =  precision_recall_curve(Y_val, pred_val)
            auc_pr_val  =  auc(recall_val, precision_val)
            ECE_val = calcCalibrationErrors(np.asarray(Y_val.cpu()), pred_val_logits_cpu,10)[0]
            ACE_val = calcCalibrationErrors(np.asarray(Y_val.cpu()), pred_val_logits_cpu,10)[1]
            brier_val = calcBrier(np.asarray(Y_val.cpu()), pred_val_logits_cpu)

            # Current metric values
            current_metrics = {
                'loss_train/'+str(model_idx)+'/': loss_train, 'acc_train/'+str(model_idx)+'/': acc_train, 'auc_roc_train/'+str(model_idx)+'/': auc_roc_train,
                'auc_pr_train/'+str(model_idx)+'/': auc_pr_train, 'ECE_train/'+str(model_idx)+'/': ECE_train, 'ACE_train/'+str(model_idx)+'/': ACE_train,
                'brier_train/'+str(model_idx)+'/': brier_train, 'loss_val/'+str(model_idx)+'/': loss_val, 'acc_val/'+str(model_idx)+'/': acc_val,
                'auc_roc_val/'+str(model_idx)+'/': auc_roc_val, 'auc_pr_val/'+str(model_idx)+'/': auc_pr_val, 'ECE_val/'+str(model_idx)+'/': ECE_val,
                'ACE_val/'+str(model_idx)+'/': ACE_val, 'brier_val/'+str(model_idx)+'/': brier_val
            }

            wandb.log(current_metrics)

            # Update best metric values
            for metric, current_value in current_metrics.items():
                compare = comparison_functions[metric]
                best_metrics[metric] = compare(current_value, best_metrics[metric])


        # Update WandB summary and sums
        
        for metric, suffix in comparison_functions.items():

            phase = metric.split('_')[-1]
            wandb.summary[f'{phase.capitalize()}/{metric.capitalize()}/{model_idx}/.{suffix}'] = best_metrics[metric]
            metric_sums[metric] += best_metrics[metric]


    for metric, suffix in comparison_functions.items():
        wandb.summary[f'{phase.capitalize()}/{metric.capitalize()}/average']  = metric_sums[metric]/nr_models

    
def main():
    with open("config/sweep_baseline.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader) 
    wandb.init(config=config) 

    id = wandb.config.targetid
    epochs = wandb.config.epoch_number
    bs = wandb.config.batch_size
    nr_m = wandb.config.nr_models
    lr = wandb.config.learning_rate
    hs = wandb.config.hidden_sizes
    do = wandb.config.dropout
    wd = wandb.config.weight_decay 

    train(targetid = id, epoch_number = epochs, batch_size = bs, nr_models = nr_m, hidden_sizes = hs, dropout = do, weight_decay = wd, learning_rate = lr)
    wandb.finish()

main()


