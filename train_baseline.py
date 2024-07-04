import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.special import expit
from sklearn.metrics import roc_auc_score, accuracy_score
from calibration_study.models import Baseline
from calibration_study.utils import get_predictions, generate_file_path, calcCalibrationErrors

parser  =  argparse.ArgumentParser(description = 'Training a single-task classification model. Save MLP predictions and Hidden Layer for MLP model.')
parser.add_argument('--targetid', type = int, help = 'ChEMBL-ID')
parser.add_argument('--hidden_sizes', type = int)
parser.add_argument('--dropout', type = float)
parser.add_argument('--weight_decay', type = float)
parser.add_argument('--learning_rate', type = float)
parser.add_argument('--epoch_number', type = int, default = 400)
parser.add_argument('--batch_size', type = int, default = 200)
parser.add_argument('--hp_metric', type = str, default = 'bce', help = 'HP-metric used for early stopping')
parser.add_argument('--nr_models', type = int, default = 10, help = 'Nr of model repeats')
parser.add_argument('--nr_ensemble_estimators', type  =  int, default = 10, help = 'Nr of base estimators in ensemble models') 
args = parser.parse_args()

targetid=args.targetid
hp_metric = args.hp_metric
hidden_sizes = args.hidden_sizes
dropout = args.dropout
weight_decay = args.weight_decay
learning_rate = args.learning_rate
epoch_number = args.epoch_number
batch_size = args.batch_size
nr_models = args.nr_models
nr_ensemble_estimators = args.nr_ensemble_estimators


os.environ['CUDA_VISIBLE_DEVICES']='0'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

assert hp_metric in ['acc', 'rocauc', 'bce', 'ace'], f'No implementation for HP Metric {hp_metric}. Available options are Accuracy (acc), AUC (rocauc), BCE-loss (bce), Adaptive Calibrtion Error (ace)'
assert targetid in  [240, 340, 1951], f'No data for Target with ChEMBL-ID {targetid}. Available Targets with ChEMBL IDs: 240, 340, 1951'

model_reps = nr_models * nr_ensemble_estimators
data_path = f'data/CHEMBL{targetid}'

X_train = np.load(f'{data_path}_X_train.npy', allow_pickle = True)
Y_train = np.load(f'{data_path}_Y_train.npy', allow_pickle = True)
X_val = np.load(f'{data_path}_X_val.npy', allow_pickle = True)
Y_val = np.load(f'{data_path}_Y_val.npy', allow_pickle = True)
X_test = np.load(f'{data_path}_X_test.npy', allow_pickle = True)
Y_test = np.load(f'{data_path}_Y_test.npy', allow_pickle = True)

X_train_torch = torch.from_numpy(X_train).float().to(device)
Y_train_torch = torch.from_numpy(Y_train).to(device)
X_val_torch = torch.from_numpy(X_val).float().to(device)
Y_val_torch = torch.from_numpy(Y_val).to(device)
X_test_torch = torch.from_numpy(X_test).float().to(device)
num_input_features = X_train.shape[1]

pred_val_list = []
pred_test_list = []


for model_idx in range(model_reps):
    metric_best = 0
    
    net = Baseline(hidden_sizes=hidden_sizes, input_features=num_input_features, output_features=1, dropout=dropout).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(epoch_number): 
        permutation = torch.randperm(X_train_torch.size()[0])
        i=0
        for i in range(0,X_train_torch.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            inputs, labels = X_train_torch[indices], Y_train_torch[indices]        
            
            net.eval()
            outputs = net(inputs, return_hidden=0)

            net.train()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        net.eval()

        pred_val_logits=net(X_val_torch,return_hidden=0).detach()
        pred_val_logits_cpu=pred_val_logits.cpu().numpy()
        
        pred_val=torch.special.expit(pred_val_logits).cpu().numpy()
        pred_val_labels=np.where(pred_val>0.5,1.0,0.0)
        

        metrics_dict = {
            'acc': lambda: accuracy_score(Y_val, pred_val_labels),
            'bce': lambda: criterion(net(X_val_torch, return_hidden=0), Y_val_torch).detach().item(),
            'ace': lambda: calcCalibrationErrors(Y_val, pred_val_logits_cpu, 10)[1],
            'rocauc': lambda: roc_auc_score(Y_val, pred_val)
        }

        if hp_metric in metrics_dict:
            metric = metrics_dict[hp_metric]()
        else:
            raise NotImplementedError(f"Metric '{hp_metric}' is not implemented.")

        if epoch == 0:
            metric_best = metric
            preds = get_predictions(net, X_val_torch, X_test_torch, X_train_torch)
            model_file = generate_file_path(type = 'models', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}')
            torch.save(net.state_dict(), model_file)
        else:
            is_better = (hp_metric in ['acc', 'rocauc'] and metric > metric_best) or (hp_metric in ['bce', 'ace'] and metric < metric_best)
            if is_better:
                metric_best = metric
                preds = get_predictions(net, X_val_torch, X_test_torch, X_train_torch)                
                model_file = generate_file_path(type = 'models', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}')
                torch.save(net.state_dict(), model_file)


    
    if model_idx < nr_models:
        # Construct file paths
        predictions_file_test_baseline = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_test')
        predictions_file_val_baseline = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_val')

        predictions_file_test_hidden_bayesian = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_test_hidden')
        predictions_file_val_hidden_bayesian = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_val_hidden')
        predictions_file_train_hidden_bayesian = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_train_hidden')

        # Save Baseline Model Predictions
        np.save(predictions_file_test_baseline, expit(preds['test'].numpy()).flatten())
        np.save(predictions_file_val_baseline, expit(preds['val'].numpy()).flatten())

        #Save Hidden Layer for BLP Model
        np.save(predictions_file_test_hidden_bayesian, preds['test_hidden'])
        np.save(predictions_file_val_hidden_bayesian, preds['val_hidden'])
        np.save(predictions_file_train_hidden_bayesian, preds['train_hidden'])

        pred_val_list.append(expit(preds['val'].numpy()).flatten())
        pred_test_list.append(expit(preds['test'].numpy()).flatten())

    else:
        pred_val_list.append(expit(preds['val'].numpy()).flatten())
        pred_test_list.append(expit(preds['test'].numpy()).flatten())

pred_val = np.array(pred_val_list)
pred_test = np.array(pred_test_list)

#Save Ensemble Model
if nr_ensemble_estimators > 1:
    sm = 0
    for ensemble_idx in range(0, model_reps, nr_ensemble_estimators):
        mean_val = pred_val[ensemble_idx : ensemble_idx + nr_ensemble_estimators].mean(axis=0)
        mean_test = pred_test[ensemble_idx : ensemble_idx + nr_ensemble_estimators].mean(axis=0)
        file_path_ensemble_val = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{sm}_{hp_metric}_val_ensemble{nr_ensemble_estimators}')
        file_path_ensemble_test = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{sm}_{hp_metric}_test_ensemble{nr_ensemble_estimators}')
        np.save(file_path_ensemble_val,  mean_val)
        np.save(file_path_ensemble_test, mean_test)
        sm += 1