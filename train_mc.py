import os
import argparse

import numpy as np
import torch
import torch.optim as optim
from scipy.special import expit
from sklearn.metrics import roc_auc_score, accuracy_score
from calibration_study.models import Baseline
from calibration_study.utils import get_predictions, generate_file_path



parser  =  argparse.ArgumentParser(description = 'Generating a Monte-Carlo dropout model from a baseline model.')

parser.add_argument('--targetid', type = int, help = 'ChEMBL-ID')
parser.add_argument('--hidden_sizes', type = int)
parser.add_argument('--dropout', type = float)
parser.add_argument('--weight_decay', type = float)
parser.add_argument('--learning_rate', type = float)
parser.add_argument('--epoch_number', type = int, default = 400)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--hp_metric', type = str, required=True, help="Optimization metric used for learning the original NN", choices=["bce","rocauc","ace","acc"])
parser.add_argument('--nr_models', type = int, default = 10, help = 'Nr of model repeats')
parser.add_argument('--nr_mc_estimators', type  =  int, default = 100, help = 'Nr of base estimators for inference') 
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES']='0'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

targetid=args.targetid
hp_metric = args.hp_metric
hidden_sizes = args.hidden_sizes
dropout = args.dropout
weight_decay = args.weight_decay
learning_rate = args.learning_rate
epoch_number = args.epoch_number
batch_size = args.batch_size
nr_models = args.nr_models
nr_mc_estimators = args.nr_mc_estimators



data_path = f'data/CHEMBL{targetid}'

#to Torch
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
Y_test_torch = torch.from_numpy(Y_test).float().to(device)
num_input_features = X_train.shape[1]

pred_val_list = []
pred_test_list = []

for model_idx in range(nr_models):
    
    net = Baseline(hidden_sizes=hidden_sizes, input_features=num_input_features, output_features=1, dropout=dropout).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    model_path = generate_file_path(type = 'models', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}')

    #load_model
    state_dict = torch.load(model_path)
    net.load_state_dict(state_dict, strict=False)
        
    net.train()

    predictions_sum = np.zeros(shape = Y_test_torch.shape)
    for mc_estimator in range(nr_mc_estimators): 

        #predict MC Dropout
        pred_test = net(X_test_torch, return_hidden=0)
        pred_cpu_test = pred_test.detach()

        #average
        predictions_sum+=expit(pred_cpu_test.cpu().numpy())
    
    prediction_te_mc = predictions_sum/nr_mc_estimators
    prediction_file_test= generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_test_mc{nr_mc_estimators}')
    np.save(prediction_file_test, prediction_te_mc)