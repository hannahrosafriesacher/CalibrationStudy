import os 
import numpy as np
import argparse

from calibration_study.utils import generate_file_path

from calibration_study.utils import calcCalibrationErrors, calcBrier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

import pandas as pd

# Define a custom argument type for a list of strings
def list_of_strings(arg):
    return arg.split(',')

parser  =  argparse.ArgumentParser(description = 'Obtain results from predictions.')
parser.add_argument('--targets', type=list_of_strings)
parser.add_argument('--hp_metrics', type=list_of_strings)
parser.add_argument('--nr_models', type = int, default = 10)
parser.add_argument('--nr_ensemble_estimators', type = int, default = 10)
parser.add_argument('--nr_mc_estimators', type = int, default = 100)
args = parser.parse_args()

targets = args.targets
metrics = args.hp_metrics
nr_models = args.nr_models
nr_ensemble_estimators = args.nr_ensemble_estimators
nr_mc_estimators = args.nr_mc_estimators

results = pd.DataFrame({})
for targetid in targets:
    data_path = f'data/CHEMBL{targetid}'
    Y_test = np.load(f'{data_path}_Y_test.npy', allow_pickle = True)
    for hp_metric in metrics:
        list_acc = []
        list_auc = []
        list_bs = []
        list_ece = []
        list_ace = []
        model_list = []
        model_rep_list = []
        for model_idx in range(nr_models):
            paths = {}
            paths['baseline'] = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_test')
            paths['ensemble'] = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_test_ensemble{nr_ensemble_estimators}')
            paths['mc']  = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_test_mc{nr_mc_estimators}')
            paths['blp'] = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_test_BLP')
            paths['platt']  = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_test')
            paths['platt_ensemble']  = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_test_ensemble_platt')
            paths['platt_blp']  = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_test_blp_platt')


            
            for path in paths:
                assert os.path.exists(paths[path]), f'No predictions found in: {paths[path]}'
                preds = np.load(paths[path])
                model_list.append(path)
                model_rep_list.append(model_idx)
                list_acc.append(accuracy_score(Y_test, np.where(preds > 0.5,1.0,0.0)))
                list_auc.append(roc_auc_score(Y_test, preds))
                list_bs.append(calcBrier(Y_test, preds))
                list_ece.append(calcCalibrationErrors(Y_test, preds, 10)[0])
                list_ace.append(calcCalibrationErrors(Y_test, preds, 10)[1])
        
    results[f'Model'] = model_list
    results[f'Model Rep'] = model_rep_list
    results[f'CHEMBL{targetid} ({hp_metric}): acc'] = list_acc
    results[f'CHEMBL{targetid} ({hp_metric}): auc'] = list_auc
    results[f'CHEMBL{targetid} ({hp_metric}): bs'] = list_bs
    results[f'CHEMBL{targetid} ({hp_metric}): ece'] = list_ece
    results[f'CHEMBL{targetid} ({hp_metric}): ace'] = list_ace
        

results.sort_values(by = ['Model', 'Model Rep']).to_csv('predictions/results.csv', index=False)


