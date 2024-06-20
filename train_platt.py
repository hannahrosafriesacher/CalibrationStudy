import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from calibration_study.utils import generate_file_path

import argparse

parser  =  argparse.ArgumentParser(description = 'Performs Platt Scling on Baseline Model, Ensemble Model or BLP Model.')
parser.add_argument('--targetid', type = int, help = 'ChEMBL-ID')
parser.add_argument('--hp_metric', type = str, default = 'bce', help = 'HP-metric used for early stopping')
parser.add_argument('--nr_models', type = int, default = 10, help = 'Nr of model repeats')
parser.add_argument('--nr_ensemble_estimators', type  =  int, default = 10, help = 'Nr of base estimators in ensemble models')
parser.add_argument('--from_ensemble', type  =  bool, default = False, help = 'Platt Scaling of Ensemble Model')
parser.add_argument('--from_BLP', type  =  bool, default = False, help = 'Platt Scaling of BLP Model')
args  =  parser.parse_args()

targetid=args.targetid
hp_metric = args.hp_metric
nr_models = args.nr_models
nr_ensemble_estimators = args.nr_ensemble_estimators
from_ensemble = args.from_ensemble
from_BLP = args.from_BLP


assert from_ensemble == False or from_BLP == False, 'Both from_enesmble and from_BLP is set to true. Please specify one Option.'

data_path = f'data/CHEMBL{targetid}'
Y_val = np.load(f'{data_path}/Y_val.npy', allow_pickle = True)

for model_idx in range(nr_models):

    if from_ensemble:
        path_val = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_val_ensemble{nr_ensemble_estimators}')
        path_te = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_test_ensemble{nr_ensemble_estimators}')

        file_path_test = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{v}_{hp_metric}_test_ensemble_platt')


    elif from_BLP:
        path_val = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_val_blp')
        path_te = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_test_blp')

        file_path_test = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{v}_{hp_metric}_test_blp_platt')


    else:
        path_val = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_val')
        path_te = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_test')

        file_path_test = generate_file_path(type = 'predictions', targetid = targetid, suffix = f'rep{model_idx}_{hp_metric}_test_platt')
   

    #Obtain Classification Scores for Platt Scaling
    y_val_preplatt = logit(np.load(path_val).reshape(-1, 1))
    y_te_preplatt = logit(np.load(path_te).reshape(-1, 1))

    #Train model on Validation fold
    lr=LogisticRegression().fit(y_val_preplatt, Y_val)

    #Predict Test fold
    y_platt_te=lr.predict_proba(y_te_preplatt)[:, 1]
    y_platt_te=y_platt_te.flatten()

    np.save(file_path_test,  y_platt_te)





