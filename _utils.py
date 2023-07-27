''' File paths, output folders, model params, etc. '''
import os
from os import listdir
from os.path import isfile, join


'''Folders to read the data from'''
FOLDER_complete_dataset = 'MoC_dataset' # directory of the dataset
TYPE_of_labelling = 'three_labels' # or "five_labels"
TYPE_of_agreement = 'majority' # or "intersection" (i.e., requiring majority or perfect agreement)
FOLDER_dataset = FOLDER_complete_dataset+'/'+TYPE_of_labelling+'/'+TYPE_of_agreement+'/'
FILE_tlid_to_multilabel = 'three_label_dict.p'


'''Initialising a list which contains all the timelines to be used in each fold'''
NUM_folds = 5
FOLD_to_TIMELINE = [] # list with NUM_folds sublists, each containing the paths to the corresponding fold's timelines
for _fld in range(NUM_folds):
    _tmp_fldr = FOLDER_dataset+str(_fld)+'/'
    FOLD_to_TIMELINE.append([_tmp_fldr+f for f in listdir(_tmp_fldr) if isfile(join(_tmp_fldr, f))])


'''Folders to save the models and the results on'''
FOLDER_models = 'models/'
FOLDER_results = 'results/'
FOLDER_representations = 'historical_representations/'
if not os.path.exists(FOLDER_models):
    os.makedirs(FOLDER_models)
if not os.path.exists(FOLDER_results):
    os.makedirs(FOLDER_results)


'''Model Parameters'''
PARAMS_logreg = {'C':[.001, .01, .1, 1.0, 10.0, 100.0]}
PARAMS_rf = {'n_estimators':[50, 100, 250, 500]}
PARAMS_bilstm = {'epochs': 100,
                 'early_stop_patience': 5,
                 'num_timesteps': 35, # for 80% percentile
                 'batch_size':[128,256],
                 'lr': [0.001, 0.0001],
                 'num_units_1': [64,128,256],
                 'num_units_2': [64,128,256],
                 'dropout': [.25, .5, .75]}
PARAMS_bilstm_tl = {'epochs': 100,
                 'early_stop_patience': 5,
                 'num_timesteps': 124, # all of them
                 'batch_size':[16,32,64],
                 'lr': [0.001, 0.0001],
                 'num_units_1': [64,128,256],
                 'num_units_2': [124],
                 'dropout': [.25, .5, .75]}
BERT_params = {'dout': [.25],#[.25, .5]
               'lr': [1e-5, 3e-5], #[1e-04, 1e-05], 5e-5, 3e-5.
               'max_len': [512],
               'train_batch': 8,
               'valid_batch': 8,
               'epochs':3}