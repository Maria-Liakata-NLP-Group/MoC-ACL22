seed_value = 0
import os, random, pickle
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)


import nlp.vectorising_approaches, data_handler
from _utils import FOLD_to_TIMELINE, FOLDER_models, PARAMS_bilstm_tl

from sklearn.metrics import f1_score
from collections import Counter

'''
Code for training Orthogonal Procrustes on the timeline-level, using five-fold CV (posts from 100 
timelines per fold):

                                train_model()

will train the model for each (test) fold sequentially. Model and results (predicitons) are stored 
in FOLDER_models and FOLDER_results, respectively.

                    Execution time (approx.) for all folds: 10'
'''


def convert_labels_to_categorical(labels, three_labels=True):
    '''
    Converting string labels to their one-hot version.
    '''
    if three_labels:
        vals = {'0': [1,0,0], 'IE': [0,1,0], 'IS': [0,0,1]}
    else:
        vals = {'0': [1,0,0,0,0], 'IE': [0,1,0,0,0], 'IEP': [0,0,1,0,0], 'IS':[0,0,0,1,0], 'ISB':[0,0,0,0,1]}    
    return np.array([vals[k] for k in labels])


def convert_categorical_to_labels(categories, three_labels=True):
    '''
    Converting one-hot predictions to their actual (string) class.
    '''
    if three_labels:
        vals = ['0', 'IE', 'IS']
    else:
        vals = ['0', 'IE', 'IEP', 'IS', 'ISB']
    return np.array([vals[k] for k in categories])


def get_sequences_per_timeline(ids, pids, features, labels):
    '''
    Given the (train or test) timeline_ids, post_ids, features and labels on a per-post
    basis, it returns their equivalent in a [POST_pre, POST_post] manner.
    '''
    tl_ids, tl_postids, tl_X1, tl_X2, tl_Y = [], [], [], [], []
    for iid in set(ids):
            idx = np.where(np.array(ids)==iid)[0]
            for cnt in range(len(idx)-1):
                tl_ids.append(iid)
                tl_postids.append(np.array(pids)[idx[cnt]])
                tl_X1.append(np.array(features)[idx[cnt]])
                tl_X2.append(np.array(features)[idx[cnt+1]])
                tl_Y.append(np.array(labels)[idx[cnt+1]])
    return np.array(tl_ids), np.array(tl_postids), np.array(tl_X1), np.array(tl_X2), np.array(tl_Y)

def train_model(use_three_labels=True):
    for test_fold in range(len(FOLD_to_TIMELINE)): # for each (test) fold
        print('Training Orthogonal Procrustes\tFold: '+str(test_fold+1)+'/'+str(len(FOLD_to_TIMELINE)))
        
        # Just getting the train/test data: timelines_ids, posts_id, texts, labels
        test_tl_ids, test_pids, test_texts, test_labels = data_handler.get_timelines_for_fold(test_fold)
        train_tl_ids, train_pids, train_texts, train_labels = data_handler.get_timelines_except_for_fold(test_fold)

        # Converting to a 3-label prediction task
        if use_three_labels:
            train_labels, test_labels = data_handler.get_three_labels(train_labels, test_labels)
        Ytrain = convert_labels_to_categorical(train_labels)
        Ytest = convert_labels_to_categorical(test_labels)

        # Feature extraction,
        dummy_txt = "my name is test"
        _, Xtrain, __ = nlp.vectorising_approaches.vectorise(dummy_txt, train_texts, approach="wemb", train=True)
        _, Xtest = nlp.vectorising_approaches.vectorise(dummy_txt, test_texts, approach="wemb", test=True)

        return Xtrain, Xtest
        # Forming sequences of posts in the timelines
        tl_ids_train, postids_train, X1_train, X2_train, Y_train = get_sequences_per_timeline(train_tl_ids, train_pids, Xtrain, Ytrain)
        tl_ids_test, postids_test, X1_test, X2_test, Y_test = get_sequences_per_timeline(test_tl_ids, test_pids, Xtest, Ytest)
