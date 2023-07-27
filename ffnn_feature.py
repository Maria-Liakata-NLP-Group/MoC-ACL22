seed_value = 0
import os, random, pickle
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

import pandas as pd

from sklearn.model_selection import train_test_split

import nlp.vectorising_approaches, data_handler
from _utils import FOLD_to_TIMELINE, FOLDER_models, PARAMS_bilstm_tl

from sklearn.metrics import f1_score
from collections import Counter

'''
Code for training BiLSTM on the timeline-level (i.e., apredicting the label of all posts in a
given timeline), using five-fold CV (posts from 100 timelines per fold):

                                train_model()

will train the model for each (test) fold sequentially, performing a 67/33 split on the training set
for finding the optimal hyperparameter values (check _utils.py for tested parameters/values). Model 
and results (predicitons) are stored in FOLDER_models and FOLDER_results, respectively.

                    Execution time (approx.) for all folds: 8 hours (CPU)
'''


def zero_pad_vectors(data, max_seq_len=124): # 124: Max number of posts in a timeline (minimum is 10)
    '''
    Zero-padding the word vectors so that they have a length of _utils.PARAMS_BILSTM_tl['num_timesteps'].
    '''
    dimensions = data[0].shape[1] #300 for embeddings; 3 for target
    vals_to_add_ = np.array([123 for i in range(dimensions)])

    new_data, weights = [], []
    for i in range(len(data)):
        tmp_weights = np.zeros(max_seq_len)
        tmp = data[i]
        if len(tmp)>=max_seq_len: # Executed only for = and only once (tl with max_#posts)
            new_tmp = tmp[:max_seq_len]
            for j in range(len(tmp)): # Adjust weights
                tmp_weights[j] = 1
        else:
            vec1 = [v for v in tmp]
            for j in range(len(vec1)): # Adjust weights
                tmp_weights[j] = 1
            num_zeros_to_add = max_seq_len-len(vec1)
            vec2 = [vals_to_add_ for k in range(num_zeros_to_add)]
            new_tmp = np.concatenate((vec1,vec2))
        new_data.append(new_tmp)
        weights.append(tmp_weights)
    return np.array(new_data), np.array(weights)


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
    basis, it returns their equivalent in timeline-basis
    '''
    tl_ids, tl_postids, tl_X, tl_Y = [], [], [], []
    iids = []
    for iid in ids:
        if iid not in iids:
            iids.append(iid)
    for iid in iids:
            idx = np.where(np.array(ids)==iid)[0]
            tl_ids.append(iid)
            tl_postids.append(np.array(pids)[idx])
            tl_X.append(np.array(features)[idx])
            tl_Y.append(np.array(labels)[idx])
    return np.array(tl_ids), np.array(tl_postids), np.array(tl_X), np.array(tl_Y)


def get_features_for_data(ids, type_of_model):
    '''
    type_of_model: fsdc, fsdnn, tweeteval, deepmoji 
    '''
    infile = 'em_fsd_scd/'
    keyword = 'post_id'
    if type_of_model=='fsdc':
        df = pd.read_csv(infile+'fsdc_output', sep='\t')
        start = 2
    elif type_of_model=='fsdnn':
        df = pd.read_csv(infile+'fsdnn_output', sep='\t')
        start = 2
    elif type_of_model=='deepmoji':
        df = pd.read_csv(infile+'deepmoji_output.csv')
        keyword = 'Id'
        start = 1
    elif type_of_model=='tweeteval':
        df = pd.read_csv(infile+'tweeteval_output.tsv', sep='\t')
        keyword = 'postid'
        start = 2
    elif type_of_model[:3]=='scd':
        df = pd.read_csv(infile+type_of_model+'.tsv', sep='\t')
        keyword = 'post_id'
        start = 2
    feats = []
    for i in range(len(ids)):
        if type_of_model[:3]!='scd':
            vals = df[df[keyword]==int(ids[i])].values[0][start:]
        else:
            vals = [float(v) for v in df[df[keyword]==int(ids[i])].values[0][start][1:-1].replace('\n', ' ').split()]
        feats.append(vals)
    return np.array(feats)


if __name__=='__main__':
    feature_type = 'scd_op_we_vecs'
    print(feature_type)
    use_three_labels=True
    for test_fold in range(len(FOLD_to_TIMELINE)): # for each (test) fold
        print('Training BiLSTM \tFold: '+str(test_fold+1)+'/'+str(len(FOLD_to_TIMELINE)))
        
        # Just getting the train/test data: timelines_ids, posts_id, texts, labels
        test_tl_ids, test_pids, test_texts, test_labels = data_handler.get_timelines_for_fold(test_fold)
        train_tl_ids, train_pids, train_texts, train_labels = data_handler.get_timelines_except_for_fold(test_fold)

        # Converting to a 3-label prediction task
        if use_three_labels:
            train_labels, test_labels = data_handler.get_three_labels(train_labels, test_labels)
        Ytrain = convert_labels_to_categorical(train_labels)
        Ytest = convert_labels_to_categorical(test_labels)

        # Feature extraction
        if feature_type[:4]!='bert':
            Xtrain = get_features_for_data(train_pids, feature_type)
            Xtest = get_features_for_data(test_pids, feature_type)
        else:
            #This is where I load the BERT model representations:
            mydict = pickle.load(open("em_fsd_scd/representations_"+feature_type+'.p', 'rb'))
            Xtest = np.array(mydict['test_vectors'][test_fold])
            Xtrain = np.array(mydict['train_vectors'][test_fold])


        X_train_actual, Xval, y_train_actual, Yval = train_test_split(Xtrain, Ytrain, test_size=0.33, random_state=seed_value)
        print(X_train_actual.shape, Xval.shape)

        all_results = dict()
        best_val_loss_overall = 1000000 # keeping track of best loss
        best_eval_score = -1.0

        # Now, training for all combinations of different parameters
        for batch in PARAMS_bilstm_tl['batch_size']:
            for lr in PARAMS_bilstm_tl['lr']:
                for dout in PARAMS_bilstm_tl['dropout']:
                    for units1 in PARAMS_bilstm_tl['num_units_1']:
                        for units2 in [124]:

                            outfile = str(batch)+'_'+str(lr)+'_'+str(dout)+'_'+str(units1)+'_'+str(units2)

                            # Define model
                            model = Sequential()
                            #model.add(Masking(mask_value=123, input_shape=(124, Xval.shape[2])))
                            model.add(Dense(units1, activation='relu'))
                            model.add(Dropout(dout))
                            model.add(Dense(units1, activation='relu'))
                            model.add(Dropout(dout))
                            model.add(Dense(3, activation='softmax'))

                            # Early stopping if performance does not increase; set optimizer & loss
                            es = EarlyStopping(monitor='val_loss', mode='min', patience=PARAMS_bilstm_tl['early_stop_patience'])
                            model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['categorical_accuracy'])

                            history = model.fit(X_train_actual, y_train_actual, validation_data=(Xval, Yval), verbose=0, shuffle=True,
                                                callbacks=[es], epochs=PARAMS_bilstm_tl['epochs'], batch_size=batch)
                            

                            # Making preds on val set to measure F1-macro
                            pred_scores_val = model.predict(Xval)
                            preds_val, actual_val = [], []
                            for instance in range(len(pred_scores_val)):
                                actual_val.append(Yval[instance].argmax())
                                preds_val.append(pred_scores_val[instance].argmax())
                            actual_val, preds_val = np.array(actual_val), np.array(preds_val)
                            preds_val = convert_categorical_to_labels(preds_val)
                            actual_val = convert_categorical_to_labels(actual_val)

                            val_score = f1_score(actual_val, preds_val, average='macro') #the score we need
                            
                            #Making preds on test set
                            pred_scores = model.predict(Xtest) # shape: (num_timelines, num_timesteps, num_labels)
                            preds, actual = [], []
                            for instance in range(len(pred_scores)):
                                actual.append(Ytest[instance].argmax())
                                preds.append(pred_scores[instance].argmax())
                            actual, preds = np.array(actual), np.array(preds)
                            preds = convert_categorical_to_labels(preds)
                            actual = convert_categorical_to_labels(actual)
                            eval_score = f1_score(actual, preds, average='macro') #accuracy on test set  

                            n_epoch = len(history.history['val_loss'])                            
                            
                            if val_score>best_eval_score:
                                best_eval_score = val_score
                                data_handler.save_results(model, test_tl_ids, test_pids, actual, preds, feature_type, test_fold, use_three_labels, 'ffnnnn_feature')
                            
                            print(test_fold, '\t', batch, '\t', lr, '\t', dout, '\t', units1, '\t', units2, '\t', 
                                    Counter(preds), '\t', n_epoch, '\t', val_score, '\t', eval_score)
                            