seed_value = 0
import os, random, pickle
from re import X
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

from sklearn.model_selection import train_test_split

import nlp.vectorising_approaches, data_handler
from _utils import FOLD_to_TIMELINE, FOLDER_models, PARAMS_bilstm

from sklearn.metrics import f1_score
from collections import Counter

'''
Code for training BiLSTM on the post-level (i.e., assuming completely independent posts
regardless of the timeline they come from), using five-fold CV (posts from 100 timelines per fold):

                                train_model()

will train the model for each (test) fold sequentially, performing a 67/33 split on the training set
for finding the optimal hyperparameter values (check _utils.py for tested parameters/values). Model 
and results (predicitons) are stored in FOLDER_models and FOLDER_results, respectively.

                    Execution time (approx.) for all folds: ~30hrs (CPU)
'''



def zero_pad_vectors(data): 
    '''
    Zero-padding the word vectors so that they have a length of _utils.PARAMS_bilstm['num_timesteps'].
    '''
    max_seq_len = PARAMS_bilstm['num_timesteps']
    new_data = []
    for i in range(len(data)):
        tmp = data[i]
        if len(tmp)>=max_seq_len:
            new_tmp = tmp[:max_seq_len]
        else:
            vec1 = tmp
            num_zeros_to_add = max_seq_len-len(tmp)
            vec2 = [np.zeros(300) for k in range(num_zeros_to_add)]
            new_tmp = np.concatenate((vec1,vec2))
        new_data.append(new_tmp)
    return np.array(new_data)


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


def train_model(use_three_labels=True):
    for test_fold in range(len(FOLD_to_TIMELINE)): # for each (test) fold
        print('Training BiLSTM \tFold: '+str(test_fold+1)+'/'+str(len(FOLD_to_TIMELINE)))
        
        # Just getting the train/test data: timelines_ids, posts_id, texts, labels
        test_tl_ids, test_pids, test_texts, test_labels = data_handler.get_timelines_for_fold(test_fold)
        train_tl_ids, train_pids, train_texts, train_labels = data_handler.get_timelines_except_for_fold(test_fold)

        # Converting to a 3-label prediction task
        if use_three_labels:
            train_labels, test_labels = data_handler.get_three_labels(train_labels, test_labels)
        Ytrain = convert_labels_to_categorical(train_labels)

        # Feature extraction & zero-padding
        dummy_txt = "my name is test"
        _, Xtrain, __ = nlp.vectorising_approaches.vectorise(dummy_txt, train_texts, approach="wemb_word", train=True)
        _, Xtest = nlp.vectorising_approaches.vectorise(dummy_txt, test_texts, approach="wemb_word", test=True)
        Xtrain, Xtest = zero_pad_vectors(Xtrain), zero_pad_vectors(Xtest)

        X_train_actual, X_val, y_train_actual, y_val = train_test_split(Xtrain, Ytrain, test_size=0.33, random_state=0)        
        
        all_results = dict()
        best_val_loss_overall = 1000000 # keeping track of best loss
        
        best_eval_score = -1.0
        # Now, training for all combinations of different parameters
        for batch in PARAMS_bilstm['batch_size']:
            for lr in PARAMS_bilstm['lr']:
                for dout in PARAMS_bilstm['dropout']:
                    for units1 in PARAMS_bilstm['num_units_1']:
                        for units2 in PARAMS_bilstm['num_units_2']:

                            outfile = str(batch)+'_'+str(lr)+'_'+str(dout)+'_'+str(units1)+'_'+str(units2)

                            # Define model
                            model = Sequential()
                            model.add(Masking(mask_value=0., input_shape=(PARAMS_bilstm['num_timesteps'], 300)))
                            model.add(Bidirectional(LSTM(units1, return_sequences=True)))
                            model.add(Dropout(dout))
                            model.add(Bidirectional(LSTM(units2, return_sequences=False)))
                            model.add(Dropout(dout))
                            model.add(Dense(3, activation='softmax'))

                            # Early stopping if performance does not increase; set optimizer & loss
                            es = EarlyStopping(monitor='val_loss', mode='min', patience=PARAMS_bilstm['early_stop_patience'])
                            model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['categorical_accuracy'])

                            history = model.fit(X_train_actual, y_train_actual, validation_data=(X_val, y_val), verbose=0, shuffle=True,
                                                callbacks=[es], epochs=PARAMS_bilstm['epochs'], batch_size=batch)
                            
                            
                            # f-score on val set:
                            val_preds = model.predict(X_val).argmax(axis=1)
                            val_preds = convert_categorical_to_labels(val_preds)
                            y_val_converted = convert_categorical_to_labels(y_val.argmax(axis=1))
                            val_score = f1_score(y_val_converted, val_preds, average='macro')
                            
                            # test set:
                            idx_preds = model.predict(Xtest).argmax(axis=1)
                            preds = convert_categorical_to_labels(idx_preds)
                            eval_score = f1_score(test_labels, preds, average='macro')
                            
                            best_loss = np.min(history.history['val_loss']) #best val loss
                            n_epoch = len(history.history['val_loss'])
                            all_results[outfile] = [eval_score, best_loss, n_epoch, history.history['val_loss']]
                            pickle.dump(all_results, open(FOLDER_models+'_log_bilstm.p', 'wb'))
                            
                            if val_score>best_eval_score:
                                best_eval_score = val_score
                                data_handler.save_results(model, test_tl_ids, test_pids, test_labels, preds, "wemb", test_fold, use_three_labels, 'bilstm_post')
                            
                            print(test_fold, '\t', batch, '\t', lr, '\t', dout, '\t', units1, '\t', units2, '\t', 
                                    Counter(preds), '\t', n_epoch, '\t', best_loss, '\t', val_score, '\t', eval_score)
                                