import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import nlp.vectorising_approaches, data_handler
from _utils import FOLD_to_TIMELINE, PARAMS_rf

np.random.seed(0) 



'''
Code for training Random Forest on the post-level (i.e., assuming completely independent posts
regardless of the timeline they come from), using five-fold CV (posts from 100 timelines per fold):

                                train_model('tfidf')

will train the model for each (test) fold sequentially, performing an additional 5-fold CV for finding
the optimal regularisation value C (check _utils.py for tested values). Model and results (predicitons)  
are stored in FOLDER_models and FOLDER_results, respectively.

                    Execution time (approx.) for all folds: 75'
'''


def train_rf_model(feature_type='tfidf', use_three_labels=True):
    for test_fold in range(len(FOLD_to_TIMELINE)): # for each (test) fold
        print('Training RandomForest on', feature_type,'\tFold: '+str(test_fold+1)+'/'+str(len(FOLD_to_TIMELINE)))
        
        # Just getting the train/test data: timelines_ids, posts_id, texts, labels
        test_tl_ids, test_pids, test_texts, test_labels = data_handler.get_timelines_for_fold(test_fold)
        train_tl_ids, train_pids, train_texts, train_labels = data_handler.get_timelines_except_for_fold(test_fold)

        # Converting to a 3-label prediction task
        if use_three_labels:
            train_labels, test_labels = data_handler.get_three_labels(train_labels, test_labels)

        # Feature extraction
        _, Xtrain, vectoriser = nlp.vectorising_approaches.vectorise("my name is test", train_texts, approach=feature_type, train=True)
        _, Xtest = nlp.vectorising_approaches.vectorise("my name is test", test_texts, test=True, approach=feature_type, trained_vectorizer=vectoriser)

        print(Xtrain.shape, '\t', Xtrain[0].shape)

        best_val_score = -1.0
        # Defining/training the model 
        for param in PARAMS_rf['n_estimators']:
            clf = RandomForestClassifier(random_state=0)
            X_train_actual, X_val, y_train_actual, y_val = train_test_split(Xtrain, train_labels, test_size=0.33, random_state=0)
            clf.fit(X_train_actual, y_train_actual)
            val_preds = clf.predict(X_val)
            val_score = f1_score(y_val, val_preds, average='macro')
            print(test_fold, '\t', param, '\t', val_score)
            if val_score>best_val_score:
                # Making the predictions and storing the results
                preds = clf.predict(Xtest)
                print('Best F-score (macro) on test set:', f1_score(test_labels, preds, average='macro'))
                data_handler.save_results(clf, test_tl_ids, test_pids, test_labels, preds, feature_type, test_fold, use_three_labels, 'rf')
                best_val_score = val_score