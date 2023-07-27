import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import data_handler, nlp.vectorising_approaches
from _utils import NUM_folds, PARAMS_rf


## nearest neighbour
def nn_fsd(post_vector, corpus_vector):
    max_sim = 0.0
    for vector in corpus_vector:
        sim = cosine_similarity(post_vector.reshape(1, -1),vector.reshape(1, -1))[0][0]
        if sim>max_sim:
            max_sim = sim
    return max_sim

## centroid similarity
def centroid_fsd(post_vector, corpus_vectors):
    centroid = np.average(corpus_vectors, axis=0).reshape(1, -1)
    sim = cosine_similarity(post_vector,centroid)[0][0]
    return sim


def get_post_indices_of_timeline(tlid, timelines):
    return np.where(np.array(timelines)==tlid)[0]


def get_fsd_scores(approach, tlids, postids, X, labels, window):
    '''Returns a dictionary, with one key per timeline. The values of the
    dictionary are:
        a) the post ids corresponding to this timeline
        b) the original feature representation of each post
        c) the score that resulted by applying the 'approach' approach 
        d) the label of each post'''
    diction = dict()
    windows = [i+1 for i in range(window)]
    for tlid in set(tlids): # re-arranging bits on timeline level
        idx = get_post_indices_of_timeline(tlid, tlids)
        tl_X, tl_postids, tl_Y, timeline_scores = X[idx], postids[idx], labels[idx], []
        timeline_scores.append(np.array([0 for i in (range(window+1))]))
        for i in range(1, len(tl_X)): # for each post of the timeline
            post_scores = []
            
            #First use the full timeline
            if approach=='nn':
                score = nn_fsd(tl_X[i], tl_X[:i])
            elif approach=='centroid':
                score = centroid_fsd(tl_X[i].reshape(1,-1), tl_X[:i])
            post_scores.append(score)

            #Now use the windows
            for win in windows:
                if approach=='nn':
                    score = nn_fsd(tl_X[i], tl_X[max(0,i-win):i])
                elif approach=='centroid':
                    score = centroid_fsd(tl_X[i].reshape(1,-1), tl_X[max(0,i-win):i])
                post_scores.append(score)
            post_scores = np.array(post_scores)
            timeline_scores.append(post_scores)
        timeline_scores = np.array(timeline_scores)
        diction[tlid] = [tl_postids, tl_X, timeline_scores, tl_Y]
    return diction


def run_fsd_method(approach='nn', representation='sentence-bert', use_three_labels=True, window=10):
    _post = "hi there my name is"

    all_pids, all_feats = [], []
    for fold in range(NUM_folds): # for each (test) fold
        print('Training for fold#'+str(fold))

        # Getting the train/test data: timelines_ids, posts_id, texts, labels (posts must be (and are) ordered time-wise)
        test_tldids, test_postids, test_texts, test_labels = data_handler.get_timelines_for_fold(fold)
        train_tldids, train_postids, train_texts, train_labels = data_handler.get_timelines_except_for_fold(fold)
        
        # Converting to a 3-label prediction task
        if use_three_labels:
            train_labels, test_labels = data_handler.get_three_labels(train_labels, test_labels)
       
       # Extracting post representations with the specified ('representation') mode:
        _, trainX, __ = nlp.vectorising_approaches.vectorise(_post, train_texts, approach=representation, train=True)
        _, testX = nlp.vectorising_approaches.vectorise(_post, test_texts, approach=representation, train=False)

        # Extracting the scores and populating the dicts (main part of the FSD methods)
        training_dict = get_fsd_scores(approach, train_tldids, np.array(train_postids), trainX, train_labels, window)
        testing_dict = get_fsd_scores(approach, test_tldids, np.array(test_postids), testX, test_labels, window)

        # Re-arranging the dicts so that they are ready for LogReg
        Xtrain, Ytrain = [], []
        for tlid in training_dict.keys():
            tl_postids, _, meta_feats, labels = training_dict[tlid]
            for i in range(len(tl_postids)):
                Xtrain.append(meta_feats[i])
                Ytrain.append(labels[i])
        Xtrain = np.array(Xtrain)#.reshape(-1,1)

        tlids, postids, Xtest, Ytest = [], [], [], []
        for tlid in testing_dict.keys():
            tl_postids, _, meta_feats, labels = testing_dict[tlid]
            for i in range(len(tl_postids)):
                tlids.append(tlid)
                postids.append(tl_postids[i])
                Xtest.append(meta_feats[i])
                Ytest.append(labels[i])
        Xtest = np.array(Xtest)#.reshape(-1,1)

        for p in range(len(Xtest)):
            all_pids.append(postids[p])
            all_feats.append(Xtest[p])
    return all_pids, all_feats