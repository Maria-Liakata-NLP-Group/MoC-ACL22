import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import data_handler, nlp.vectorising_approaches
from _utils import NUM_folds, PARAMS_rf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

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


def orthogonal_procrustes(previous, next):
    return None

def get_post_indices_of_timeline(tlid, timelines):
    return np.where(np.array(timelines)==tlid)[0]


def generate_scd_examples_op(tlids, postids, X, labels, window):
    '''Returns a dictionary, with one key per timeline. The values of the
    dictionary are:
        a) the post ids corresponding to this timeline
        b) the original feature representation of each post
        c) the score that resulted by applying the 'approach' approach 
        d) the label of each post'''
    diction = dict()
    for tlid in set(tlids): # re-arranging bits on timeline level
        idx = get_post_indices_of_timeline(tlid, tlids)
        tl_X, tl_postids, tl_Y = X[idx], postids[idx], labels[idx]

        mypostids, before, after = [], [], []
        for i in range(1,len(tl_X)):
            before.append(tl_X[i-1])
            after.append(tl_X[i])
            mypostids.append(tl_postids[i])

        diction[tlid] = [mypostids, before, after]
    return diction


def generate_scd_examples_fp(tlids, postids, X, labels, window):
    '''Returns a dictionary, with one key per timeline. The values of the
    dictionary are:
        a) the post ids corresponding to this timeline
        b) the original feature representation of each post
        c) the score that resulted by applying the 'approach' approach 
        d) the label of each post'''
    diction = dict()
    for tlid in set(tlids): # re-arranging bits on timeline level
        idx = get_post_indices_of_timeline(tlid, tlids)
        tl_X, tl_postids, tl_Y = np.array(X[idx]), postids[idx], labels[idx]
        
        dims = len(tl_X[0])

        mypostids, before, after = [], [], []
        for i in range(1,len(tl_X)):
            if i==1:
                vec1 = np.array([-1 for kk in range(dims)]).reshape(1,-1)#np.zeros((2,dims))
                vec2 = np.array([-1 for kk in range(dims)]).reshape(1,-1)#np.zeros((2,dims))
                vec3 = tl_X[0:1]
                vec4 = np.concatenate((vec1, vec2, vec3), axis=0)
                before.append(vec4)
            elif i==2:
                vec1 = np.array([-1 for kk in range(dims)]).reshape(1,-1)#np.zeros((1,dims))
                vec2 = tl_X[0:2]
                vec3 = np.concatenate((vec1, vec2), axis=0)
                before.append(vec3)
            else:
                before.append(tl_X[i-3:i])
            after.append(tl_X[i])
            mypostids.append(tl_postids[i])

        diction[tlid] = [mypostids, before, after]
    return diction


def run_scd_method(approach='op', representation='sentence-bert', use_three_labels=True, window=10):
    _post = "hi there my name is"

    fold = 6

    # Getting the train/test data: timelines_ids, posts_id, texts, labels (posts must be (and are) ordered time-wise)
    tldids, postids, texts, labels = data_handler.get_timelines_except_for_fold(fold)

    # Extracting post representations with the specified ('representation') mode:
    _, trainX, __ = nlp.vectorising_approaches.vectorise(_post, texts, approach=representation, train=True)

    # Extracting the [be]
    if approach=='op':
        training_dict = generate_scd_examples_op(tldids, np.array(postids), trainX, labels, window)

        train_tlids, train_postids, train_before, train_after = [], [], [], []
        for key in training_dict.keys():
            tl_postids, bef, aft = training_dict[key]
            for i in range(len(tl_postids)):
                train_tlids.append(key)
                train_postids.append(tl_postids[i])
                train_before.append(bef[i])
                train_after.append(aft[i])
        scores, diffs = procrustes_all_together(train_before, train_after)
        for postid in postids:
            if postid not in train_postids:
                train_postids.append(postid)
                scores.append(1.0)
                diffs.append(np.zeros(len(train_before[0])))
        return train_postids, scores, diffs
    
    elif approach=='fp':
        all_pids, all_vals, all_vectors = [], [], []
        all_data = generate_scd_examples_fp(tldids, np.array(postids), trainX, labels, window)
        for fold in range(5):
            train_tldids, train_postids, _, labels = data_handler.get_timelines_except_for_fold(fold)
            test_tldids, test_postids, _, labels = data_handler.get_timelines_for_fold(fold)

            trainX, trainY = [], []
            for tlid in all_data.keys():
                if tlid in train_tldids:
                    _postids, _pre, _post = all_data[tlid]
                    trainX.extend(_pre)
                    trainY.extend(_post)
            trainX, trainY = np.array(trainX), np.array(trainY)

            test_pids, testX, testY = [], [], []
            for tlid in all_data.keys():
                if tlid in test_tldids:
                    _postids, _pre, _post = all_data[tlid]
                    test_pids.extend(_postids)
                    testX.extend(_pre)
                    testY.extend(_post)
            testX, testY = np.array(testX), np.array(testY)

            print(fold, '\t', trainX.shape, '\t', trainY.shape)

            
            model = Sequential()
            model.add(Masking(mask_value=-1, input_shape=(trainX.shape[1], trainX.shape[2])))
            model.add(Bidirectional(LSTM(128, return_sequences=True)))
            model.add(Dropout(0.25))
            model.add(Bidirectional(LSTM(128, return_sequences=False)))
            model.add(Dropout(0.25))
            model.add(Dense(trainY.shape[1]))

            print(fold)
            # Early stopping if performance does not increase; set optimizer & loss
            es = EarlyStopping(monitor='val_loss', mode='min', patience=5)
            model.compile(loss='cosine_similarity', optimizer=Adam(learning_rate=0.0001), metrics=['cosine_similarity'])

            history = model.fit(trainX, trainY, validation_split=0.33, verbose=2, shuffle=True,callbacks=[es], epochs=100, batch_size=64)

            preds = model.predict(testX)
            for i in range(len(preds)):
                all_pids.append(test_pids[i])
                all_vals.append(cosine_similarity(preds[i].reshape(1,-1), testY[i].reshape(1,-1))[0][0])
                all_vectors.append(preds[i]-testY[i])
            #add the first posts
            for pid in test_postids:
                if pid not in all_pids:
                    all_pids.append(pid)
                    all_vals.append(1.0)
                    all_vectors.append(np.zeros(preds.shape[1]))
            
        return all_pids, all_vals, all_vectors
            

def procrustes(X, Y):
    #bringing everything in the same space
    muX, muY = np.average(X, axis=0), np.average(Y, axis=0)
    X0, Y0 = X - muX, Y - muY
    normX, normY = np.linalg.norm(X0), np.linalg.norm(Y0)
    X0/=normX
    Y0/=normY
    #orthogonal procrustes
    u, w, vt = np.linalg.svd(Y0.T.dot(X0).T)
    R = u.dot(vt)
    s = w.sum()
    mtx = np.dot(Y0, R.T)*s
    err = np.sum(np.square(X0 - mtx)) #not really needed
    return  err, X0, mtx, {'muX': muX, 'muY': muY, 'normX':normX, 'normY':normY, 'R':R, 's':s}

def procrustes_all_together(x_all, y_all):
    _, x_test_all, y_test_all, __ = procrustes(x_all, y_all) #getting the transformed matrices

    scores, diffs = [], []
    for w in range(len(x_all)):
        cosSim = cosine_similarity(x_test_all[w].reshape((1,-1)),y_test_all[w].reshape((1,-1)))
        diffs.append(x_test_all[w]-y_test_all[w])
        scores.append(cosSim[0, 0])
    return scores, diffs