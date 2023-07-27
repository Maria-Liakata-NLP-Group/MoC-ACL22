import numpy as np
import pandas as pd
import os, pickle, torch
import tensorflow as tf
from tensorflow import keras
from _utils import FOLD_to_TIMELINE, FOLDER_models, FOLDER_results


'''
This file implements some helpful functions for all of our models:
extracting the timelines of a particular (or many) fold(s), storing
the trained models/results and converting a five-label task to a 
3-label one.
'''


def get_timelines_for_fold(fold):
    '''
    Returns lists of different fields of all timelines IN the specified fold.
    Input:
        - fold (int): the fold we want to retrieve the timelines from
    Output (lists of posts):
        - timeline_ids: one tl_id per post
        - post_ids: the post_ids
        - texts: the text of each post
        - labels: the label of each post (5 possible labels)
    '''
    timelines_tsv = FOLD_to_TIMELINE[fold]
    timeline_ids, post_ids, texts, labels = [], [], [], []
    for tsv in timelines_tsv:
        df = pd.read_csv(tsv, sep='\t')
        if '374448_217' in tsv: #manually found (post 5723227 was not incorporated for some reason)
            df = pd.read_csv(tsv, sep='\t', quotechar='\'')
        pstid, txt, lbl = df.postid.values, df.content.values, df.label.values
        for i in range(len(pstid)):
            timeline_ids.append(tsv.split('/')[-1][:-4])
            post_ids.append(pstid[i])
            texts.append(str(txt[i]))
            labels.append(lbl[i])
    return timeline_ids, post_ids, texts, np.array(labels)


def get_timelines_except_for_fold(fold):
    '''
    Returns lists of different fields of all timelines EXCEPT FOR the specified fold.
    Input:
        - fold (int): the fold we want to avoid retrieving the timelines from
    Output (lists of posts):
        - timeline_ids: one tl_id per post
        - post_ids: the post_ids
        - texts: the text of each post
        - labels: the label of each post (5 possible labels)
    '''
    timeline_ids, post_ids, texts, labels = [], [], [], []
    for f in range(len(FOLD_to_TIMELINE)):
        if f!=fold:
            tlids, pstid, txt, lbl = get_timelines_for_fold(f)
            for i in range(len(pstid)):
                timeline_ids.append(tlids[i])
                post_ids.append(pstid[i])
                texts.append(str(txt[i]))
                labels.append(lbl[i])
    return timeline_ids, post_ids, texts, np.array(labels)


def get_three_labels(train_labels, test_labels):
    '''
    Replaces our ground truth labels: IEP with IE & ISB with IS.
    '''
    test_labels[test_labels=='ISB'] = 'IS'
    test_labels[test_labels=='IEP'] = 'IE'
    train_labels[train_labels=='ISB'] = 'IS'
    train_labels[train_labels=='IEP'] = 'IE'
    return train_labels, test_labels


def save_results(model, tl_ids, pids, labels, preds, feature_type, test_fold, use_three_labels, model_name):
    '''
    Stores the results and the trained model in the specified folders (check _utils.py).
    Input:
        - model:            any pickle-able object (presumably, the trained model)
        - tl_ids:           list of timeline_ids (one per post) on the test set
        - pids:             list of post ids (one per post) on the test set
        - labels:           list of actual labels (one per post) on the test set
        - preds:            list of predicted labels (one per post) on the test set
        - feature_type:     string (tfidf, wemb, bert)
        - test_fold:        the fold number (0-4, if working with five folds - as defined in _utils.NUM_FOLDS)
        - use_three_labels: True/False, depending on whether we performed a 3- or 5-label classification task
        - model_name:       the name of the model (logreg, bilstm, etc.) as string
    '''

    results = {'tl_ids':tl_ids, 'pids':pids, 'actual':labels, 'predicted':preds}

    pickle.dump(results, open(FOLDER_results+model_name+'_'+feature_type+'_'+str(test_fold)+'_threeLabels'+str(use_three_labels)+'.p', 'wb'))
    if model_name[:4]=='bert':
        output_model_file = FOLDER_models+model_name+'_model_'+str(test_fold)+'.bin'
        output_vocab_file = FOLDER_models+model_name+'_vocab_'+str(test_fold)+'.bin'
        model_to_save = model[0]
        tokenizer = model[1]
        torch.save(model_to_save, output_model_file)
        tokenizer.save_vocabulary(output_vocab_file)
    elif (model_name not in ['bilstm', 'bilstm_timeline', 'bilstm_post']) & (model_name[:10]!='lstm_bert_') & (model_name[:14] not in ['bilstm_feature', 'ffnnnn_feature']):
        pickle.dump(model, open(FOLDER_models+model_name+'_'+feature_type+'_'+str(test_fold)+'_threeLabels'+str(use_three_labels)+'.p', 'wb'))
    else:
        outfolder = FOLDER_models+model_name+'_'+feature_type+'_'+str(test_fold)+'_threeLabels'+str(use_three_labels)+'/'
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        model.save(outfolder)



class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(DOUT)
        if use_three_labels:
            self.l3 = torch.nn.Linear(768, 3)
        else:
            self.l3 = torch.nn.Linear(768, 5)
            
    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output