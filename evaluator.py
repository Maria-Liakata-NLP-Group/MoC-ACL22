import pickle
import numpy as np
import pandas as pd
from _utils import FOLDER_results, NUM_folds, FOLDER_dataset


def read_results(model_name, representation, fold, use_three_labels=True):
    extension = '_threeLabelsFalse.p'
    if use_three_labels:
        extension = '_threeLabelsTrue.p'
    return pickle.load(open(FOLDER_results+model_name+'_'+representation+'_'+str(fold)+extension, 'rb'))
   

def get_fscore_post(model_name, representation, cls='IS', use_three_labels=True):
    from sklearn.metrics import f1_score, recall_score, precision_score

    all_preds, all_actuals = [], []
    for fold in range(NUM_folds):
        data = read_results(model_name, representation, fold, use_three_labels)
        fold_preds, fold_actuals = data['predicted'], data['actual']
        all_preds.extend(fold_preds)
        all_actuals.extend(fold_actuals)
    p = np.round(precision_score(all_actuals, all_preds, labels=cls, average='macro'),10)
    r = np.round(recall_score(all_actuals, all_preds, labels=cls, average='macro'),10)
    f1 = np.round(f1_score(all_actuals, all_preds, labels=cls, average='macro'),10)
    #print(model_name, '\t', representation, '\t', p,r,f1)
    #print(model_name, representation, f1_score(all_actuals, all_preds, average='macro'))
    return p,r,f1


def get_fscore_post_random(model_name, representation, use_three_labels=True):
    from sklearn.metrics import f1_score, recall_score, precision_score
    from collections import Counter

    fscores_0, precs_0, recs_0 = [], [], []
    fscores_IE, precs_IE, recs_IE = [], [], []
    fscores_IS, precs_IS, recs_IS = [], [], []
    fscores, precs, recs = [], [], []

    all_actuals = []
    for fold in range(NUM_folds):
        data = read_results(model_name, representation, fold, use_three_labels)
        fold_actuals = np.array(data['actual'])
        all_actuals.extend(fold_actuals)
    
    cntr = Counter(all_actuals)
    num_positions = 10000
    prob_0 = int(num_positions*cntr['0']/len(all_actuals))
    prob_is = int(num_positions*cntr['IS']/len(all_actuals))
    prob_ie = int(num_positions*cntr['IE']/len(all_actuals))
    wheel = ['0' for position in range(prob_0)]
    wheel.extend(['IS' for position in range(prob_is)])
    wheel.extend(['IE' for position in range(prob_ie)])
    num_positions = len(wheel)
    
    for trial in range(10000):
        all_preds = []
        for rr in all_actuals:
            ran = np.random.randint(num_positions)
            all_preds.append(wheel[ran])

        precs_0.append(precision_score(all_actuals, all_preds, labels=['0'], average='macro'))
        recs_0.append(recall_score(all_actuals, all_preds, labels=['0'], average='macro'))
        fscores_0.append(f1_score(all_actuals, all_preds, labels=['0'], average='macro'))

        precs_IS.append(precision_score(all_actuals, all_preds, labels=['IS'], average='macro'))
        recs_IS.append(recall_score(all_actuals, all_preds, labels=['IS'], average='macro'))
        fscores_IS.append(f1_score(all_actuals, all_preds, labels=['IS'], average='macro'))

        precs_IE.append(precision_score(all_actuals, all_preds, labels=['IE'], average='macro'))
        recs_IE.append(recall_score(all_actuals, all_preds, labels=['IE'], average='macro'))
        fscores_IE.append(f1_score(all_actuals, all_preds, labels=['IE'], average='macro'))

        precs.append(precision_score(all_actuals, all_preds, average='macro'))
        recs.append(recall_score(all_actuals, all_preds, average='macro'))
        fscores.append(f1_score(all_actuals, all_preds, average='macro'))

        print(trial, np.average(precs_IS), np.average(precs_IE))
    print(np.average(precs_IS), np.average(recs_IS), np.average(fscores_IS)) 
    print(np.average(precs_IE), np.average(recs_IE), np.average(fscores_IE)) 
    print(np.average(precs_0), np.average(recs_0), np.average(fscores_0)) 
    print(np.average(precs), np.average(recs), np.average(fscores)) 


def get_fscore_timeline(model_name, representation, cls='IS', use_three_labels=True, window=1):
    all_recalls, all_precisions, all_fscores = [], [], [] #per timeline
    for fold in range(NUM_folds):
        data = read_results(model_name, representation, fold, use_three_labels)
        fold_preds, fold_actuals, fold_tls = np.array(data['predicted']), np.array(data['actual']), np.array(data['tl_ids'])
        for tlid in set(fold_tls):
            tl_preds, tl_actuals = fold_preds[np.where(fold_tls==tlid)], fold_actuals[np.where(fold_tls==tlid)]      
            s_prec, s_rec, s_sup = get_precision_recall_timeline(tl_preds, tl_actuals, window, cls)     
            s_fscore = np.nan
            if s_prec+s_rec>0:
                s_fscore = (2*s_prec*s_rec)/(s_prec+s_rec)
            elif s_prec+s_rec==0:
                s_fscore = 0.0
            all_recalls.append(s_rec)
            all_precisions.append(s_prec)
            all_fscores.append(s_fscore)
    p = np.nanmean(all_precisions)
    r = np.nanmean(all_recalls)
    f1 = np.nanmean(all_fscores) # miserable
    #print(model_name, '\t', representation, r, p, f1)
    return p, r, f1


def get_precision_recall_timeline(predicted_labels, actual_labels, window, label='IS'):
    '''
    Given the lists of predicted and actual labels, the label to calculate the metrics for and
    the window to use (allowing +-window predictions to be considered as accurate), it returns:
        (a) the precision using that window for the specified label
        (b) the recall -//-
        (c) the support (number of actual labels that we had to predict)
    '''
    # Find the indices of the specified predicted and actual label
    preds = [i for i in range(len(predicted_labels)) if predicted_labels[i]==label]
    actuals = [i for i in range(len(actual_labels)) if actual_labels[i]==label]

    if label=='IS':
        prob = int(0.05*len(actual_labels))
    elif label=='IE':
        prob = int(0.1*len(actual_labels))
    else:
        prob = int(0.85*len(actual_labels))
    #preds = [np.random.randint(len(predicted_labels)) for i in range(prob)]
    #if label=='IS':
    #    print(actuals)

    #print(len(preds), '\t', len(actuals), '\t', len(actual_labels))
    # Check if we can proceed or not
    support = len(actuals)
    if len(actuals)==0: # cannot divide by zero (Recall is undefined)
        recall, precision = np.nan, np.nan
        if len(preds)>0:
            precision = 0.0
    elif len(preds)==0: # cannot divide by zero (Precision is undefined, but Recall is 0)
        precision = np.nan
        recall = 0.0
    else:
        already_used = []
        for l in actuals: 
            for p in preds: 
                if (np.abs(l-p)<=window) & (p not in already_used): 
                    already_used.append(p) 
                    break 
        precision = len(set(already_used))/len(preds)
        recall = len(set(already_used))/len(actuals)
    return precision, recall, support



def get_coverage_fscore(model_name, representation, use_three_labels=True):
    rec_switch = get_coverage_recall(model_name, representation, 'IS', use_three_labels)
    rec_escala = get_coverage_recall(model_name, representation, 'IE', use_three_labels)
    rec_normal = get_coverage_recall(model_name, representation, '0', use_three_labels)

    prec_switch = get_coverage_precision(model_name, representation, 'IS', use_three_labels)
    prec_escala = get_coverage_precision(model_name, representation, 'IE', use_three_labels)
    prec_normal = get_coverage_precision(model_name, representation, '0', use_three_labels)

    return (prec_normal+prec_escala+prec_switch)/3.0, (rec_switch+rec_escala+rec_normal)/3.0



def get_right_order(infile, postids, actual, preds):
    df = pd.read_csv(infile, sep='\t')
    postids_right_order = df.postid.values
    new_actual, new_preds = [], []
    for ordered_postid in postids_right_order:
        idx = np.where(postids==ordered_postid)[0][0]
        new_actual.append(actual[idx])
        new_preds.append(preds[idx])
    return np.array(new_actual), np.array(new_preds)



def get_coverage_recall(model_name, representation, cls='IS', specified_window=1, use_three_labels=True): #adtsakal: also, precision
    all_coverages = []
    for fold in range(NUM_folds):
        data = read_results(model_name, representation, fold, use_three_labels)
        fold_preds, fold_actuals, fold_tls = np.array(data['predicted']), np.array(data['actual']), np.array(data['tl_ids'])
        fold_postids = np.array(data['pids'])
        for tlid in set(fold_tls):
            tl_preds, tl_actuals = fold_preds[np.where(fold_tls==tlid)], fold_actuals[np.where(fold_tls==tlid)]
            tl_postids = fold_postids[np.where(fold_tls==tlid)]
            
            '''
            ran = np.random.randint(18702, size=len(tl_actuals))
            tl_preds = np.array(['AA' for _ in tl_actuals])
            tl_preds[ran<885] = 'IS'
            tl_preds[(ran>=885) & (ran<2018+885)] = 'IE'
            tl_preds[ran>=2018+885] = '0'
            '''
            
            #need to make sure they are in the correct order!
            stri = FOLDER_dataset+str(fold)+'/'+str(tlid)+'.tsv'
            tl_actuals, tl_preds = get_right_order(stri, tl_postids, tl_actuals, tl_preds)

            preds_regions, preds_regions_neg = extract_regions(tl_preds, cls)
            actual_regions_raw, actual_regions_neg_raw = extract_regions(tl_actuals, cls)
            actual_regions = [r for r in actual_regions_raw if len(r)>=specified_window]
            actual_regions_neg = [r for r in actual_regions_neg_raw if len(r)>=specified_window]

            

            #we start processing a single timeline
            total_sum, denom = 0.0, 0.0 # timeline basis

            #First for the positive cases:
            if len(actual_regions)>0:

                for region in actual_regions: # For each actual region within the timeline
                    ac = set(region)
                    Orrs = [] #calculate per region
                    max_cov_for_region = 0.0 #calculated per region

                    #Find the maximum ORR for this actual region:
                    if len(preds_regions)>0: 

                        for predicted_region in preds_regions: 
                            pr = set(predicted_region)
                            Orrs.append(len(ac.intersection(pr))/len(ac.union(pr))) # Intersection over Union
                        max_cov_for_region = np.max(Orrs)
                    
                    #Now multiply it by the length of the region
                    total_sum = total_sum + (len(ac)*max_cov_for_region)
                    denom += len(ac)            
                all_coverages.append(total_sum/denom)
    #print(np.average(all_coverages))
    return np.average(all_coverages)


def get_coverage_precision(model_name, representation, cls='IS', specified_window=1, use_three_labels=True): #adtsakal: also, precision
    all_coverages = []
    for fold in range(NUM_folds):
        data = read_results(model_name, representation, fold, use_three_labels)
        fold_preds, fold_actuals, fold_tls = np.array(data['predicted']), np.array(data['actual']), np.array(data['tl_ids'])
        fold_postids = np.array(data['pids'])
        for tlid in set(fold_tls):
            tl_preds, tl_actuals = fold_preds[np.where(fold_tls==tlid)], fold_actuals[np.where(fold_tls==tlid)]
            tl_postids = fold_postids[np.where(fold_tls==tlid)]
            
            
            '''
            ran = np.random.randint(18702, size=len(tl_actuals))
            tl_preds = np.array(['AA' for _ in tl_actuals])
            tl_preds[ran<885] = 'IS'
            tl_preds[(ran>=885) & (ran<2018+885)] = 'IE'
            tl_preds[ran>=2018+885] = '0'
            '''

            #need to make sure they are in the correct order!
            stri = FOLDER_dataset+str(fold)+'/'+str(tlid)+'.tsv'
            tl_actuals, tl_preds = get_right_order(stri, tl_postids, tl_actuals, tl_preds)

            preds_regions, preds_regions_neg = extract_regions(tl_preds, cls)
            actual_regions_raw, actual_regions_neg_raw = extract_regions(tl_actuals, cls)
            actual_regions = [r for r in actual_regions_raw if len(r)>=specified_window]
            actual_regions_neg = [r for r in actual_regions_neg_raw if len(r)>=specified_window]

            #we start processing a single timeline
            total_sum, denom = 0.0, 0.0 # big sum and 1/N, respectively

            if len(preds_regions)>0:

                for region in preds_regions: # For each predicted region within the timeline
                    ac = set(region)
                    Orrs = []
                    max_cov_for_region = 0.0

                    #Find the maximum ORR for this predicted region:
                    if len(actual_regions)>0: 

                        for predicted_region in actual_regions: 
                            pr = set(predicted_region)
                            Orrs.append(len(ac.intersection(pr))/len(ac.union(pr))) # Intersection over Union
                        max_cov_for_region = np.max(Orrs)
                    
                    #Now multiply it by 
                    total_sum = total_sum + (len(ac)*max_cov_for_region)
                    denom += len(ac)            
                all_coverages.append(total_sum/denom)
    #print(np.average(all_coverages))
    return np.average(all_coverages)


            
def extract_regions(vals, cls):
    #Convert labels to boolean based on the class we are looking for:
    vals_boolean = np.zeros(len(vals))
    vals_boolean[vals==cls] = 1

    #Find the indices of the positive cases
    indices = np.where(vals_boolean==1)[0]
    neg_indices = np.where(vals_boolean==0)[0]
    
    actual_regions = get_regions(indices)
    actual_regions_neg = get_regions(neg_indices)
    return actual_regions, actual_regions_neg


def get_regions(indices):
    #Start scanning
    actual_regions = []         
    if len(indices)>0:
        current_set = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i]-indices[i-1]==1: #if they are consecutive
                current_set.append(indices[i])
            else:
                actual_regions.append([v for v in current_set])
                current_set = [indices[i]]
        actual_regions.append(current_set)
    return actual_regions