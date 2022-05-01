import numpy as np
from collections import Counter

def exact_match_func(pred, truth):
    extra_truth = []
    for t in truth:
        extra_truth.append(list(filter(lambda x: len(x) >= 3, t)))
    truth += extra_truth
    for t in truth:
        if pred == t:
            return 1
    return 0

def f1_score_func(pred, truth):
    extra_truth = []
    for t in truth:
        extra_truth.append(list(filter(lambda x: len(x) >= 3, t)))
    truth += extra_truth
    f1List = []
    for t in truth:
        common = Counter(pred) & Counter(t)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        if len(list(filter(lambda x: len(x) >= 3, pred))) != 0:
            pred = list(filter(lambda x: len(x) >= 3, pred))
        precision = 1.0 * num_same / len(pred)
        recall = 1.0 * num_same / len(t)
        f1List.append((2 * precision * recall) / (precision + recall))
    f1 = max(f1List)
    return f1

def evaluate(valid_df, predictions):
    ground_truth = {}
    for i, f in valid_df.iterrows():
        s, e = f.loc['label_idx']
        index_list = np.arange(s, e+1)
        answer_tokens = [f.loc['Small_Words'][i] for i in index_list]
        try:
            ground_truth[f.loc['id']].append(answer_tokens)
        except:
            ground_truth[f.loc['id']] = [answer_tokens]
    
    total = len(predictions.keys())
    em, f1 = 0, 0
    not_in_gt_indices = []
    for idx in predictions.keys():
        if idx not in ground_truth:
            not_in_gt_indices.append(idx)
        else:  
            em += exact_match_func(predictions[idx], ground_truth[idx])
            f1 += f1_score_func(predictions[idx], ground_truth[idx])
            
    
    exact_match = 100.0 * em / total
    f1_score = 100.0 * f1 / total
    return exact_match, f1_score

def epoch_time(start_time, end_time):
    
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

