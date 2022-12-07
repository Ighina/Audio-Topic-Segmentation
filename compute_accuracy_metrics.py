# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 18:01:20 2022

@author: User
"""
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import f1_score

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import segeval
import pickle
import json
import os
import re

def get_boundaries(boundaries):
    tot_sents = 0
    masses = []
    for boundary in boundaries:
        tot_sents += 1
        if boundary:
            masses.append(tot_sents)
            tot_sents = 0
    return masses

def B_measure(boundaries, ground_truth):
    """
    Boundary edit distance-based methods for text segmentation evaluation (Fournier2013)
    """
    boundaries[-1] = 1
    ground_truth[-1] = 1
    h = get_boundaries(boundaries)
    t = get_boundaries(ground_truth)
    # value errors occur when there is no boundary in the reference segmentation
    # if len(t)<2:     
    cm = segeval.boundary_confusion_matrix(h, t, n_t = 10)
    b_precision = float(segeval.precision(cm, classification = 1))
    b_recall = float(segeval.recall(cm, classification = 1))
    try:
        b_f1 = 2*(b_precision*b_recall)/(b_precision+b_recall)
    except ZeroDivisionError:
        b_f1 = 0.0
    # b_f1 = segeval.fmeasure(cm, classification = 1)
    # else:
    #    b_precision = 0
    #    b_recall = 0
    #    b_f1 = 0
    # if len(t)<2:
    b = segeval.boundary_similarity(h, t, n_t = 10)
    # else:
    #    b = 0
    return float(b_precision), float(b_recall), float(b_f1), float(b)

def sig(x):
 return 1/(1 + np.exp(-x))


def bootstrap(data, samples = 10000):
  if isinstance(data, list):
    data = pd.DataFrame(data)
  boot = []
  for sample in range(samples):
    boot.append(data.sample(len(data), replace = True).mean()[0])
  return boot
    
    

data = 'bmat'
mask = True

if data=='nonnews':
    # files = os.listdir("data/non_news_2021-07-05/audio/")
    
    # for index, file in enumerate(files):
    #     files[index] = file[:-4]+'.npy'
    
    #files = [f for f in os.listdir("NonNews-BBC/NonNewsUniform1/x-vectors") if f[-6:]!="tifier"]
    #print(files)
    
    with open("NonNews-BBC/NonNews_split.json") as f:
        files = json.load(f)["test"]
    
    with open('NonNews-BBC/NonNewsUniform1/labs_dict.pkl', 'rb') as f:
        lab = pickle.load(f)
    
    prefix = os.path.join("NonNews-BBC", "NewExperiments")
    prefix_lab = os.path.join("NonNews-BBC", "NonNewsUniform1")
    # prefix = 'nonnews_adaptive'
    # lab_prefix = 'data/non_news_2021-07-05/audio\\'
    # suffix = '.mp3'

# elif data=='nonnews_1':
#     files = os.listdir("data/non_news_2021-07-05/audio/")
    
#     for index, file in enumerate(files):
#         files[index] = file[:-4]+'.npy'
    
#     with open('NonNewsUniform1/1_nonnews_labels/labs_dict.pkl', 'rb') as f:
#         lab = pickle.load(f)
#     prefix = 'NonNewsUniform1'
#     lab_prefix = 'data/non_news_2021-07-05/audio/'
#     suffix = '.mp3'

# elif data=='podcast_1':
#     files = os.listdir("data/AudioBBC/audio/")
    
#     for index, file in enumerate(files):
#         files[index] = file[:-4]+'.npy'
    
#     with open('PodcastUniform1/1_podcast_labels/labs_dict.pkl', 'rb') as f:
#         lab = pickle.load(f)
#     prefix = 'PodcastUniform1'
#     lab_prefix = 'data/AudioBBC/audio/'
#     suffix = '.mp3'
    
elif data=='radionews':
    # podcast_files_unfiltered = os.listdir("data/AudioBBC/audio")
    # files = []
    # for file in podcast_files_unfiltered:
    #     if not re.findall("(24580|25539|25684|26071|26214|26321|26427)", file):
    #         files.append(file)
    
    # for index, file in enumerate(files):
    #     files[index] = file[:-4]+'.npy'
    
    with open('RadioNews-BBC/RadioNewsUniform1/labs_dict.pkl', 'rb') as f:
        lab = pickle.load(f)
        
    with open("RadioNews-BBC/RadioNews_split.json") as f:
        files = json.load(f)["test"]
    
    prefix = os.path.join("RadioNews-BBC", "NewExperiments")
    prefix_lab = os.path.join("RadioNews-BBC", "RadioNewsUniform1")
    
    
    # prefix = 'podcast_adaptive'
    # lab_prefix = 'data/AudioBBC/audio\\'
    # suffix = '.mp3'

elif data=='bmat':
    # podcast_files_unfiltered = os.listdir("data/AudioBBC/audio")
    # files = []
    # for file in podcast_files_unfiltered:
    #     if not re.findall("(24580|25539|25684|26071|26214|26321|26427)", file):
    #         files.append(file)
    
    # for index, file in enumerate(files):
    #     files[index] = file[:-4]+'.npy'
    
    with open('OpenBMAT/BMAT_ATS1/labs_dict.pkl', 'rb') as f:
        lab = pickle.load(f)
        
    with open("OpenBMAT/BMAT_split.json") as f:
        files = json.load(f)["test"]
    
    prefix = os.path.join("OpenBMAT", "NewExperiments")
    prefix_lab = os.path.join("OpenBMAT", "BMAT_ATS1")
    
    
    # prefix = 'podcast_adaptive'
    # lab_prefix = 'data/AudioBBC/audio\\'
    # suffix = '.mp3'

# elif data=='BMAT-1':
#     files = os.listdir("data/OpenBMAT/BMAT-ATS")
    
#     for index, file in enumerate(files):
#         files[index] = file[:-4]+'.npy'
    
#     with open('BMATUniform1/1_BMAT_labels/labs_dict.pkl', 'rb') as f:
#         lab = pickle.load(f)
#     prefix = 'BMATUniform1'
#     lab_prefix = 'data/OpenBMAT/BMAT-ATS\\'
#     suffix = '.wav'
    
# elif data=='BMAT':
#     files = os.listdir("data/OpenBMAT/BMAT-ATS")
    
#     for index, file in enumerate(files):
#         files[index] = file[:-4]+'.npy'
    
#     with open('BMAT_adaptive/adaptive_BMAT_labels/labs_dict.pkl', 'rb') as f:
#         lab = pickle.load(f)
#     prefix = 'BMAT_adaptive'
#     lab_prefix = 'data/OpenBMAT/BMAT-ATS/'
#     suffix = '.wav'

df = {"Precision":[], "Precision Confidence":[], "Recall":[], "Recall Confidence":[], "F1":[], "F1 Confidence":[], "B-F1":[], "B-Precision":[], "B-Recall":[], "B-F1 Confidence":[], "B-Precision Confidence":[], "B-Recall Confidence":[], "embedding":[]}

all_scores_f1 = {}
all_scores_precision = {}
all_scores_recall = {}

all_scores_bf1 = {}
all_scores_bprecision = {}
all_scores_brecall = {}

op = True
for enc in ["x-vectors", "wav2vec/_mean_std", "wav2vec/_max", "wav2vec/_last", "wav2vec/_delta_gap", "openl3/_mean_std", "openl3/_max", "openl3/_last", "openl3/_delta_gap", "crepe/_mean_std", "crepe/_max", "crepe/_last", "crepe/_delta_gap", "mfcc", "prosodic"]:
    with open(os.path.join(prefix,"BiLSTM_bs10_" + enc,"all_scores.json")) as f:
        d = json.load(f)
        
    f1_scores = []
    recall_scores = []
    precision_scores = []
    bf1_scores = []
    bprecision_scores = []
    brecall_scores = []
    for k in files:
        #np.random.seed(1)
        lab_k = os.path.join(k[:-4])
        # new_list = []
        # if op:
        #     for index, l in enumerate(lab[lab_k]):
        #         if np.random.rand()>.7 and not l:
        #             pass
        #         else:
        #             new_list.append(l)
        
        #     if len(new_list)>len(d[k]): 
        #         lab[lab_k] = new_list[:-1]
        #     else:
        #         lab[lab_k] = new_list
        
        pred = (sig(np.array(d[k]).reshape(-1))>0.5)+0
        
        # In the podcast data there were few instances with inconsistency, mainly due to rounding errors in the embedding extraction step
        # if len(pred)>len(lab[lab_k]):
        #     pred = pred[:len(lab[lab_k])]
        # elif len(pred)<len(lab[lab_k]):
        #     lab[lab_k] = lab[lab_k][:len(pred)]
        f1_scores.append(f1_score(lab[lab_k][:-1], pred[:-1]))
            
        recall_scores.append(recall_score(lab[lab_k][:-1], pred[:-1]))
            
        precision_scores.append(precision_score(lab[lab_k][:-1], pred[:-1]))
        
        prec, rec, f1, _ = B_measure(pred, lab[lab_k])
        
        bf1_scores.append(f1)
        bprecision_scores.append(prec)
        brecall_scores.append(rec)
            
    all_scores_f1[enc] = f1_scores
    all_scores_precision[enc] = precision_scores
    all_scores_recall[enc] = recall_scores
    
    all_scores_bf1[enc] = bf1_scores
    all_scores_bprecision[enc] = bprecision_scores
    all_scores_brecall[enc] = brecall_scores
    
    boots = bootstrap(f1_scores)
    
    confidence = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5))/2
    
    df["F1"].append(np.mean(boots))
    df["F1 Confidence"].append(confidence)
    
    boots = bootstrap(precision_scores)
    
    confidence = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5))/2
    
    df["Precision"].append(np.mean(boots))
    df["Precision Confidence"].append(confidence)
    
    boots = bootstrap(recall_scores)
    
    confidence = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5))/2
    
    df["Recall"].append(np.mean(boots))
    df["Recall Confidence"].append(confidence)
    
    boots = bootstrap(bf1_scores)
    
    confidence = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5))/2
    
    df["B-F1"].append(np.mean(boots))
    df["B-F1 Confidence"].append(confidence)
    
    boots = bootstrap(bprecision_scores)
    
    confidence = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5))/2
    
    df["B-Precision"].append(np.mean(boots))
    df["B-Precision Confidence"].append(confidence)
    
    boots = bootstrap(brecall_scores)
    
    confidence = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5))/2
    
    df["B-Recall"].append(np.mean(boots))
    df["B-Recall Confidence"].append(confidence)
    
    df["embedding"].append(enc)
    op = False


df = pd.DataFrame(df)

f1_sorted_idx = df["F1"].sort_values(ascending = False).index
precision_sorted_idx = df["Precision"].sort_values(ascending = False).index
recall_sorted_idx = df["Recall"].sort_values(ascending = False).index

bf1_sorted_idx = df["B-F1"].sort_values(ascending = False).index
bprecision_sorted_idx = df["B-Precision"].sort_values(ascending = False).index
brecall_sorted_idx = df["B-Recall"].sort_values(ascending = False).index

f1_p = np.zeros(len(df))
precision_p = np.zeros(len(df))
recall_p = np.zeros(len(df))

bf1_p = np.zeros(len(df))
bprecision_p = np.zeros(len(df))
brecall_p = np.zeros(len(df))

for index, e in enumerate(f1_sorted_idx[:-1]):
    a = all_scores_f1[df.iloc[e, -1]]
    b = all_scores_f1["mfcc"]
    
    
    
    # b = all_scores_f1[df.iloc[f1_sorted_idx[index+1], -1]]
    
    var_a = np.var(a)
    var_b = np.var(b)
    
    if var_a>var_b:
        var_ratio = var_a/var_b
    else:
        var_ratio = var_b/var_a
        
    if var_ratio>4:
        f1_p[e] = ttest_ind(a, b, equal_var=False).pvalue
    else:
        f1_p[e] = ttest_ind(a, b).pvalue
        
for index, e in enumerate(precision_sorted_idx[:-1]):
    a = all_scores_precision[df.iloc[e, -1]]
    #b = all_scores_precision[df.iloc[precision_sorted_idx[index+1], -1]]
    b = all_scores_precision["mfcc"]
    var_a = np.var(a)
    var_b = np.var(b)
    
    if var_a>var_b:
        var_ratio = var_a/var_b
    else:
        var_ratio = var_b/var_a
        
    if var_ratio>4:
        precision_p[e] = ttest_ind(a, b, equal_var=False).pvalue
    else:
        precision_p[e] = ttest_ind(a, b).pvalue
        
for index, e in enumerate(recall_sorted_idx[:-1]):
    a = all_scores_recall[df.iloc[e, -1]]
    #b = all_scores_recall[df.iloc[recall_sorted_idx[index+1], -1]]
    b = all_scores_recall["mfcc"]
    var_a = np.var(a)
    var_b = np.var(b)
    
    if var_a>var_b:
        var_ratio = var_a/var_b
    else:
        var_ratio = var_b/var_a
        
    if var_ratio>4:
        recall_p[e] = ttest_ind(a, b, equal_var=False).pvalue
    else:
        recall_p[e] = ttest_ind(a, b).pvalue
        
for index, e in enumerate(bf1_sorted_idx[:-1]):
    a = all_scores_bf1[df.iloc[e, -1]]
    #b = all_scores_bf1[df.iloc[bf1_sorted_idx[index+1], -1]]
    b = all_scores_bf1["mfcc"]
    var_a = np.var(a)
    var_b = np.var(b)
    
    if var_a>var_b:
        var_ratio = var_a/var_b
    else:
        var_ratio = var_b/var_a
        
    if var_ratio>4:
        bf1_p[e] = ttest_ind(a, b, equal_var=False).pvalue
    else:
        bf1_p[e] = ttest_ind(a, b).pvalue
        
for index, e in enumerate(precision_sorted_idx[:-1]):
    a = all_scores_bprecision[df.iloc[e, -1]]
    #b = all_scores_bprecision[df.iloc[bprecision_sorted_idx[index+1], -1]]
    b = all_scores_bprecision["mfcc"]
    var_a = np.var(a)
    var_b = np.var(b)
    
    if var_a>var_b:
        var_ratio = var_a/var_b
    else:
        var_ratio = var_b/var_a
        
    if var_ratio>4:
        bprecision_p[e] = ttest_ind(a, b, equal_var=False).pvalue
    else:
        bprecision_p[e] = ttest_ind(a, b).pvalue
        
for index, e in enumerate(brecall_sorted_idx[:-1]):
    a = all_scores_brecall[df.iloc[e, -1]]
    #b = all_scores_brecall[df.iloc[brecall_sorted_idx[index+1], -1]]
    b = all_scores_brecall["mfcc"]
    var_a = np.var(a)
    var_b = np.var(b)
    
    if var_a>var_b:
        var_ratio = var_a/var_b
    else:
        var_ratio = var_b/var_a
        
    if var_ratio>4:
        brecall_p[e] = ttest_ind(a, b, equal_var=False).pvalue
    else:
        brecall_p[e] = ttest_ind(a, b).pvalue

# for index, e in enumerate(f1_sorted_idx[:-1]):
#     a = all_scores_f1[df.iloc[e, -1]]
#     b = all_scores_f1[df.iloc[f1_sorted_idx[index+1], -1]]
    
#     var_a = np.var(a)
#     var_b = np.var(b)
    
#     if var_a>var_b:
#         var_ratio = var_a/var_b
#     else:
#         var_ratio = var_b/var_a
        
#     if var_ratio>4:
#         f1_p[e] = ttest_ind(a, b, equal_var=False).pvalue
#     else:
#         f1_p[e] = ttest_ind(a, b).pvalue
        
# for index, e in enumerate(precision_sorted_idx[:-1]):
#     a = all_scores_precision[df.iloc[e, -1]]
#     b = all_scores_precision[df.iloc[precision_sorted_idx[index+1], -1]]
    
#     var_a = np.var(a)
#     var_b = np.var(b)
    
#     if var_a>var_b:
#         var_ratio = var_a/var_b
#     else:
#         var_ratio = var_b/var_a
        
#     if var_ratio>4:
#         precision_p[e] = ttest_ind(a, b, equal_var=False).pvalue
#     else:
#         precision_p[e] = ttest_ind(a, b).pvalue
        
# for index, e in enumerate(recall_sorted_idx[:-1]):
#     a = all_scores_recall[df.iloc[e, -1]]
#     b = all_scores_recall[df.iloc[recall_sorted_idx[index+1], -1]]
    
#     var_a = np.var(a)
#     var_b = np.var(b)
    
#     if var_a>var_b:
#         var_ratio = var_a/var_b
#     else:
#         var_ratio = var_b/var_a
        
#     if var_ratio>4:
#         recall_p[e] = ttest_ind(a, b, equal_var=False).pvalue
#     else:
#         recall_p[e] = ttest_ind(a, b).pvalue
        
# for index, e in enumerate(bf1_sorted_idx[:-1]):
#     a = all_scores_bf1[df.iloc[e, -1]]
#     b = all_scores_bf1[df.iloc[bf1_sorted_idx[index+1], -1]]
    
#     var_a = np.var(a)
#     var_b = np.var(b)
    
#     if var_a>var_b:
#         var_ratio = var_a/var_b
#     else:
#         var_ratio = var_b/var_a
        
#     if var_ratio>4:
#         bf1_p[e] = ttest_ind(a, b, equal_var=False).pvalue
#     else:
#         bf1_p[e] = ttest_ind(a, b).pvalue
        
# for index, e in enumerate(precision_sorted_idx[:-1]):
#     a = all_scores_bprecision[df.iloc[e, -1]]
#     b = all_scores_bprecision[df.iloc[bprecision_sorted_idx[index+1], -1]]
    
#     var_a = np.var(a)
#     var_b = np.var(b)
    
#     if var_a>var_b:
#         var_ratio = var_a/var_b
#     else:
#         var_ratio = var_b/var_a
        
#     if var_ratio>4:
#         bprecision_p[e] = ttest_ind(a, b, equal_var=False).pvalue
#     else:
#         bprecision_p[e] = ttest_ind(a, b).pvalue
        
# for index, e in enumerate(brecall_sorted_idx[:-1]):
#     a = all_scores_brecall[df.iloc[e, -1]]
#     b = all_scores_brecall[df.iloc[brecall_sorted_idx[index+1], -1]]
    
#     var_a = np.var(a)
#     var_b = np.var(b)
    
#     if var_a>var_b:
#         var_ratio = var_a/var_b
#     else:
#         var_ratio = var_b/var_a
        
#     if var_ratio>4:
#         brecall_p[e] = ttest_ind(a, b, equal_var=False).pvalue
#     else:
#         brecall_p[e] = ttest_ind(a, b).pvalue
        
df["F1 P-value"] = f1_p
df["Precision P-value"] = precision_p
df["Recall P-value"] = recall_p

df["B-F1 P-value"] = bf1_p
df["B-Precision P-value"] = bprecision_p
df["B-Recall P-value"] = brecall_p

df.to_csv(os.path.join(prefix, "all_result_bilstm.csv"), index=False)
        
    
    


