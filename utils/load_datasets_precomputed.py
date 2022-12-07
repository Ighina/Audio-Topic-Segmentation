# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 15:37:42 2022

@author: User
"""

import os
import json
import pickle
import numpy as np
import torch

def cross_validation_split(dataset, num_folds = 5, n_test_folds = 1, inverse_augmentation = True):
  unit_size = len(dataset)//num_folds
  test_size = len(dataset)//num_folds * n_test_folds
  folds = []
  for i in range(num_folds):
    test_start_idx = i*unit_size
    test_end_idx = i*unit_size + test_size
    test = dataset[test_start_idx:test_end_idx]
    if i == num_folds+1-n_test_folds:
        test += dataset[:test_size//n_test_folds]
        train = dataset[test_size//n_test_folds:-test_size//n_test_folds]
    else:
        train = dataset[:test_start_idx] + dataset[test_end_idx:]
    break_point = 10000
    new_data = 0
    if inverse_augmentation:
            print("previous train size"+ str(len(train)))
            max_new_programs = 10
            for i, tup in enumerate(train):
                new_data += len(tup[1])
                if max_new_programs<i:
                    print(i)
                    break
                    
                start_index = 0
                temp_data = []
                temp_lab = []
                temp_segment_lab = []
                for index, lab in enumerate(tup[1]):
                    temp_segment_lab.append(lab)
                    if lab:
                        temp_data.append(tup[0][start_index:index+1])
                        start_index = index+1
                        temp_lab.append(temp_segment_lab)
                        temp_segment_lab = []
                print()
                combined = [torch.tensor([]),[]]
                for index in reversed(range(len(temp_data))):
                    combined[0] = torch.cat((combined[0], temp_data[index]), axis = 0)
                    combined[1].extend(temp_lab[index])
                train.append(combined)
            print("new train size"+str((len(train))))
    
    folds.append([train, test])
  return folds

def load_dataset_from_precomputed(embedding_directory,
                                  lab_file,
                                  delete_last_sentence = False, 
                                  compute_confidence_intervals = False,
                                  inverse_augmentation = False,
                                  umap_project = False,
                                  k_folds = 5,
                                  mask_inner_sentences = False,
                                  mask_probability = 0.9,
                                  split = None):
    standard_split = False    
    if split is not None:
        with open(split) as f:
            split = json.load(f)
        standard_split = True
        data = [[],[],[]]
    else:
        data = []
    original_data = []
    with open(lab_file, 'rb') as f:
        labs = pickle.load(f)
    assert isinstance(labs, dict)
    #audio_dir = os.path.dirname(list(labs.keys())[0])
    
    root = embedding_directory
    
    if standard_split:
        Train = True
        Test = False
    
    for index, file in enumerate(os.listdir(embedding_directory)):
        if file[-16:]==":Zone.Identifier": # artifacts from the downloading process
            continue
        if file[:-4] in ("24580", "25539", "25684", "26071", "26214", "26321", "26427"): # get rid of files that are too long from the Podcast dataset
            continue
        
        if standard_split:
            if len(split["train"]):
                file = split["train"].pop()
            elif len(split["test"]):
                file = split["test"].pop()
                Train = False
                Test = True
            else:
                Train = False
                Test = False
                file = split["validation"].pop()    
        
        embs = torch.from_numpy(np.load(os.path.join(root, file)).squeeze()) # squeezing in case an extra dimension made its way into the embedding collection (should be 2d)
        
        # try:
        #     try:
        #         file_name = audio_dir + '/' + file[:-4] + '.mp3'
        #         if len(labs[file_name])<1:
        #             print("Warning: {} has no data".format(file_name))
        #             continue
        #         labs[file_name][-1] = 0
        #     except KeyError:
        #         file_name = audio_dir + '/' + file[:-4] + '.wav'
        #         if len(labs[file_name])<1:
        #             continue
        #         labs[file_name][-1] = 0
        #     # data.append((embs, labs[file_name]))
        # except KeyError:
        #     try:
        #         file_name = audio_dir + '/audio\\' + file[:-4] + '.mp3'
        #         if len(labs[file_name])<1:
        #             continue
        #         labs[file_name][-1] = 0
        #     except KeyError:
        #         file_name = audio_dir + '/BMAT-ATS\\' + file[:-4] + '.wav'
        #         if len(labs[file_name])<1:
        #             continue
        #         labs[file_name][-1] = 0
        file_name = file[:-4]
        if len(labs[file_name])<1:
            print("Warning: {} has no data".format(file_name))
            continue
        labs[file_name][-1] = 0    
            
        if mask_inner_sentences:
            original_data.append((embs, labs[file_name].copy(), file))
            np.random.seed(1)

            embs_list = embs.tolist()
            popped = 0
            for index_e, emb in enumerate(embs):
                if np.random.rand()>mask_probability and not labs[file_name][index_e-popped]:
                    embs_list.pop(index_e-popped)
                    labs[file_name].pop(index_e-popped)
                    popped+=1
            embs = torch.tensor(embs_list)
        

        # assert sum(labs[file_name])>0, "{} has no positive topic boundaries".format(file_name)
        if sum(labs[file_name])<1:
            print("Warning: {} has no positive topic boundaries".format(file_name))     
        
        if standard_split:
            if Train:
                data[0].append((embs, labs[file_name], file))
            elif Test:
                data[1].append((embs, labs[file_name], file))
            else:
                data[2].append((embs, labs[file_name], file))
        else:
            data.append((embs, labs[file_name], file))

    
    if standard_split:
        return [data]

    folds = cross_validation_split(data, num_folds = k_folds, inverse_augmentation = False)
    if mask_inner_sentences: # restore the original test data in the cross-validation folder
        for index in range(len(folds)):
            folds[index][1] = [original_data[index]]
    
        
    return folds



def load_dataset_for_inference(embedding_directory):
    
    data = []
    root = embedding_directory
    
    files = os.listdir(embedding_directory)
    
    for index, file in enumerate(os.listdir(embedding_directory)):
        embs = torch.from_numpy(np.load(os.path.join(root, file)).squeeze()) # squeezing in case an extra dimension made its way into the embedding collection (should be 2d)
        
        data.append(embs)

    return data, files