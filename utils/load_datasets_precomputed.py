# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 15:37:42 2022

@author: User
"""

import os
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
                                  lab_from_array = False,
                                  inverse_augmentation = False,
                                  umap_project = False,
                                  k_folds = 5,
                                  mask_inner_sentences = False,
                                  mask_probability = 0.9):
    
    data = []
    original_data = []
    if lab_from_array:
        labs = np.load(lab_file, allow_pickle = True)
    else:
        with open(lab_file, 'rb') as f:
            labs = pickle.load(f)
        assert isinstance(labs, dict)
        audio_dir = os.path.dirname(list(labs.keys())[0])
    
    root = embedding_directory
    
    files = os.listdir(embedding_directory)
    
    # Below: specifying file order for reproducibility across operating systems
    if len(files)==57:
        """
        NonNews dataset
        """
        files = ['b06vmxny.npy', 
                'b06tvswy.npy', 
                'b06rw50x.npy', 
                'b0bgw8c6.npy', 
                'b047w54x.npy', 
                'b04xp15f.npy', 
                'b04c9gsd.npy', 
                'b00pchsr.npy', 
                'b06zv3x9.npy', 
                'b06vkj1y.npy', 
                'b06ztttm.npy', 
                'b04xn99p.npy', 
                'b0499j2m.npy', 
                'b0b7d6r2.npy', 
                'b070nqx1.npy', 
                'b00v11ck.npy', 
                'b04wwkhd.npy', 
                'b048nlfg.npy', 
                'b06fl8yq.npy', 
                'b071fyq9.npy', 
                'b070dq8c.npy', 
                'b01mn32h.npy', 
                'b06wg6y1.npy', 
                'b06k8x4d.npy', 
                'b0b5qgp0.npy', 
                'b04xfc1f.npy', 
                'b00mvcxc.npy', 
                'b0b7cdp3.npy', 
                'b0bjyq89.npy', 
                'b06gp9p8.npy', 
                'b06pv3gz.npy', 
                'b06s75n5.npy', 
                'b0bbnrct.npy', 
                'b06p4jvl.npy', 
                'b0bgp09w.npy', 
                'b06wv9c8.npy', 
                'b070d28r.npy', 
                'b0bcdd4d.npy', 
                'b0b4yb4y.npy', 
                'b0bjyw68.npy', 
                'b048033z.npy', 
                'b06whswj.npy', 
                'b06zvdll.npy', 
                'b049p9yw.npy', 
                'b070fn1w.npy', 
                'b0705765.npy', 
                'b0b6btzq.npy', 
                'b0b42tlv.npy', 
                'b04d0hxv.npy', 
                'b070hn0y.npy', 
                'b06wcq19.npy', 
                'b048hxpp.npy', 
                'b06wc6qp.npy', 
                'b07lhh75.npy', 
                'b04xrv9s.npy', 
                'b0b5s5t8.npy', 
                'b06vn700.npy']
    
    
    
    for index, file in enumerate(os.listdir(embedding_directory)):
        if file[:-4] in ("24580", "25539", "25684", "26071", "26214", "26321", "26427"): # get rid of files that are too long from the Podcast dataset
            continue
        embs = torch.from_numpy(np.load(os.path.join(root, file)).squeeze()) # squeezing in case an extra dimension made its way into the embedding collection (should be 2d)
        
        if lab_from_array:
            labs[index][-1] = 0
            data.append((embs, labs[index], file))
        else:
            try:
                try:
                    file_name = audio_dir + '/' + file[:-4] + '.mp3'
                    if len(labs[file_name])<1:
                        print("Warning: {} has no data".format(file_name))
                        continue
                    labs[file_name][-1] = 0
                except KeyError:
                    file_name = audio_dir + '/' + file[:-4] + '.wav'
                    if len(labs[file_name])<1:
                        continue
                    labs[file_name][-1] = 0
                # data.append((embs, labs[file_name]))
            except KeyError:
                try:
                    file_name = audio_dir + '/audio\\' + file[:-4] + '.mp3'
                    if len(labs[file_name])<1:
                        continue
                    labs[file_name][-1] = 0
                except KeyError:
                    file_name = audio_dir + '/BMAT-ATS\\' + file[:-4] + '.wav'
                    if len(labs[file_name])<1:
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
            
            data.append((embs, labs[file_name], file))

    
    
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