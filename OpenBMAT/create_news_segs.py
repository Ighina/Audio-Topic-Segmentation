# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:01:04 2022

@author: User
"""

import re
import os
import shutil
import json
import librosa
import soundfile as sf

def time_to_sample(sr, time):
    return int(sr*time)

just_uk = input("Use just UK news? If yes, enter anything, otherwise enter nothing...\n")

agreement_threshold = 0.5

out_dir = "news_segments"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

with open(os.path.join("annotations", "json", "MD_mapping.json"), 'rb') as f:
    MD_annotations = json.load(f)

if just_uk:
    pattern = "United_Kingdom_News"
else:
    pattern = "News"

cut_music = False

for root, _, files in os.walk("audio"):
    for file_index, file in enumerate(files):
        if re.findall(pattern, file):
            
            source = os.path.join(root, file)
            
            if cut_music:
            
                audio, sr = librosa.load(source)
                
                if float(MD_annotations["agreement"][file[:-4]]["vals"]["full"])>agreement_threshold:
                    segs = MD_annotations["annotations"]["annotator_a"][file[:-4]]
                    for subseg_index, subseg in segs.items():
                        if subseg["class"]!="music":
                            outfile = os.path.join(out_dir, "file{}_{}.wav".format(file_index, subseg_index))
                            start = time_to_sample(sr, subseg["start"])
                            end = time_to_sample(sr, subseg["end"])
                            sf.write(outfile, audio[start:end], sr)
                            
            else:
                    shutil.copy(source, os.path.join(out_dir, file))
        