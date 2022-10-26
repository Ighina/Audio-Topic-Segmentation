# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:00:35 2022

@author: User
"""

import argparse
import os
import numpy as np
import json
import librosa
import soundfile as sf
import speech_recognition as sr
import sys

def sr_to_time(sr, sample):
    return sample/sr

r = sr.Recognizer()

def main(args):
    out_dir = args.out_directory
    in_dir = args.in_directory
    
    ground_truths = {}
    
    if args.transcribe:
        with open(args.transcription_file) as f:
            transcription_file = json.load(f)
    
    if args.new_dataset:
        
        if not os.path.exists(args.out_directory):
            os.mkdir(args.out_directory)
        
        instructions = {}
        
        used_combinations = set()
        
        files = os.listdir(in_dir)
            
        if args.number_documents*args.number_segments > len(files)**2:
            raise ValueError("The size of the dataset is too big to include just unique combinations of consecutive segments!")
            
        
        for doc_index in range(args.number_documents):
            
            segments = []
            
            for segment_index in range(args.number_segments//2):
                
                seg = np.random.choice(files, 2)
                
                while tuple(seg) in used_combinations:
                    seg = np.random.choice(files, 2)
                
                used_combinations.add(tuple(seg))
            
                segments.extend(seg)
                
            instructions["doc"+str(doc_index)] = segments
            
        with open(args.dataset_compose, "w") as f:
            json.dump(instructions, f)
                
        
                
    else:
        instructions = json.load(args.dataset_compose)
    
    index = 0
    
    for doc, segments in instructions.items():
        
        labs = []
        
        transcript = ""
        
        new_audio = []
        
        for segment in segments:
            filename = os.path.join(in_dir, segment)
            
            audio, sample = librosa.load(filename)
            
            new_audio.append(audio)
            
            labs.append(sr_to_time(sample, len(audio)))
            
            if args.transcribe:
                text = transcription_file[segment]
                
                text = " ".join([word["word"] for word in text])
            
                transcript = "\n====\n".join([transcript, text])
        
        if args.transcribe:
            if not os.path.exists(args.transcript_directory):
                os.mkdir(args.transcript_directory)
            transcript_file = os.path.join(args.transcript_directory, "_".join([doc, str(index)]))
            with open(transcript_file, "w") as f:
                f.write(transcript)
                
        new_audio = np.concatenate(new_audio)
        
        outfile = os.path.join(out_dir, "_".join([doc, str(index)])+'.wav')
        
        sf.write(outfile, new_audio, sample)
        
        ground_truths["_".join([doc, str(index)])] = labs
                
        index += 1
        
    with open(args.ground_truth_file, "w") as f:
        json.dump(ground_truths, f)
        
        
if __name__=='__main__':
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
            
    
    parser = MyParser(
                description = 'Create a dataset for audio topic segmentation starting from separated audio segments.')
    
    parser.add_argument('--in_directory', '-in', default = "news_segments", type = str,
                        help = "The directory containing the separate files.")
    
    parser.add_argument('--out_directory', '-out', default = "BMAT-ATS", type = str,
                        help = "The directory where to save the resulting dataset.")
    
    parser.add_argument('--new_dataset', '-new', action = "store_true",
                        help = "Whether to create a new dataset, then saving the composing elements in the file defined by the dataset_compose argument.")
    
    parser.add_argument('--dataset_compose', '-dc', default = "BMAT-ATS.json", type = str, 
                        help = "The path to the file defining the composition of the dataset.")
    
    parser.add_argument('--number_documents', '-nd', default = 100, type = int,
                        help = "The number of documents to be included in a new dataset.")
    
    parser.add_argument('--number_segments', '-ns', default = 10, type = int, 
                        help = "The number of segments in each document to be included in a new dataset.")
    
    parser.add_argument('--transcribe', '-t', action = 'store_true',
                        help = "Whether to transcribe the input audio files and save the relative transcripts, as well.")
    
    parser.add_argument('--transcription_file', "-tf", default = "BMAT_transcripts.json", type = str, help = "The file including the transcript for each segment.")
    
    parser.add_argument('--transcript_directory', '-td', default = "BMAT-ATS_transcripts", type = str,
                        help = "The directory where to store the transcripts of each document.")
    
    parser.add_argument('--ground_truth_file', '-gtf', default = "BMAT-ATS_labels.json", type = str,
                        help = "Where to store the ground truths timing separating each segment in each document.")
    
    args = parser.parse_args()
    
    main(args)