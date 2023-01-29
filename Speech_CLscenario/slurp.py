#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:15:09 2022

@author: umbertocappellazzo
"""

import os
from Speech_CLscenario.base_dataset import _ContinuumDataset
#from base_dataset import _ContinuumDataset
#from class_incremental import ClassIncremental
import numpy as np
from typing import Union
import string

class Slurp(_ContinuumDataset):
    def __init__(self, data_path, max_len_text = None, train: Union[bool, str] = True, download: bool = True):
        if not isinstance(train, bool) and train not in ("train", "valid", "test","train_real","train_synthetic"):
            raise ValueError(f"`train` arg ({train}) must be a bool or train/valid/test/train_synthetic/train_real.")
        if isinstance(train, bool):
            if train:
                train = "train"
            else:
                train = "test"

        data_path = os.path.expanduser(data_path)
        self.max_len_text = max_len_text
        super().__init__(data_path, train, download)
        
    
    def get_data(self):
        path_to_wavs = "/data/cappellazzo/slurp"
        x, y, transcriptions = [], [], []  # For now ENTITIES are not taken into account.
        #digits = ['0','1','2','3','4','5','6','7','8','9'] 
        
        punctuations = list(string.punctuation)
        punctuations.append('’')
        
        with open(os.path.join(self.data_path, f"{self.train}.csv")) as f:
            lines = f.readlines()[1:]
        for line in lines:
            items = line[:-1].split(';')
            
            transcription = items[3].lower()
            # Remove first 2 characters 's1' or 's2' containing numbers. 
            #if transcription[:2] == 's1' or transcription[:2] == 's2':
            #    transcription = transcription[2:]
            
            # 47 transcriptions still have some numbers --> remove them.
            
            skip = False
            #if any((c in digits) for c in transcription):
            #    skip = True
  
            if skip or len(transcription) > self.max_len_text:
                pass
            
            else:
                
                transc = [char for char in transcription if char not in punctuations]
                transcription = ''
                for charr in transc:
                    transcription += charr
                transcriptions.append(transcription)
                
                
                scenario, action = items[4:6]
                #transcriptions.append(scenario)
                if 'synth' in items[2]:
                    #x.append(os.path.join(self.data_path,'slurp_synth', items[2]))
                    x.append(os.path.join(path_to_wavs,'slurp_synth', items[2]))
                else:
                    #x.append(os.path.join(self.data_path,'slurp_real', items[2]))
                    x.append(os.path.join(path_to_wavs,'slurp_real', items[2]))
                #transcription = items[3].lower()
                
                #transcriptions.append(items[3].lower())
                y.append([
                    self.scenarios[scenario],
                    self.actions[action],
                ])
        
        return np.array(x), np.array(y), None, transcriptions
        
        
        
    @property
    def transformations(self):
        return None    
    
    @property 
    def scenarios(self):
        return {
            'alarm': 0,
            'audio': 1,
            'calendar': 2,
            'cooking': 3,
            'datetime': 4,
            'email': 5,
            'general': 6,
            'iot': 7,
            'lists': 8,
            'music': 9,
            'news': 10,
            'play': 11,
            'qa': 12,
            'recommendation': 13,
            'social': 14,
            'takeaway': 15,
            'transport': 16,
            'weather': 17,
            }
    
    
    @property 
    def actions(self):
        return {
            'addcontact': 0,
            'affirm': 1, 
            'audiobook': 2,
            'cleaning': 3,
            'coffee': 4, 
            'commandstop': 5, 
            'confirm': 6, 
            'convert': 7, 
            'createoradd': 8, 
            'currency': 9, 
            'definition': 10, 
            'dislikeness': 11, 
            'dontcare': 12, 
            'events': 13, 
            'explain': 14, 
            'factoid': 15, 
            'game': 16, 
            'greet': 17, 
            'hue_lightchange': 18, 
            'hue_lightdim': 19, 
            'hue_lightoff': 20, 
            'hue_lighton': 21, 
            'hue_lightup': 22, 
            'joke': 23, 
            'likeness': 24, 
            'locations': 25, 
            'maths': 26, 
            'movies': 27, 
            'music': 28, 
            'negate': 29, 
            'order': 30, 
            'podcasts': 31, 
            'post': 32, 
            'praise': 33, 
            'query': 34, 
            'querycontact': 35, 
            'quirky': 36, 
            'radio': 37, 
            'recipe': 38, 
            'remove': 39, 
            'repeat': 40, 
            'sendemail': 41, 
            'set': 42, 
            'settings': 43, 
            'stock': 44, 
            'taxi': 45, 
            'ticket': 46, 
            'traffic': 47, 
            'volume_down': 48, 
            'volume_mute': 49, 
            'volume_other': 50, 
            'volume_up': 51, 
            'wemo_off': 52, 
            'wemo_on': 53,
            }





# if __name__ == "__main__":
#     data_path ='/Users/umbertocappellazzo/Desktop/PHD/CL_SLU/slurp'
#     a = Slurp(data_path,train=True,download=False)
#     _,_,_,b = a.get_data()
#     tot = len(b)
#     maxx = 0
#     print(tot)
#     text_transform = TextTransform()
#     count = 0
#     count_maxx = 0
#     check = ['0','1','2','3','4','5','6','7','8','9']
#     for x in b:
#         print(x)
#         c = text_transform.text_to_int(x)
#         #count += len(c)
#         #if len(c) > maxx:
#         #    maxx = len(c)
#         #    print("New max: ", maxx )
            
#         if len(c) > 129:
#             #print(x)
#             count_maxx += 1
#     #print(count/tot)
#     #print(maxx)
#     print(count_maxx)
#     #print(count)
        
        
    
        