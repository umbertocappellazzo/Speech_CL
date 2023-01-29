#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 13:57:07 2023

@author: umbertocappellazzo
"""

from torch.utils.data import DataLoader

from Speech_CLscenario.slurp_aug import Slurp
from Speech_CLscenario.class_incremental import ClassIncremental

import torch
from Speech_CLscenario.memory import RehearsalMemory
import warnings
warnings.simplefilter("ignore", UserWarning)
import os
from tokenizers import Tokenizer
import numpy as np

def data_processing(data,max_len_audio, tokenizer, SOS_token=2, EOS_token=3, PAD_token=0):
    
    SOS = torch.tensor([SOS_token])
    EOS = torch.tensor([EOS_token])
    
    # label_lengths are used if CTC loss is exploited in the experiments.
    
    #label_lengths = []
    transcripts_labels = [] 
    x = []
    y = []
    t= []
    
    for i in range(len(data)):
        audio_sig = data[i][0]
    
        if len(audio_sig) > max_len_audio:
            pass
        else:
            x.append(audio_sig)
            
            transcript = data[i][3]
            
            label = torch.tensor(tokenizer.encode(transcript).ids)
            label = torch.cat((SOS,label,EOS))
            
            #label_lengths.append(len(label))
            transcripts_labels.append(label)
            y.append(torch.tensor(data[i][1]))
            t.append(torch.tensor(data[i][2]))
    
    transcripts_labels = torch.nn.utils.rnn.pad_sequence(transcripts_labels, batch_first=True, padding_value=PAD_token)
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value = 0)
    y = torch.stack(y)
    t = torch.stack(t)
    
    return x,y,t,transcripts_labels  #,torch.tensor(label_lengths)
    


if __name__ == '__main__':
    data_path = '/data/cappellazzo/CL_SLU'
    
    dataset_train = Slurp(data_path, max_len_text=130, train="train", download=False)
    dataset_valid = Slurp(data_path, max_len_text=130, train="valid", download=False)
    
    scenario_train = ClassIncremental(dataset_train,increment=6,splitting_crit='scenario')  #18 scenarios overall, 6 scenarios per task --> 3 tasks. 
    scenario_valid = ClassIncremental(dataset_valid,increment=6,splitting_crit='scenario')
                                     
    max_len_audio = 112000
    memory_size =500
    
    path_2_tok = os.getcwd() + '/tokenizer_SLURP_BPE_1000_noblank_intents_SFaug.json'
    tokenizer = Tokenizer.from_file(path_2_tok)
    
    memory = None
    if memory_size > 0:
        memory = RehearsalMemory(memory_size, herding_method= 'random', 
                                            fixed_memory=True, nb_total_classes=18,splitting_crit='scenario')
    #print("Memory per class: ",memory.memory_per_class) 
    
    for task_id, exp_train in enumerate(scenario_train):
        
        
        if task_id > 0 and memory is not None:
            
            exp_train.add_samples(*memory.get())  # Include samples from the rehearsal memory.
        
        print("Shape of exp_train: ",len(exp_train))
        print(f"Starting task id {task_id}/{len(scenario_train) - 1}")
        
        valid_taskset = scenario_valid[:task_id+1]    # Evaluation on all seen tasks.
        
        train_loader = DataLoader(exp_train, batch_size=16, shuffle=True, num_workers=10,  
                                  collate_fn=lambda x: data_processing(x,max_len_audio,tokenizer),pin_memory=True,drop_last=False,)
        valid_loader = DataLoader(valid_taskset, batch_size=16, shuffle=False, num_workers=10,
                                  collate_fn = lambda x: data_processing(x,max_len_audio,tokenizer),pin_memory=True, drop_last=False,)
        
        for x, y, t, text in train_loader:
            print(x.shape)
            # Various computations...
            break
        
        
        if memory is not None:
            memory.add(*scenario_train[task_id].get_raw_samples(),z=None) 
            print("Seen classes by the memory: ",memory.seen_classes)
            
       
        