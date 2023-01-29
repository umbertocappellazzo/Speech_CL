#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 16:22:26 2022

@author: umbertocappellazzo
"""

from typing import Tuple, Union, Optional, List
from functools import lru_cache
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset as TorchDataset
try:
    import soundfile
except:
    soundfile = None
#from tools.utils import TextTransform


def _tensorize_list(x):
    if isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return torch.tensor(x)


class AudioTaskSet(TorchDataset):
    """A task dataset specific to  text returned by the CLLoader.
    :param x: The data, text here
    :param y: The targets, not one-hot encoded.
    :param t: The task id of each sample.
    :param trsf: The transformations to apply on the images.
    :param target_trsf: The transformations to apply on the labels.
    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            t: np.ndarray,
            transcripts,
            trsf,
            target_trsf: Optional[Union[transforms.Compose, List[transforms.Compose]]] = None,
            splitting_crit=None
    ):
        if not soundfile:
            raise ImportError("You need to install the soundfile library to work on audio data.")

        self._x, self._y, self._t, self._transcripts = x, y, t, transcripts
        if self._t is None:
            self._t = -1 * np.ones(len(x), dtype=np.int64)

        self.trsf = trsf
        
        self.target_trsf = target_trsf

        self._to_tensor = transforms.ToTensor()
        self.splitting_crit = splitting_crit
        

    def _transform_y(self, y, t):
        """Array of all classes contained in the current task."""
        for i, (y_, t_) in enumerate(zip(y, t)):
            y[i] = self.get_task_target_trsf(t_)(y_)
        return y
    
    @property
    def nb_classes(self):
        """The number of classes contained in the current task."""
        return len(self.get_classes())
    
    @lru_cache(maxsize=1)
    def get_classes(self) -> List[int]:
        """Array of all classes contained in the current task."""
        if self.target_trsf is not None:
            y = self._transform_y(self._y, self._t)
        else:
            y = self._y
        if self.splitting_crit is None:
            
            return np.unique(y)
        else:
            if self.splitting_crit == 'scenario':
                return np.unique(y[:,0])
            else:
                return np.unique(y[:,1])
    
    
    def concat(self, *task_sets):
        """Concat others task sets.
        :param task_sets: One or many task sets.
        """
        for task_set in task_sets:
            

            self.add_samples(task_set._x, task_set._y, task_set._t,task_set._transcripts)
    
    
    
    def add_samples(self, x, y, t, transcripts):
        """Add memory for rehearsal.
        :param x: Sampled data chosen for rehearsal.
        :param y: The associated targets of `x_memory`.
        :param t: The associated task ids. If not provided, they will be
                         defaulted to -1.
        """
        self._x = np.concatenate((self._x, x))
        self._y = np.concatenate((self._y, y))
        if t is not None:
            self._t = np.concatenate((self._t, t))
        else:
            self._t = np.concatenate((self._t, -1 * np.ones(len(x))))
        #self._transcripts = np.concatenate((self._transcripts,transcripts))
        #print(len(transcripts))
        #print(len(self._transcripts))
        
        self._transcripts = self._transcripts + transcripts

    def __len__(self) -> int:
        """The amount of images in the current task."""
        return self._y.shape[0]
    
    def get_random_samples(self, nb_samples):
        nb_tot_samples = self._x.shape[0]
        indexes = np.random.randint(0, nb_tot_samples, nb_samples)
        return self.get_samples(indexes)
    
    def get_samples(self, indexes):
        samples, targets, tasks, transcripts = [], [], [], []

        for index in indexes:
            # we need to use __getitem__ to have the transform used
            sample, y, t, transcript = self[index]
            samples.append(sample)
            targets.append(y)
            tasks.append(t)
            transcripts.append(transcript)

        return _tensorize_list(samples), _tensorize_list(targets), _tensorize_list(tasks), _tensorize_list(transcripts)
    
    def get_raw_samples(self, indexes=None):
        """Get samples without preprocessing, for split train/val for example."""
        
        if indexes is None:
            return self._x, self._y, self._t, self._transcripts
        return self._x[indexes], self._y[indexes], self._t[indexes], self._transcripts[indexes]

    def get_sample(self, index: int) -> np.ndarray:
        """Returns a sample data corresponding to the given `index`.
        :param index: Index to query the image.
        :return: the sample data.
        """
        return torch.FloatTensor(soundfile.read(self._x[index])[0])

    def __getitem__(self, index: int):
        """Method used by PyTorch's DataLoaders to query a sample and its target."""
        x = self.get_sample(index)
        y = self._y[index]
        t = self._t[index]
        transcripts = self._transcripts[index]
        #map_transcripts = TextTransform() 
        #transcripts = map_transcripts.text_to_int(transcripts)
        
        trsf = self.get_task_trsf(t)
        
        if trsf:
            x = trsf(x)

        if self.target_trsf is not None:
            y = self.get_task_target_trsf(t)(y)

        return x, y, t, transcripts
    
    def get_task_trsf(self, t: int):
        if isinstance(self.trsf, list):
            return self.trsf[t]
        return self.trsf

    def get_task_target_trsf(self, t: int):
        if isinstance(self.target_trsf, list):
            return self.target_trsf[t]
        return self.target_trsf
