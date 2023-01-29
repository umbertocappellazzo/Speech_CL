#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 14:53:32 2022

@author: umbertocappellazzo
"""

import abc
import os

from typing import List, Union


class _ContinuumDataset(abc.ABC):

    def __init__(self, data_path: str = "", train: bool = True, download: bool = True) -> None:
        self.data_path = os.path.expanduser(data_path) if data_path is not None else None
        self.download = download
        self.train = train

        if self.data_path is not None and self.data_path != "" and not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        if self.download:
            self._download()
        
    def get_data(self):
        """Returns the loaded data under the form of x, y, and t."""
        raise NotImplementedError("This method should be implemented!")

    def _download(self):
        pass

    
    @property
    def nb_classes(self) -> List[int]:
        return None

    @property
    def class_order(self) -> Union[None, List[int]]:
        return None    