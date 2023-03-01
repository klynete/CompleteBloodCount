#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from torch.utils.data import DataLoader, Dataset
import os
import cv2
import time
import pandas as pd
import numpy as np
import torch

class blood_cell_dataset(Dataset):
    
    def __init__(self, dataframe, image_dir, mode = 'train', transforms = None):
        
        super().__init__()
        
        self.image_names = dataframe["filename"].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        self.mode = mode
        self.cell_types=dataframe['cell_type'].unique()
        self.classes = np.insert(self.cell_types, 0, "background", axis=0)
        self.class_to_int = {self.classes[i] : i for i in range(len(self.classes))}
        self.int_to_class = {i : self.classes[i] for i in range(len(self.classes))}
        
    def __getitem__(self, index: int):
        
        #Retrive Image name and its records ('xmin', 'ymin', 'xmax', 'ymax', 'cell_type') from df
        image_name = self.image_names[index]
        records = self.df[self.df["filename"] == image_name]
        
        #Loading Image
        image = cv2.imread(os.path.join(self.image_dir,image_name), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        if self.mode == 'train':
            
            #Get bounding box co-ordinates for each box
            boxes = records[['xmin', 'ymin', 'xmax', 'ymax']].values

            #Getting labels for each box
            temp_labels = records[['cell_type']].values
            labels = []
            for label in temp_labels:
                label = self.class_to_int[label[0]]
                labels.append(label)

            #Converting boxes & labels into torch tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            #Creating target
            target = {}
            target['boxes'] = boxes
            target['labels'] = labels

            #Transforms
            if self.transforms:
                image = self.transforms(image)


            return image, target, image_name
        
        elif self.mode == 'test':

            if self.transforms:
                image = self.transforms(image)

            return image, image_name
    
    def __len__(self):
        return len(self.image_names)

