import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
from config import *

class KvasirDataset(Dataset):
    def __init__(self, image_ids, images, masks, augment=False):
        self.image_ids = image_ids
        self.images = images
        self.masks = masks
        self.augment = augment and USE_AUGMENTATION
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = self.images[image_id]
        mask = self.masks[image_id]
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)
        return image, mask
    
    def _apply_augmentation(self, image, mask):
        if HORIZONTAL_FLIP and torch.rand(1) > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            
        if VERTICAL_FLIP and torch.rand(1) > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            
        if MAX_ROTATION > 0:
            angle = torch.randint(-MAX_ROTATION, MAX_ROTATION, (1,)).item()
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        
        # Re-binarize mask after transformations
        mask = (mask > 0.5).float()
        return image, mask
