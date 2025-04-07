import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import cv2
import torchvision.transforms.functional as TF
from PIL import ImageFilter
from config import *
import random

class KvasirDataset(Dataset):
    def __init__(self, image_ids, images, masks, augment=False):
        self.image_ids = image_ids
        self.images = images
        self.masks = masks
        self.augment = augment
    
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
        mask = (mask > 0.5).float()
        return image, mask

class BUSIDataset(Dataset):
    def __init__(self,  images, masks, augment=False):
        self.images = images
        self.masks = masks
        self.augment = augment and USE_AUGMENTATION
        self.image_ids = list(images.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = self.images[image_id].clone()
        mask = self.masks[image_id].clone()

        # CLAHE, speckle reduction for image
        image = image.permute(1, 2, 0).numpy()
        image = self._apply_clahe(image)
        image = self._apply_speckle_reduction(image)
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Apply data augmentation
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)
        mask = (mask > 0.5).float()
        
        return image, mask
    
    def _apply_clahe(self, image):
        # Get LAB color space
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge and convert to RGB
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        return enhanced.astype(np.float32) / 255.0
    
    def _apply_speckle_reduction(self, image):
        # Convert to grayscale for denoising
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Apply Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Convert to RGB and combine with original
        denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        blended = cv2.addWeighted((image * 255).astype(np.uint8), 0.7, denoised_rgb, 0.3, 0)
        
        return blended.astype(np.float32) / 255.0
    
    def _add_shadow_artifact(self, image):
        # Create a shadow mask tensor
        h, w = image.shape[1], image.shape[2]
        shadow_mask = torch.ones_like(image)
        
        # Decide on vertical or horizontal shadow
        if random.random() < 0.5:  # Horizontal shadow
            x1, x2 = sorted([random.randint(0, w-1), random.randint(0, w-1)])
            shadow_width = max(5, int((x2 - x1) * random.uniform(0.3, 0.7)))
            x_center = (x1 + x2) // 2
            x1, x2 = x_center - shadow_width // 2, x_center + shadow_width // 2
            # Gradual shadow effect
            for i in range(x1, min(x2, w)):
                factor = 1.0 - random.uniform(0.3, 0.7) * np.sin(np.pi * (i - x1) / max(1, (x2 - x1)))
                shadow_mask[:, :, i] = factor
        else:  # Vertical shadow
            y1, y2 = sorted([random.randint(0, h-1), random.randint(0, h-1)])
            shadow_height = max(5, int((y2 - y1) * random.uniform(0.3, 0.7)))
            y_center = (y1 + y2) // 2
            y1, y2 = y_center - shadow_height // 2, y_center + shadow_height // 2
            
            # Gradual shadow effect
            for i in range(y1, min(y2, h)):
                factor = 1.0 - random.uniform(0.3, 0.7) * np.sin(np.pi * (i - y1) / max(1, (y2 - y1)))
                shadow_mask[:, i, :] = factor
        
        # Apply shadow
        image = image * shadow_mask
        
        return image
    
    def _apply_augmentation(self, image, mask):
        # Horizontal flip
        if HORIZONTAL_FLIP and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Vertical flip
        if VERTICAL_FLIP and random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Rotation
        if MAX_ROTATION > 0 and random.random() > 0.3:
            angle = random.uniform(-MAX_ROTATION, MAX_ROTATION)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        
        # Random brightness and contrast
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)
            image = TF.adjust_contrast(image, contrast_factor)
        
        # Random blur
        if random.random() > 0.7:
            # Convert to PIL for Gaussian blur
            image_pil = TF.to_pil_image(image)
            blur_radius = random.uniform(0, 1.5)
            image_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            image = TF.to_tensor(image_pil)
        
        # Add simulated ultrasound shadow artifact
        if random.random() > 0.7:
            image = self._add_shadow_artifact(image)
        
        # Add gaussian noise
        if random.random() > 0.7:
            noise = torch.randn_like(image) * random.uniform(0.01, 0.07)
            image = torch.clamp(image + noise, 0, 1)
            
        return image, mask