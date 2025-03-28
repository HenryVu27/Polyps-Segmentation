import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# Original UNet
class OriginalUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(CHANNELS, FILTERS_ROOT)
        self.enc2 = ConvBlock(FILTERS_ROOT, FILTERS_ROOT * 2)
        self.enc3 = ConvBlock(FILTERS_ROOT * 2, FILTERS_ROOT * 4)
        self.enc4 = ConvBlock(FILTERS_ROOT * 4, FILTERS_ROOT * 8)
        
        # Bottleneck
        self.bottleneck = ConvBlock(FILTERS_ROOT * 8, FILTERS_ROOT * 16)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(FILTERS_ROOT * 16, FILTERS_ROOT * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(FILTERS_ROOT * 16, FILTERS_ROOT * 8)
        
        self.up3 = nn.ConvTranspose2d(FILTERS_ROOT * 8, FILTERS_ROOT * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(FILTERS_ROOT * 8, FILTERS_ROOT * 4)
        
        self.up2 = nn.ConvTranspose2d(FILTERS_ROOT * 4, FILTERS_ROOT * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(FILTERS_ROOT * 4, FILTERS_ROOT * 2)
        
        self.up1 = nn.ConvTranspose2d(FILTERS_ROOT * 2, FILTERS_ROOT, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(FILTERS_ROOT * 2, FILTERS_ROOT)
        
        # Output
        self.out = nn.Conv2d(FILTERS_ROOT, MASK_CHANNELS, kernel_size=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        # Output
        out = self.out(d1)
        return torch.sigmoid(out)

# Attention UNet
# Paper: https://arxiv.org/abs/1804.03999
class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder (downsampling)
        self.enc1 = ConvBlock(CHANNELS, FILTERS_ROOT)
        self.enc2 = ConvBlock(FILTERS_ROOT, FILTERS_ROOT * 2)
        self.enc3 = ConvBlock(FILTERS_ROOT * 2, FILTERS_ROOT * 4)
        self.enc4 = ConvBlock(FILTERS_ROOT * 4, FILTERS_ROOT * 8)
        
        # Bottleneck
        self.bottleneck = ConvBlock(FILTERS_ROOT * 8, FILTERS_ROOT * 16)
        
        # Attention
        self.att1 = AttentionGate(F_g=FILTERS_ROOT * 8, F_l=FILTERS_ROOT * 8, F_int=FILTERS_ROOT * 4)
        self.att2 = AttentionGate(F_g=FILTERS_ROOT * 4, F_l=FILTERS_ROOT * 4, F_int=FILTERS_ROOT * 2)
        self.att3 = AttentionGate(F_g=FILTERS_ROOT * 2, F_l=FILTERS_ROOT * 2, F_int=FILTERS_ROOT)
        
        # Decoder with attention
        self.up4 = nn.ConvTranspose2d(FILTERS_ROOT * 16, FILTERS_ROOT * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(FILTERS_ROOT * 16, FILTERS_ROOT * 8)
        
        self.up3 = nn.ConvTranspose2d(FILTERS_ROOT * 8, FILTERS_ROOT * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(FILTERS_ROOT * 8, FILTERS_ROOT * 4)
        
        self.up2 = nn.ConvTranspose2d(FILTERS_ROOT * 4, FILTERS_ROOT * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(FILTERS_ROOT * 4, FILTERS_ROOT * 2)
        
        self.up1 = nn.ConvTranspose2d(FILTERS_ROOT * 2, FILTERS_ROOT, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(FILTERS_ROOT * 2, FILTERS_ROOT)
        
        # Output
        self.out = nn.Conv2d(FILTERS_ROOT, MASK_CHANNELS, kernel_size=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder with attention
        d4 = self.up4(b)
        e4 = self.att1(g=d4, x=e4)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        e3 = self.att2(g=d3, x=e3)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        e2 = self.att3(g=d2, x=e2)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        # Output
        out = self.out(d1)
        return torch.sigmoid(out)

# nnUNet-inspired Architecture
# Inspired by: https://arxiv.org/abs/1809.10486
class NNUNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
        self.in1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.in1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.in2(x)
        x = self.leakyrelu(x)
        return x

class NNUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = NNUNetConvBlock(CHANNELS, FILTERS_ROOT)
        self.enc2 = NNUNetConvBlock(FILTERS_ROOT, FILTERS_ROOT * 2)
        self.enc3 = NNUNetConvBlock(FILTERS_ROOT * 2, FILTERS_ROOT * 4)
        self.enc4 = NNUNetConvBlock(FILTERS_ROOT * 4, FILTERS_ROOT * 8)
        self.enc5 = NNUNetConvBlock(FILTERS_ROOT * 8, FILTERS_ROOT * 16)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(FILTERS_ROOT * 16, FILTERS_ROOT * 8, kernel_size=2, stride=2)
        self.dec4 = NNUNetConvBlock(FILTERS_ROOT * 16, FILTERS_ROOT * 8)
        
        self.up3 = nn.ConvTranspose2d(FILTERS_ROOT * 8, FILTERS_ROOT * 4, kernel_size=2, stride=2)
        self.dec3 = NNUNetConvBlock(FILTERS_ROOT * 8, FILTERS_ROOT * 4)
        
        self.up2 = nn.ConvTranspose2d(FILTERS_ROOT * 4, FILTERS_ROOT * 2, kernel_size=2, stride=2)
        self.dec2 = NNUNetConvBlock(FILTERS_ROOT * 4, FILTERS_ROOT * 2)
        
        self.up1 = nn.ConvTranspose2d(FILTERS_ROOT * 2, FILTERS_ROOT, kernel_size=2, stride=2)
        self.dec1 = NNUNetConvBlock(FILTERS_ROOT * 2, FILTERS_ROOT)
        
        # Deep supervision outputs
        self.ds1 = nn.Conv2d(FILTERS_ROOT, MASK_CHANNELS, kernel_size=1)
        self.ds2 = nn.Sequential(
            nn.Conv2d(FILTERS_ROOT * 2, MASK_CHANNELS, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.ds3 = nn.Sequential(
            nn.Conv2d(FILTERS_ROOT * 4, MASK_CHANNELS, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )
        
        # Final output
        self.output = nn.Conv2d(FILTERS_ROOT, MASK_CHANNELS, kernel_size=1)
        
    def forward(self, x, return_deep_supervision=False):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        x5 = self.enc5(self.pool(x4))
        
        # Decoder
        d4 = self.up4(x5)
        d4 = torch.cat([x4, d4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([x3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.dec1(d1)
        
        # Output
        output = torch.sigmoid(self.output(d1))
        
        # Deep supervision for training
        if return_deep_supervision:
            ds1 = torch.sigmoid(self.ds1(d1))
            ds2 = torch.sigmoid(self.ds2(d2))
            ds3 = torch.sigmoid(self.ds3(d3))
            return output, [ds1, ds2, ds3]
        
        return output


# Loss (BCE + Dice)
class BCEDiceLoss(nn.Module):
    def __init__(self, weight=0.5):
        super().__init__()
        self.weight = weight
        self.bce = nn.BCELoss()
        
    def forward(self, pred, target):
        # For deep supervision
        if isinstance(pred, tuple) and len(pred) > 1:
            main_pred = pred[0]
            aux_preds = pred[1]
            loss = self._compute_loss(main_pred, target)
            
            # Add auxiliary losses
            for aux_pred in aux_preds:
                loss += 0.5 * self._compute_loss(aux_pred, target)
            return loss
        
        # Regular prediction
        return self._compute_loss(pred, target)
    
    def _compute_loss(self, pred, target):
        bce_loss = self.bce(pred, target)
        smooth = 1e-5
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
        return self.weight * bce_loss + (1 - self.weight) * dice_loss
    
# U-Net factory
class UNet(nn.Module):
    def __init__(self, arch="original"):
        super().__init__()
        
        arch_map = {
            "original": OriginalUNet,
            "attention": AttentionUNet,
            "nnunet": NNUNet
        }
        
        if arch.lower() not in arch_map:
            raise ValueError(f"Unknown architecture type: {arch}. Options are: {list(arch_map.keys())}")
        
        self.model = arch_map[arch.lower()]()
        self.arch = arch.lower()
        
    def forward(self, x, return_deep_supervision=False):
        if self.arch == "nnunet" and return_deep_supervision:
            return self.model(x, return_deep_supervision)
        return self.model(x)