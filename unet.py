import torch
import torch.nn as nn
from config import *

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # downsampling
        self.enc1 = DoubleConv(CHANNELS, FILTERS_ROOT)
        self.enc2 = DoubleConv(FILTERS_ROOT, FILTERS_ROOT*2)
        self.enc3 = DoubleConv(FILTERS_ROOT*2, FILTERS_ROOT*4)
        self.enc4 = DoubleConv(FILTERS_ROOT*4, FILTERS_ROOT*8)
        
        # bottleneck
        self.bottleneck = DoubleConv(FILTERS_ROOT*8, FILTERS_ROOT*16)
        
        # upsampling
        self.upconv4 = nn.ConvTranspose2d(FILTERS_ROOT*16, FILTERS_ROOT*8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(FILTERS_ROOT*16, FILTERS_ROOT*8)  # *16 because of concatenation
        
        self.upconv3 = nn.ConvTranspose2d(FILTERS_ROOT*8, FILTERS_ROOT*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(FILTERS_ROOT*8, FILTERS_ROOT*4)   # *8 because of concatenation
        
        self.upconv2 = nn.ConvTranspose2d(FILTERS_ROOT*4, FILTERS_ROOT*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(FILTERS_ROOT*4, FILTERS_ROOT*2)   # *4 because of concatenation
        
        self.upconv1 = nn.ConvTranspose2d(FILTERS_ROOT*2, FILTERS_ROOT, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(FILTERS_ROOT*2, FILTERS_ROOT)     # *2 because of concatenation
        
	# output
        self.outconv = nn.Conv2d(FILTERS_ROOT, MASK_CHANNELS, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        # output
        out = self.outconv(d1)
        return torch.sigmoid(out)