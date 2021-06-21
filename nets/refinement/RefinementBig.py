import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))

class HourglassRefinement(nn.Module):
    """Height and width must be divisible by 16"""

    def __init__(self):
        super(HourglassRefinement, self).__init__()

        # mask conv
        self.mask_conv1 = BasicConv(1, 16, kernel_size=3, stride=3, dilation=3, padding=2)
        self.mask_conv2 = BasicConv(16, 1, kernel_size=3, stride=3, dilation=3, padding=2)
        
        # depth conv
        self.conv1 = conv2d(2, 16)

        self.conv_start = BasicConv(16, 64, kernel_size=1)

        self.conv1a = BasicConv(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(256, 512, kernel_size=3, stride=2, dilation=2, padding=2)
        self.conv4a = BasicConv(512, 1024, kernel_size=3, stride=2, dilation=2, padding=2)

        self.deconv4a = Conv2x(1024, 512, deconv=True)
        self.deconv3a = Conv2x(512, 256, deconv=True)
        self.deconv2a = Conv2x(256, 128, deconv=True)
        self.deconv1a = Conv2x(128, 64, deconv=True)

        self.conv1b = Conv2x(64, 128)
        self.conv2b = Conv2x(128, 256)
        self.conv3b = Conv2x(256, 512)
        self.conv4b = Conv2x(512, 1024)

        self.deconv4b = Conv2x(1024, 512, deconv=True)
        self.deconv3b = Conv2x(512, 256, deconv=True)
        self.deconv2b = Conv2x(256, 128, deconv=True)
        self.deconv1b = Conv2x(128, 64, deconv=True)

        self.final_conv = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, depth, mask):
        B = depth.shape[0]
        mask = self.mask_conv1(mask)
        mask = self.mask_conv2(mask)
        mask = mask.repeat(B,1,1,1) # [B, 1, H, W]
        
        x = torch.cat((depth, mask), dim=1)
        
        conv1 = self.conv1(x)  # [B, 16, H, W]
        x = self.conv_start(conv1)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        
        x = self.deconv4a(x, rem3)
        rem3 = x
        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)  # [B, 32, H, W]

        residual_depth = self.final_conv(x)  # [B, 1, H, W]

        depth = F.relu(depth + residual_depth, inplace=True)  # [B, 1, H, W]
        return depth
    
class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True,
                 mdconv=False):
        super(Conv2x, self).__init__()
        self.concat = concat

        if deconv and is_3d:
            kernel = (3, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel,
                               stride=2, padding=1)

        if self.concat:
            self.conv2 = BasicConv(out_channels * 2, out_channels, False, is_3d, bn, relu, kernel_size=3,
                                    stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        assert (x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x
    
class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x