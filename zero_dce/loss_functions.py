import torch
import torch.nn as nn
from torch.nn import functional as F

# Color Constancy Loss
# This loss function is used to correct the potential color deviations in enhanced image
# and build the relations among 3 adjusted channels.
class ColorConstancyLoss(nn.Module):

    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, x, y):
        # Compute the mean intensity for each RGB channel across spatial dimensions (H x W)
        # Shape of x tensor: (batch_size, color_channels=3, H, W)
        # Shape of mean_rgb tensor: (batch_size, color_channels=3, 1, 1)
        mean_rgb = torch.mean(x, dim=(2, 3), keepdim=True)

        # Split the mean_rgb tensor into 3 separate tensors for each RGB channel
        mr, mg, mb = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]

        # Calculate the squared difference between the mean intensity of different color pairs
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mg - mb, 2)

        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        return k

# Exposure Control Loss
# This loss function measures the distance between the average intensity value to a local region to the well-exposedness of E.
# The E is set to the gray level in the RGB color space, at 0.6.
# M represents the number of non-overlapping local regions of size 16x16.
class ExposureControlLoss(nn.Module):

    def __init__(self, patch_size, mean_val):
        super(ExposureControlLoss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.E = mean_val

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        mean = self.pool(x)

        return torch.mean(torch.pow(mean - torch.FloatTensor([self.E]).to(x.device), 2))

# Illumination Smoothness Loss
# This loss function is used to preserve the monotonicity relations between neighboring pixels
# Penalizes abrupt changes in illumination maps to encourage smooth transitions across spatial regions
class IlluminationSmoothnessLoss(nn.Module):

    def __init__(self, loss_weight=1):
        super(IlluminationSmoothnessLoss, self).__init__()

    # X shape: (batch_size, 24, H, W)
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]  # height
        w_x = x.size()[3]  # width

        # Calculate the total number of vertical and horizontal differences
        count_h = (h_x-1) * w_x
        count_w = h_x * (w_x-1)

        # Sum of squared vertical differences between adjacent pixels
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        # Sum of squared horizontal differences between adjacent pixels
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()

        return self.loss_weight * (h_tv / count_h + w_tv / count_w) / batch_size

# Spatial Consistency Loss
# This loss function is used to ensure the spatial consistency between the input and output images
# It measures the difference between the input and output images in terms of spatial structure
class SpatialConsistencyLoss(nn.Module):

    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()

        # unsqueeze 2 time to make it 2D tensors (H x W)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)

        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org_img, enhanced_img):
        org_mean = torch.mean(org_img, 1, keepdim=True)
        enhanced_mean = torch.mean(enhanced_img, 1, keepdim=True)
        org_pool = self.pool(org_mean)
        enhanced_pool = self.pool(enhanced_mean)

        # Applies convolution to calculate total intensity differences in all 4 directions for
        # both original and enhanced images
        d_org_left = F.conv2d(org_pool, self.weight_left, padding=1)
        d_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        d_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        d_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        d_enhanced_left = F.conv2d(enhanced_pool, self.weight_left, padding=1)
        d_enhanced_right = F.conv2d(enhanced_pool, self.weight_right, padding=1)
        d_enhanced_up = F.conv2d(enhanced_pool, self.weight_up, padding=1)
        d_enhanced_down = F.conv2d(enhanced_pool, self.weight_down, padding=1)

        # Calculate the mean squared difference between the original and enhanced images
        d_left = torch.pow(d_org_left - d_enhanced_left, 2)
        d_right = torch.pow(d_org_right - d_enhanced_right, 2)
        d_up = torch.pow(d_org_up - d_enhanced_up, 2)
        d_down = torch.pow(d_org_down - d_enhanced_down, 2)

        return d_left + d_right + d_up + d_down


