import torch
import torch.nn as nn

# Deep Curve Estimation Network (DCE-Net) model
class DCENet(nn.Module):

    def __init__(self, n_filters=32):
        super(DCENet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=n_filters, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv3 = nn.Conv2d(
            in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv4 = nn.Conv2d(
            in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv5 = nn.Conv2d(
            in_channels=2*n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv6 = nn.Conv2d(
            in_channels=2*n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv7 = nn.Conv2d(
            in_channels=2*n_filters, out_channels=24, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        x7 = self.conv7(torch.cat([x1, x6], dim=1))

        # Result tensor shape: (batch_size, 24, H, W)
        # Tanh activation function ensures the output curve parameter values range from -1 to 1
        x_r = torch.tanh(x7)

        # Split the result tensor into 8 residual maps (each with 3 RGB channels)
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, split_size_or_sections=3, dim=1)

        # Enhancement process to compute the light-enhancement curve (LE-curve) using residual maps
        x = x + r1 * (torch.pow(x,2) - x)
        x = x + r2 * (torch.pow(x,2) - x)
        x = x + r3 * (torch.pow(x,2) - x)
        enhanced_image_1 = x + r4 * (torch.pow(x,2) - x)
        x = enhanced_image_1 + r5 * (torch.pow(enhanced_image_1,2) - enhanced_image_1)
        x = x + r6 * (torch.pow(x,2) - x)
        x = x + r7 * (torch.pow(x,2) - x)
        enhanced_image_final = x + r8 * (torch.pow(x,2) - x)

        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], dim=1)
        return enhanced_image_1, enhanced_image_final, r
