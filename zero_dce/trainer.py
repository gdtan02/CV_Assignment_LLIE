import os
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from .model import DCENet, init_weights, EnhancedDCENet, DenoisingAutoencoder
from .dataloader import SICEDataset
from .loss_functions import *

class Trainer:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.dae = None
        self.enhanced_model = None
        self.color_loss = None
        self.exposure_loss = None
        self.illumination_smoothing_loss = None
        self.spatial_consistency_loss = None
        self.optimizer = None

    # Build the dataloader for training
    # image_path specifies the list of image files
    def load_data(self, image_path, image_size=256, batch_size=8, num_workers=4):
        dataset = SICEDataset(img_files=image_path, image_size=image_size)

        print("Dataset: ", dataset)
        dataset_size = len(dataset)
        print("Dataset size: ", len(dataset))
        val_size = int(0.1 * dataset_size)
        train_size = dataset_size - val_size
        print("Train size: ", train_size)
        print("Validation size: ", val_size)

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    # Build the DCENet
    def build_model(self, pretrain_weights=None):
        self.model = DCENet().cuda()
        self.model.apply(init_weights)

        if pretrain_weights is not None:
            self.load_weights(pretrain_weights)

    def build_dae(self, pretrain_weights=None):
        self.dae = DenoisingAutoencoder().cuda()
        self.dae.apply(init_weights)

        if pretrain_weights is not None:
            self.dae.load_state_dict(torch.load(pretrain_weights))

    # Build the enhanced DCENet
    def build_enhanced_model(self, pretrain_weights=None):
        dce_net = self.model if self.model is not None else self.build_model()
        dae = self.dae if self.dae is not None else self.build_dae()

        self.enhanced_model = EnhancedDCENet(dce_net, dae).cuda()
        self.enhanced_model.apply(init_weights)

        if pretrain_weights is not None:
            self.enhanced_model(torch.load(pretrain_weights))

    def compile(self, pretrain_weights=None, learning_rate=0.0001, weight_decay=0.0001):
        # build the model
        self.build_model(pretrain_weights=pretrain_weights)

        # define the loss functions
        self.color_loss = ColorConstancyLoss().cuda()
        self.exposure_loss = ExposureControlLoss(patch_size=16, mean_val=0.6).cuda()
        self.illumination_smoothing_loss = IlluminationSmoothnessLoss().cuda()
        self.spatial_consistency_loss = SpatialConsistencyLoss().cuda()

        # define the optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def train(self, n_epochs=200, log_frequency=100, notebook=True):

        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}/{n_epochs}: ")

            # Training phase
            # Switch the model to training mode
            self.model.train()

            if notebook:
                from tqdm.notebook import tqdm as tqdm_notebook
                tqdm = tqdm_notebook

            train_loss = 0.0

            for batch, lowlight_image in enumerate(tqdm(self.train_loader)):
                # Load the batch data low light image
                lowlight_image = lowlight_image.cuda()

                # Forward pass
                # A = curve parameter tensor
                enhanced_image_1, enhanced_image_final, A = self.model(lowlight_image)

                # Compute loss
                # Weights for each loss function:
                # color_loss: 5, exposure_loss: 10, illumination_smoothing_loss: 200, spatial_consistency_loss: 1
                loss_col = 5 * torch.mean(self.color_loss(enhanced_image_final))
                loss_exp = 10 * torch.mean(self.exposure_loss(enhanced_image_final))
                loss_tv = 200 * torch.mean(self.illumination_smoothing_loss(A))
                loss_spa = torch.mean(self.spatial_consistency_loss(enhanced_image_final, lowlight_image))

                total_loss = loss_col + loss_exp + loss_tv + loss_spa

                # Backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)  # Clip gradients to avoid instability
                self.optimizer.step()

                train_loss += total_loss.item()

            # Validation phase
            self.model.eval()

            val_loss = 0.0
            with torch.no_grad():
                for batch, lowlight_image in enumerate(tqdm(self.val_loader)):
                    lowlight_image = lowlight_image.to(self.device)

                    enhanced_image_1, enhanced_image_final, A = self.model(lowlight_image)

                    loss_col = 5 * torch.mean(self.color_loss(enhanced_image_final))
                    loss_exp = 10 * torch.mean(self.exposure_loss(enhanced_image_final))
                    loss_tv = 200 * torch.mean(self.illumination_smoothing_loss(A))
                    loss_spa = torch.mean(self.spatial_consistency_loss(enhanced_image_final, lowlight_image))

                    total_loss = loss_col + loss_exp + loss_tv + loss_spa

                    val_loss += total_loss.item()

            if epoch % 20 == 0:
                train_loss = train_loss / len(self.train_loader)
                val_loss = val_loss / len(self.val_loader)
                print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

                # Save the model checkpoints every 20 epochs
                self.save_model(save_path=f'epoch_{epoch}.pth', save_dir='./models/checkpoints')

    # Load the weights from checkpoint / pretrained models
    def load_weights(self, weights_path):
        self.model.load_state_dict(torch.load(weights_path))

    def save_model(self, save_path, save_dir="./models/checkpoints"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, save_path)

        torch.save(self.model.state_dict(), save_path)

    def inference(self, image_path):

        self.model.eval()

        with torch.no_grad():
            lowlight_image = Image.open(image_path).convert("RGB")
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.ToTensor()
            ])

            lowlight_image = transform(lowlight_image).unsqueeze(0).cuda()

            _, enhanced_image_final, _ = self.model(lowlight_image)

            enhanced_image_final = enhanced_image_final.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            return lowlight_image, enhanced_image_final