import os
import wandb
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .dataloader import SICEDataset
from .loss_functions import *
from .model import DCENet, init_weights, EnhancedDCENet, DenoisingAutoencoder


class Trainer:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = None
        self.val_loader = None
        self.dae_train_loader = None
        self.dae_val_loader = None
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
    def load_data(self, image_path, image_size=256, batch_size=8, num_workers=4, val_split=0.1):

        if val_split < 0 or val_split > 1:
            raise ValueError("Validation split must be between 0 and 1")

        dataset = SICEDataset(img_files=image_path, image_size=image_size)

        dataset_size = len(dataset)
        print("Dataset size: ", len(dataset))
        val_size = int(val_split * dataset_size)
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

        # Load DAE data
        # conv4_features = self._extract_conv4_features(dataset)
        # noisy_features = self._add_noise(conv4_features)

        # dae_dataset = TensorDataset(noisy_features, conv4_features)
        # dae_train_dataset, dae_val_dataset = torch.utils.data.random_split(dae_dataset, [train_size, val_size])

        # self.dae_train_loader = DataLoader(
        #     dae_train_dataset,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     num_workers=num_workers,
        #     pin_memory=True
        # )

        # self.dae_val_loader = DataLoader(
        #     dae_val_dataset,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     num_workers=num_workers
        # )

    def load_custom_data(self, dataset, batch_size=8, num_workers=4, val_split=0.1):

        if dataset is None:
            raise ValueError("Please provide the datasets")

        if val_split < 0 or val_split > 1:
            raise ValueError("Validation split must be between 0 and 1")

        dataset_size = len(dataset)
        print("Dataset size: ", len(dataset))
        val_size = int(val_split * dataset_size)
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

    def load_noisy_data(self, dataset, batch_size=8, num_workers=4, val_split=0.1):

        if dataset is None:
            raise ValueError("Please provide the datasets")

        if val_split < 0 or val_split > 1:
            raise ValueError("Validation split must be between 0 and 1")

        dataset_size = len(dataset)
        print("Dataset size: ", len(dataset))
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        print("Train size: ", train_size)
        print("Validation size: ", val_size)

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        self.dae_train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        self.dae_val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    # Build the DCENet
    def build_model(self, pretrain_weights=None, learning_rate=0.0001, weight_decay=0.0001):
        self.model = DCENet().to(self.device)
        self.model.apply(init_weights)

        if pretrain_weights is not None:
            self.model.load_state_dict(torch.load(pretrain_weights))

        # define the loss functions
        self.color_loss = ColorConstancyLoss().to(self.device)
        self.exposure_loss = ExposureControlLoss(patch_size=16, mean_val=0.6).to(self.device)
        self.illumination_smoothing_loss = IlluminationSmoothnessLoss().to(self.device)
        self.spatial_consistency_loss = SpatialConsistencyLoss().to(self.device)

        # define the optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        if self.model is not None:
            print("Model built successfully")
        else:
            raise ValueError("Model not built")

    def build_dae(self, pretrain_weights=None):
        self.dae = DenoisingAutoencoder()
        self.dae.apply(init_weights)

        if pretrain_weights is not None:
            self.dae.load_state_dict(torch.load(pretrain_weights))

        if self.dae is not None:
            print("DAE built successfully")
        else:
            raise ValueError("DAE not built")

    # Build the enhanced DCENet
    def build_enhanced_model(self, pretrain_weights=None, learning_rate=0.0001, weight_decay=0.0001):
        dce_net = DCENet().load_state_dict(torch.load("models/checkpoints/model200_dark_faces.pth"))

        dae = DenoisingAutoencoder().load_state_dict(torch.load("models/checkpoints/dae_model_epoch50.pth"))

        self.enhanced_model = EnhancedDCENet(dce_net, dae).to(self.device)
        self.enhanced_model.apply(init_weights)

        if pretrain_weights is not None:
            self.enhanced_model(torch.load(pretrain_weights))

        # define the loss functions
        self.color_loss = ColorConstancyLoss().to(self.device)
        self.exposure_loss = ExposureControlLoss(patch_size=16, mean_val=0.6).to(self.device)
        self.illumination_smoothing_loss = IlluminationSmoothnessLoss().to(self.device)
        self.spatial_consistency_loss = SpatialConsistencyLoss().to(self.device)

        # define the optimizer
        self.optimizer = torch.optim.Adam(
            self.enhanced_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        if self.enhanced_model is not None:
            print("Enhanced model built successfully")
        else:
            raise ValueError("Enhanced model not built")


    def train(self, n_epochs=200, log_frequency=100, notebook=True):

        if self.model is None:
            raise ValueError("Model is not built")

        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}/{n_epochs}: ")

            # Training phase
            # Switch the model to training mode
            self.model.train()

            if notebook:
                from tqdm.notebook import tqdm as tqdm_notebook
                tqdm = tqdm_notebook

            train_loss = 0.0

            for batch_idx, lowlight_image in enumerate(tqdm(self.train_loader)):
                # Load the batch data low light image
                lowlight_image = lowlight_image.to(self.device)

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
                for batch_idx, lowlight_image in enumerate(tqdm(self.val_loader)):
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

    def train_dae(self, n_epochs=100, learning_rate=0.001, early_stopping=None, notebook=False):

        if self.dae is None:
            raise ValueError("Model is not built")

        wandb.watch(self.dae)

        criterion = nn.MSELoss(reduction='sum').to(self.device)
        optimizer = torch.optim.Adam(self.dae.parameters(), lr=learning_rate)

        # early stopping
        best_val_loss = float('inf')
        min_delta = 0.001
        plateau = 0

        for epoch in range(n_epochs):

            train_losses = 0.0
            print(f"Epoch {epoch+1}/{n_epochs}:")
            self.dae.train()

            for noisy_image, clean_image in self.dae_train_loader:
                noisy_image = noisy_image.to(self.device)
                clean_image = clean_image.to(self.device)

                reconstructed_image = self.dae(noisy_image)

                loss = criterion(reconstructed_image, clean_image)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses += loss.item()

            self.dae.eval()

            with torch.no_grad():

                val_losses = 0.0
                for noisy_image, clean_image in self.dae_val_loader:
                    noisy_image = noisy_image.to(self.device)
                    clean_image = clean_image.to(self.device)

                    reconstructed_image = self.dae(noisy_image)

                    loss = criterion(reconstructed_image, clean_image)

                    val_losses += loss.item()

            train_loss = train_losses / len(self.dae_train_loader)
            val_loss = val_losses / len(self.dae_val_loader)

            wandb.log({'Train Loss': train_loss, 'Validation Loss': val_loss})

            print(f"Epoch [{epoch+1}/{n_epochs}] - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                plateau = 0
            else:
                plateau += 1

            if early_stopping is not None and plateau >= early_stopping:
                print(f"Early stopping triggered at {epoch+1} epoch.")
                self.save_model(self.dae, save_path='dae_model.pth', save_dir='./models/checkpoints')
                break

            if (epoch+1) % 10 == 0:
                self.save_model(self.dae, save_path=f'dae_model_epoch_{epoch+1}.pth', save_dir='./models/checkpoints')


    def train_enhanced_model(self, n_epochs=200, log_frequency=100, notebook=True):

        if self.enhanced_model is None:
            raise ValueError("Enhanced model is not built")

        for epoch in range(n_epochs):

            self.enhanced_model.train()

            if notebook:
                from tqdm.notebook import tqdm as tqdm_notebook
                tqdm = tqdm_notebook

            train_losses = 0.0

            for batch_idx, low_light_image in enumerate(tqdm(self.train_loader)):
                low_light_image = low_light_image.to(self.device)

                enhanced_image_1, enhanced_image_final, A = self.enhanced_model(low_light_image)

                loss_col = 5 * torch.mean(self.color_loss(enhanced_image_final))
                loss_exp = 10 * torch.mean(self.exposure_loss(enhanced_image_final))
                loss_tv = 200 * torch.mean(self.illumination_smoothing_loss(A))
                loss_spa = torch.mean(self.spatial_consistency_loss(enhanced_image_final, low_light_image))

                total_loss = loss_col + loss_exp + loss_tv + loss_spa

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.enhanced_model.parameters(), 0.1)
                self.optimizer.step()

                train_losses += total_loss.item()

            self.enhanced_model.eval()

            val_losses = 0.0

            with torch.no_grad():
                for batch_idx, low_light_image in enumerate(tqdm(self.val_loader)):
                    low_light_image = low_light_image.to(self.device)

                    enhanced_image_1, enhanced_image_final, A = self.enhanced_model(low_light_image)

                    loss_col = 5 * torch.mean(self.color_loss(enhanced_image_final))
                    loss_exp = 10 * torch.mean(self.exposure_loss(enhanced_image_final))
                    loss_tv = 200 * torch.mean(self.illumination_smoothing_loss(A))
                    loss_spa = torch.mean(self.spatial_consistency_loss(enhanced_image_final, low_light_image))

                    total_loss = loss_col + loss_exp + loss_tv + loss_spa

                    val_losses += total_loss.item()

            if epoch % 20 == 0:
                train_losses = train_losses / len(self.train_loader)
                val_losses = val_losses / len(self.val_loader)
                print(f"Epoch {epoch} - Train Loss: {train_losses:.4f}, Validation Loss: {val_losses:.4f}")

                self.save_model(self.enhanced_model, save_path=f'enhanced_epoch_{epoch}.pth', save_dir='./models/checkpoints')

    def extract_conv4_features(self, dce_net, image_loader):

        dce_net.eval()
        features = []
        with torch.no_grad():
            for images in image_loader:
                images = images.to(self.device)
                x1 = dce_net.conv1(images)
                x2 = dce_net.conv2(x1)
                x3 = dce_net.conv3(x2)
                x4 = dce_net.conv4(x3)
                features.append(x4.cpu())
        return torch.cat(features)

    # Add synthetic noise to the image
    def add_noise(self, image, noise_factor=0.1):
        noisy_image = image + noise_factor * torch.randn_like(image)
        return torch.clip(noisy_image, 0, 1)

    def save_model(self, model, save_path, save_dir="./models/checkpoints"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, save_path)

        torch.save(model.state_dict(), save_path)

    def evaluate(self, model, image_path):

        model.eval()

        with torch.no_grad():
            lowlight_image = Image.open(image_path).convert("RGB")
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.ToTensor()
            ])

            lowlight_image = transform(lowlight_image).unsqueeze(0).to(self.device)

            _, enhanced_image_final, _ = model(lowlight_image)

            enhanced_image_final = enhanced_image_final.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            return lowlight_image, enhanced_image_final