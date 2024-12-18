import os
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

    # Build the DCENet
    def build_model(self, pretrain_weights=None):
        self.model = DCENet().cuda()
        self.model.apply(init_weights)

        if pretrain_weights is not None:
            self.model.load_state_dict(torch.load(pretrain_weights))

        if self.model is not None:
            print("Model built successfully")
        else:
            raise ValueError("Model not built")

    def build_dae(self, pretrain_weights=None):
        self.dae = DenoisingAutoencoder().cuda()
        self.dae.apply(init_weights)

        if pretrain_weights is not None:
            self.dae.load_state_dict(torch.load(pretrain_weights))

        if self.dae is not None:
            print("DAE built successfully")
        else:
            raise ValueError("DAE not built")

    # Build the enhanced DCENet
    def build_enhanced_model(self, pretrain_weights=None):
        dce_net = self.model
        dae = self.dae

        self.enhanced_model = EnhancedDCENet(dce_net, dae).cuda()
        self.enhanced_model.apply(init_weights)

        if pretrain_weights is not None:
            self.enhanced_model(torch.load(pretrain_weights))

        if self.enhanced_model is not None:
            print("Enhanced model built successfully")
        else:
            raise ValueError("Enhanced model not built")

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

    def train_dae(self, dce_net, n_epochs=100, log_frequency=100, learning_rate=0.0001, weight_decay=0.0001, notebook=True):

        image_loader = DataLoader(self.train_loader.dataset, batch_size=32, shuffle=True)
        conv4_features = self._extract_conv4_features(dce_net, image_loader)
        noisy_features = self._add_noise(conv4_features)

        dataset = TensorDataset(noisy_features, conv4_features)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        dae_train_dataset, dae_val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        self.dae_train_loader = DataLoader(
            dae_train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self.dae_val_loader = DataLoader(
            dae_val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )

        optimizer = torch.optim.Adam(self.dae.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss(reduction="sum").cuda()

        for epoch in range(n_epochs):

            self.dae.train()

            if notebook:
                from tqdm.notebook import tqdm as tqdm_notebook
                tqdm = tqdm_notebook

            train_loss = 0.0

            for noisy_image, clean_image in self.dae_train_loader:
                noisy_image = noisy_image.cuda()
                clean_image = clean_image.cuda()

                # Forward pass
                denoised_image = self.dae(noisy_image)

                # Compute loss
                loss = criterion(denoised_image, clean_image).cuda()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            self.dae.eval()

            val_loss = 0.0

            with torch.no_grad():
                for noisy_image, clean_image in self.dae_val_loader:
                    noisy_image = noisy_image.cuda()
                    clean_image = clean_image.cuda()

                    denoised_image = self.dae(noisy_image)

                    loss = criterion(denoised_image, clean_image)

                    val_loss += loss.item()

            if epoch % 20 == 0:
                train_loss = train_loss / len(self.dae_train_loader)
                val_loss = val_loss / len(self.dae_val_loader)
                print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

                self.save_model(save_path=f'dae_epoch_{epoch}.pth', save_dir='./models/checkpoints')

    def train_enhanced_model(self, n_epochs=200, log_frequency=100, notebook=True):

        if self.enhanced_model is None:
            raise ValueError("Enhanced model is not built")

        for epoch in range(n_epochs):

            self.enhanced_model.train()

            if notebook:
                from tqdm.notebook import tqdm as tqdm_notebook
                tqdm = tqdm_notebook

            train_losses = 0.0

            for batch_idx, lowlight_image in enumerate(tqdm(self.train_loader)):
                low_light_image = low_light_image.cuda()

                enhanced_image_1, enhanced_image_final, A = self.enhanced_model(lowlight_image)

                loss_col = 5 * torch.mean(self.color_loss(enhanced_image_final))
                loss_exp = 10 * torch.mean(self.exposure_loss(enhanced_image_final))
                loss_tv = 200 * torch.mean(self.illumination_smoothing_loss(A))
                loss_spa = torch.mean(self.spatial_consistency_loss(enhanced_image_final, lowlight_image))

                total_loss = loss_col + loss_exp + loss_tv + loss_spa

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.enhanced_model.parameters(), 0.1)
                self.optimizer.step()

                train_losses += total_loss.item()

            self.enhanced_model.eval()

            val_losses = 0.0

            with torch.no_grad():
                for batch_idx, lowlight_image in enumerate(tqdm(self.val_loader)):
                    lowlight_image = lowlight_image.to(self.device)

                    enhanced_image_1, enhanced_image_final, A = self.enhanced_model(lowlight_image)

                    loss_col = 5 * torch.mean(self.color_loss(enhanced_image_final))
                    loss_exp = 10 * torch.mean(self.exposure_loss(enhanced_image_final))
                    loss_tv = 200 * torch.mean(self.illumination_smoothing_loss(A))
                    loss_spa = torch.mean(self.spatial_consistency_loss(enhanced_image_final, lowlight_image))

                    total_loss = loss_col + loss_exp + loss_tv + loss_spa

                    val_losses += total_loss.item()

            if epoch % 20 == 0:
                train_losses = train_losses / len(self.train_loader)
                val_losses = val_losses / len(self.val_loader)
                print(f"Epoch {epoch} - Train Loss: {train_losses:.4f}, Validation Loss: {val_losses:.4f}")

                self.save_model(save_path=f'enhanced_epoch_{epoch}.pth', save_dir='./models/checkpoints')


    def _extract_conv4_features(self, model, dataset):

        model.eval()

        features = []

        with torch.no_grad():
            for lowlight_image in tqdm(dataset):
                lowlight_image = lowlight_image.cuda()

                x1 = model.conv1(lowlight_image)
                x2 = model.conv2(x1)
                x3 = model.conv3(x2)
                x4 = model.conv4(x3)
                features.append(x4.cpu())

        return torch.cat(features)

    # Add synthetic noise to the image
    def _add_noise(self, image, noise_factor=0.1):
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

            lowlight_image = transform(lowlight_image).unsqueeze(0).cuda()

            _, enhanced_image_final, _ = model(lowlight_image)

            enhanced_image_final = enhanced_image_final.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            return lowlight_image, enhanced_image_final