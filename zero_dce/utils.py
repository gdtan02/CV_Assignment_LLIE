import os
import wandb
import torch
import torchvision
import gdown
import rarfile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

def download_train_dataset():

    if not os.path.exists("data"):
        os.makedirs("data")

    data_dir = os.path.join(os.getcwd(), "data")

    rar_path = os.path.join(data_dir, "Dataset_Part1.rar")
    print("RAR file path = ", rar_path)

    print(f"Downloading the SICE dataset to {rar_path}...")

    if os.path.exists(rar_path):
        print(f"File {rar_path} already exists.")
    else:
        gdown.download(
            url="https://drive.google.com/uc?id=1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN",
            output=str(rar_path),
            quiet=False
        )

        print("Unpack the SICE dataset...")
        try:
            rf = rarfile.RarFile(str(rar_path))
            print("rf=", rf)
            rf.extractall(data_dir)
        except Exception as e:
            print(f"Failed to unpack the SICE dataset: {e}")

    print("Done.")

def download_test_dataset():

    if not os.path.exists("data"):
        os.makedirs("data")

    data_dir = os.path.join(os.getcwd(), "data")

    dataset_path = os.path.join(data_dir, "DarkPair.zip")
    print("Dataset file path = ", dataset_path)

    print(f"Downloading the Dark Face dataset to {dataset_path}...")

    if os.path.exists(dataset_path):
        print(f"File {dataset_path} already exists.")
    else:
        gdown.download(
            url="https://drive.google.com/uc?id=11KaOhxcOh68_NyZwacBoabEJ6FgPCsnQ",
            output=str(dataset_path),
            quiet=False
        )

        print("Unzip the Dark Face dataset...")
        try:
            os.system(f"unzip {dataset_path} -d {data_dir}")
        except Exception as e:
            print(f"Failed to unzip the Dark Face dataset: {e}")

    print("Done.")

def plot_result(image, enhanced, notebook=False):

    if isinstance(image, torch.Tensor):
        image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
    if isinstance(enhanced, torch.Tensor):
        enhanced = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()

    fig = plt.figure(figsize=(12,12))
    fig.add_subplot(1,2,1).set_title('Original Image')
    _ = plt.imshow(image)
    fig.add_subplot(1,2,2).set_title('Enhanced Image')
    _ = plt.imshow(enhanced)

    if notebook:
        plt.show()
    else:
        if not os.path.exists("outputs"):
            os.makedirs("outputs")
        save_path = os.path.join(os.getcwd(), "outputs", "output.png")

        plt.savefig(save_path)  # Save the figure as a file

def add_noise(image, noise_type='gaussian', noise_factor=0.1):
    transform_to_tensor = torchvision.transforms.ToTensor()
    image = transform_to_tensor(image)

    # if noise_type == 'gaussian'
    noise = torch.randn(image.size()) * noise_factor
    noisy_image = image + noise

    return noisy_image

def init_wandb(project_name, experiment_name, api_key):
    if project_name is not None and experiment_name is not None and api_key is not None:
        os.environ["WANDB_API_KEY"] = api_key
        wandb.login(api_key)
        wandb.init(project_name, experiment_name)