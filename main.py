import os
from glob import glob
from zero_dce import (
    DCENet,
    init_weights,
    EnhancedDCENet,
    DenoisingAutoencoder,
    Trainer,
    download_train_dataset,
    plot_result, NoisyImageDataset
)

if __name__ == '__main__':
    download_train_dataset()

    trainer = Trainer()

    image_path = os.path.join(os.getcwd(), "data", "Dataset_Part1", "**", "*.JPG")
    image_files = glob(image_path)

    # Load noisy data
    noisy_data = NoisyImageDataset(img_files=image_files)

    trainer.load_noisy_data(noisy_data, batch_size=8, num_workers=4)

    # trainer.load_data(image_files)

    # trainer.build_model(pretrain_weights="models/checkpoints/model200_dark_faces.pth")
    #
    trainer.build_dae()
    trainer.train_dae(n_epochs=10, learning_rate=0.001, early_stopping=5)

    # for image_file in image_files[:5]:
    #     org_image, enhanced_image = trainer.evaluate(image_file)
    #     plot_result(org_image, enhanced_image)