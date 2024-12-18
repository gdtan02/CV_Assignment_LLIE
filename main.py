import os
from glob import glob
from zero_dce import (
    DCENet,
    init_weights,
    EnhancedDCENet,
    DenoisingAutoencoder,
    Trainer,
    download_train_dataset,
    plot_result
)

if __name__ == '__main__':
    download_train_dataset()

    trainer = Trainer()

    image_path = os.path.join(os.getcwd(), "data", "Dataset_Part1", "**", "*.JPG")
    image_files = glob(image_path)
    trainer.load_data(image_files)

    trainer.build_model(pretrain_weights="models/checkpoints/model200_dark_faces.pth")
    trainer.compile(pretrain_weights="models/checkpoints/model200_dark_faces.pth", learning_rate=1e-4, weight_decay=1e-4)

    trainer.build_dae()
    trainer.train_dae(trainer.model, n_epochs=20)

    # for image_file in image_files[:5]:
    #     org_image, enhanced_image = trainer.evaluate(image_file)
    #     plot_result(org_image, enhanced_image)