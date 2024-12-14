import os
from glob import glob
from zero_dce import (
    DCENet,
    init_weights,
    Trainer,
    download_train_dataset, plot_result
)

if __name__ == '__main__':
    download_train_dataset()

    trainer = Trainer()

    image_path = os.path.join(os.getcwd(), "data", "Dataset_Part1", "**", "*.JPG")
    image_files = glob(image_path)
    trainer.load_data(image_files)

    trainer.build_model()

    trainer.compile(pretrain_weights=None, learning_rate=1e-4, weight_decay=1e-4)

    trainer.train(n_epochs=10, log_frequency=100)

    trainer.save_model(save_path="model_200.pth", save_dir="./models/checkpoints")

    for image_file in image_files[:5]:
        org_image, enhanced_image = trainer.inference(image_file)
        plot_result(org_image, enhanced_image)