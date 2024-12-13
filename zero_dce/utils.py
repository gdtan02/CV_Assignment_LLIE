import os
import gdown
import rarfile

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


