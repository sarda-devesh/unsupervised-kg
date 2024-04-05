import os
import json
from pathlib import Path
import argparse
import subprocess

def read_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_directory', type= str, required = True, help = "The directory containing the loaded dataset")
    parser.add_argument('--save_directory', type= str, required = True, help = "The directory where we want to save the model results to")
    return parser.parse_args()

def training(data_dir, save_dir):
    train_path = Path(data_dir) / "train.tsv"
    test_path = Path(data_dir) / "test.tsv"
    valid_path = Path(data_dir) / "valid.tsv"
    config_file = "training.jsonnet"

    #Run the train command
    dataset_size = len(Path(train_path).read_text().strip().split("\n"))
    command = (
        f"train_data_path={train_path} "
        f"valid_data_path={valid_path} "
        f"dataset_size={dataset_size} "
        f"allennlp train {config_file} "
        f"--serialization-dir {save_dir} "
        f"--include-package seq2rel "
        f"-f"
    )

    print("Running the command", command)

    subprocess.run(command, shell=True)

if __name__ == "__main__":
    args = read_args()
    training(args.data_directory, args.save_directory)