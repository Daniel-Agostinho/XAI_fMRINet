# Built-in imports
import os
import csv
import pickle

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import torch

# My imports
from .tools import make_dir


class HCPLogger:
    def __init__(self, test_fold, test_subjects, network, experiment_id, atlas):

        self.main_folder = os.path.join(
            "Results",
            "Classification",
            network,
            f"{atlas}_{experiment_id}",
        )

        self.test_folder = os.path.join(
            self.main_folder,
            f"Test_{test_fold}",
        )

        make_dir(self.test_folder)
        self.save_subjects_id(test_subjects)
        self.train_fold = None

    def start_train_log(self, fold):

        self.train_fold = os.path.join(
            self.test_folder,
            f"Fold_{fold}"
        )

        make_dir(self.train_fold)

    def save_subjects_id(self, subjects):

        file_fpath = os.path.join(
            self.test_folder,
            "subjects.txt",
        )

        with open(file_fpath, "a") as file:
            for subject in subjects:
                file.write(f"{subject}\n")

    def log_train(self, data):

        file_name = os.path.join(
            self.train_fold,
            "training_summary.csv"
        )

        file_exists = os.path.isfile(file_name)

        with open(file_name, "a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
            writer.writerow(data)

    def log_test(self, model, cm, summary_log, summary_data, disp):

        model_fpath = os.path.join(
            self.test_folder,
            "test_model",
        )

        torch.save(model.state_dict(), model_fpath)

        cm_file = os.path.join(
            self.test_folder,
            "cm.npz"
        )

        np.savez_compressed(cm_file, cm=cm)

        summary_file = os.path.join(
            self.test_folder,
            "summary.txt"
        )

        with open(summary_file, "a") as file:
            file.write(summary_log)

        summary_data_file = os.path.join(
            self.test_folder,
            "summary.pkl"
        )

        with open(summary_data_file, "wb") as file:
            pickle.dump(summary_data, file)

        display_file = os.path.join(
            self.test_folder,
            "confusion_matrix",
        )

        fig = plt.figure(figsize=(19.20, 10.80))
        disp.plot(xticks_rotation='vertical')
        plt.savefig(display_file)

    def save_model(self, model):
        print("Model Saved!")

        model_path = os.path.join(
            self.train_fold,
            "final_model",
        )

        torch.save(model, model_path)

    def get_best_model(self, fold):

        model_path = os.path.join(
            self.test_folder,
            f"Fold_{fold}",
            "final_model",
        )

        return model_path


if __name__ == '__main__':
    pass
