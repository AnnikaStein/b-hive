import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import os

from rich.progress import track
from torch.utils.data import DataLoader

from tasks.base import BaseTask
from tasks.dataset import DatasetConstructorTask
from tasks.parameter_mixins import DatasetDependency, TrainingDependency
from tasks.training import InferenceTask, TrainingTask

from utils.plotting.roc import prepare_roc, plot_losses
from utils.torch.datasets import DeepJetDataset


class PlottingTask(TrainingDependency, DatasetDependency, BaseTask):
    def requires(self):
        return {
            "training": TrainingTask.req(self),
            "inference": InferenceTask.req(self),
            "dataset": DatasetConstructorTask.req(self),
        }

    def output(self):
        return self.local_target("loss.pdf")

    def run(self):
        os.makedirs(self.local_path(), exist_ok=True)
        files = np.array(
            open(self.input()["dataset"]["file_list"].path, "r").read().split("\n")[:-1]
        )

        print(self.input()["inference"])
        predictions = np.load(self.input()["inference"]["prediction"].path, allow_pickle=True)
        kinematics = np.load(self.input()["inference"]["kinematics"].path, allow_pickle=True)
        truth = np.load(self.input()["inference"]["truth"].path, allow_pickle=True)
        pts = kinematics[..., 0]

        sample_files = [
            os.path.join(self.input()["dataset"]["file_list"].parent.path, d)
            for d in os.listdir(self.input()["dataset"]["file_list"].parent.path)
            if d.endswith(".txt") and "test" in d
        ]
        samples_str_array = np.array([])
        for f in sample_files:
            samples_str_array = np.append(samples_str_array, open(f).read().split("\n")[:-2])

        prepare_roc(
            samples_str_array,
            self.local_path(),
            ["TT", "QCD"],
            truth,
            predictions,
            pts,
        )

        train_loss = np.load(self.input()["training"]["training_metrics"].path, allow_pickle=True)[
            "loss"
        ]
        validation_loss = np.load(
            self.input()["training"]["validation_metrics"].path, allow_pickle=True
        )["loss"]
        plot_losses(train_loss, validation_loss, self.local_path())
