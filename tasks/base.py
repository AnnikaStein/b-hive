from rich.console import Console
import luigi
import torch
import law
import os

c = Console()

# Creating a dictionary to store hyperparameters
config_dict = {}
config_dict["model"] = {}

# Defining the number of input parameters
config_dict["model"]["n_cpf"] = 26
config_dict["model"]["n_npf"] = 25
config_dict["model"]["n_vtx"] = 5


class BaseTask(law.Task):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        c.print("[black on yellow]Warning:", "No CUDA device available. Running on cpu...")

    debug = luigi.BoolParameter(
        default=False,
        description="Debug Flag to test things. Functionality needs to be implemented for each task",
    )

    def local_path(self, *path):
        parts = [str(p) for p in self.store_parts() + path]
        # DATA_PATH is defined in setup.sh
        return os.path.join(os.environ["DATA_PATH"], *parts)

    def local_target(self, *path, **kwargs):
        return law.LocalFileTarget(self.local_path(*path), **kwargs)

    def store_parts(self):
        """
        This function parses arguments into a path
        """
        parts = (self.__class__.__name__,)
        if self.debug:
            parts += ("debug",)
        return parts
