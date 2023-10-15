import numpy as np
import torch
import lz4.frame
from rich.progress import track
from torch.utils.data import IterableDataset

class DeepJetDataset(IterableDataset):
    def __init__(
        self,
        files,
        data_type="training",
        weighted_sampling=False,
        device="cpu",
        histogram_training=None,
        verbose=0,
        compression = False,
    ):
        self.verbose = verbose
        self.files = files
        self.bins_pt = [15, 20, 26, 35, 46, 61, 80, 106, 141, 186, 247, 326, 432, 571, 756, 1001]

        self.bins_eta = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.51]

        self.Nedges = [0]
        self.compression = compression
        self.data_type = data_type
        if data_type == "validation":
            self.data_type = "test"
        all_number_of_samples = histogram_training.sum()
        if self.data_type == "test" or self.data_type == "validation":
            all_number_of_samples /= 2
        self.weighted_sampling = weighted_sampling
        if self.compression:
            with lz4.frame.open(self.files[0], mode='r') as fp:
                output_data = fp.read()
            s = np.frombuffer(output_data, dtype='float16')
            s = s[2:].reshape(-1, int(s[1]))
            self.dataset_size = s.shape
        else:
            self.dataset_size = np.load(self.files[0], mmap_mode="r").shape
        self.dataset_chunk_size = int(self.dataset_size[0])
        self.dataset_fts_size = int(self.dataset_size[1])
        self.Nedges = np.append(
            self.Nedges,
            list(range(0, int(all_number_of_samples), self.dataset_chunk_size)),
        )
        self.Nedges = np.append(self.Nedges, int((all_number_of_samples)))
        self.device = device

    def __len__(self):
        return self.Nedges[-1]

    def __getitem__(self, index):
        true_index_in_file = index % self.chunk_size  # index - self.Nedges[loc]
        loc = index // self.chunk_size
        element = np.load(self.files[loc])[true_index_in_file]
        element = torch.tensor(element).float()
        return torch.unsqueeze(element[:-2], dim=-1), element[-2], element[-1]

    def __iter__(self):
        # Multi-worker support:
        worker_info = torch.utils.data.get_worker_info()
        files_to_read = self.files
        if worker_info is not None:
            files_to_read = np.array_split(files_to_read, worker_info.num_workers)[worker_info.id]

        for file in files_to_read:
            if self.verbose:
                print(f"Loading {file}")
            if self.compression:
                with lz4.frame.open(file, mode='r') as fp:
                    output_data = fp.read()
                s = np.frombuffer(output_data, dtype='float16')
                s = s[2:].reshape(-1, int(s[1])).astype('float32')
            else:
                s = np.load(file)
            s1 = ~np.isnan(s).any(axis = 1)
            s2 = ~np.isinf(s).any(axis = 1)
            s = s[s1*s2]
            if self.weighted_sampling:
                random_number = np.random.rand(s.shape[0])
                goods = random_number < s[:, -2]
                s = s[goods]
            for si in s:
                yield np.expand_dims(si[:-2], axis=-1), si[-2], si[-1]
        return None

    def get_all_weights(self):
        weights = np.empty((self.Nedges[-1]))
        N = 0
        for file in track(self.files, "Reading in the weights for the " + self.data_type + " data"):
            data = np.load(file)
            n_elements = int(data.shape[0])
            weights[N : N + n_elements] = data[:, -2]
            N += n_elements
        return weights
