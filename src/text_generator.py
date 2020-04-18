import numpy as np
from tensorflow.keras.utils import Sequence


class SlidingWindowGenerator(Sequence):
    """
    Efficient dataset generator. Implements a sliding window over any given dataset.
    Using a generator like this is especially useful if the dataset wouldn't fit in
    memory otherwise.
    When using a poor cpu with a strong accelerator, such as a TPU, it is possible that
    the generation pipeline is too slow, even though it's already highly optimized. When
    this happens, you should switch to a stronger CPU and increase the number of workers
    during training.
    """
    def __init__(self, batch_size, dataset, inputs):
        if isinstance(dataset, bytes) or isinstance(dataset, bytearray):
            dataset = np.frombuffer(dataset, np.uint8).astype(np.int32)
        self.inputs = inputs
        self.batch_size = batch_size
        self.dataset = dataset
        self.dset_len = dataset.shape[0] - inputs
        self.output_indices = 0
        self.input_indices = 0
        self._set_indices()
        self.len = self.dset_len // self.batch_size - 1

    def _set_indices(self):
        batch_indices = np.arange(self.batch_size)
        self.output_indices = batch_indices + self.inputs
        self.input_indices = (batch_indices.reshape(-1, 1) +
                              np.arange(self.inputs).reshape(1, -1))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.dataset[self.input_indices + idx],
                self.dataset[self.output_indices + idx])

    def add_batch(self, batch):
        self.batch_size += batch
        self._set_indices()
        self.len = self.dset_len // self.batch_size - 1

    def on_epoch_end(self):
        # A more robust linear weight decay, see https://arxiv.org/pdf/1711.00489.pdf
        self.add_batch(self.batch_size)