import numpy as np
from tensorflow.keras.utils import Sequence


class Generator(Sequence):
    def __init__(self, batch_size, dataset, outputs, index_in, inputs, steps,
                 change_per_keras_epoch, embedding,
                 base_batch=None):
        self.inputs = inputs
        self.base_batch = batch_size if base_batch is None else base_batch
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
        idx %= self.len
        return (self.dataset[self.input_indices + idx],
                self.dataset[self.output_indices + idx])

    def add_batch(self, batch):
        self.batch_size += batch
        self._set_indices()
        self.len = self.dset_len // self.batch_size - 1

    def on_epoch_end(self):
        # A more robust linear weight decay, see https://arxiv.org/pdf/1711.00489.pdf
        self.add_batch(self.batch_size)