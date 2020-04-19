import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.initializers import orthogonal as initializer
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate, Dense,
                                     Embedding, GaussianDropout, GlobalAveragePooling1D,
                                     Input, Multiply, Softmax)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow_addons import options as tfa_options
from tensorflow_addons.layers import GELU
from tensorflow_addons.optimizers import LAMB as OPTIMIZER
import numpy as np
from . import config_object, utils
from .generate_characters import GeneratorCallback


class CharNet:
    """
    A deep, dense neural network based on the attention mechanism. Its shape is similar
    to that of a transformer without decoder layers.
    """

    def _init(self):
        inp = Input(shape=(self.config.inputs,))
        layer = Embedding(256, self.config.classes)(
                inp) if self.config.embedding else inp
        if self.config.input_dropout:
            layer = GaussianDropout(self.config.input_dropout)(layer)

        for neurons, depth in zip(self.config.neuron_list, self.config.block_depth):
            def dense(in_layer):
                """
                Creates a new dense layer using keras' dense function. The
                parameters
                used
                to create it are given in the parent function call.
                This function exists as the initializer and the regularizer
                are both
                classes
                which have to be freshly instantiated when creating a new layer.
                :param in_layer: layer the new dense layer will be attached
                to in graph
                :return: dense layer
                """
                return Dense(neurons,
                             kernel_initializer=initializer())(in_layer)

            prev_in = layer
            for _ in range(depth):
                key_layer = dense(layer)
                query_layer = dense(layer)
                value_layer = GELU()(key_layer)
                value_layer = dense(value_layer)
                value_layer = Softmax(axis=-1 - self.config.class_neurons)(value_layer)
                key_layer = Multiply()([key_layer, value_layer])
                layer = Add()([query_layer, key_layer])
                layer = BatchNormalization(axis=1)(layer)
                layer = GaussianDropout(self.config.dropout)(layer)
                layer = GELU()(layer)
            layer = Concatenate(axis=-1)([prev_in, layer])

        if self.config.class_neurons:
            layer = GlobalAveragePooling1D()(layer)
        layer = Dense(units=256, activation=self.config.output_activation,
                      kernel_initializer=initializer())(layer)

        self.model = Model(inputs=[inp], outputs=[layer])
        self.model.compile(loss=self.config.loss,
                           optimizer=OPTIMIZER(lr=self.config.learning_rate,
                                               weight_decay_rate=1e-3),
                           metrics=self.config.metrics)
        self.model.summary()
        data = (np.arange(self.config.inputs).reshape(1, self.config.inputs),
                np.ones((1, 1)))
        # Freeze the model graph for improved performance and reduced RAM usage
        self.model.train_on_batch(*data)
        self.model.predict_on_batch(data)
        tf.compat.v1.get_default_graph().finalize()

    def __init__(self, config=None, config_file_path=None):
        self.model = None
        self.dtype = 'uint8'
        if config_file_path is not None:
            import json
            with open(config_file_path, 'r') as configFile:
                config = configFile.read()
            config = json.loads(config)
        self.config = config_object.CharNetConfig(config)
        if self.config.load_model:
            self.load()
        else:
            self.new()

    def plot(self, filename='model.png'):
        """
        Method to plot an existing model to disk once it's in RAM.
        It requires graphviz and pydot to be installed.
        :param filename: Name of the file the model will be plotted to
        :return: None
        """
        plot_model(self.model, to_file=filename)

    def new(self):
        """
        This is a wrapper around the ._init function that creates a new model, so that
        hardware accelerators can be used as efficiently as possible. Unfortunately
        there currently are problems with distributed training, meaning that the wrapper
        has to call the function, without "distributing" the model to a single GPU.
        Currently we simply assume that tensorflow will pick the correct device (which
        is not the case for TPUs).
        :return:
        """
        self._init()

    def load(self):
        """
        Function which can be used to restore the previous state of the model.
        Note that this does not restore the class itself, implying that you have to
        save your configuration somewhere else. It is not attached to the model.
        :return: None
        """
        utils.get_previous_weights_from_gdrive(self.config.model_folder)
        last_used_model = utils.get_latest_model_name(self.config.model_folder)
        self.model = load_model(last_used_model)
        self.model.summary()

    def train(self, dataset=None, epochs=2, verbose=1, workers=1):
        """
        Basic training API that wraps keras training api with the previously created
        config. While easier to use, it's also much more simple.
        :param dataset: Path to dataset or numpy array
        :param epochs: Number of epochs to train for. Training can be continued.
        :param verbose: Keras verbosity level
        :param workers: Number of processes used to generate dataset. Should be equal
                        to number of threads.
        :return: None
        """
        dataset = utils.prepare_dataset(dataset,
                                        self.config.batch_size,
                                        self.config.inputs,
                                        self.dtype)
        callbacks = [ModelCheckpoint(os.path.join(self.config.model_folder,
                                                  '{epoch:03d}.hdf5'),
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=False,
                                     save_weights_only=False,
                                     mode='auto'),
                     GeneratorCallback(self.config.test_string,
                                       self.config.inputs,
                                       self.config.generated_characters,
                                       self.dtype)
                     ]
        for i in range(epochs):
            self.model.fit(dataset,
                           initial_epoch=i,
                           epochs=i + 1,
                           verbose=verbose,
                           use_multiprocessing=True,
                           workers=workers,
                           callbacks=callbacks)
