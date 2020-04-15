import itertools

import numpy as np
import tensorflow as tf

from .mlp import generate_characters, model_creator, text_generator, utils


class CharNet:
    def __init__(self, config=None, config_file_path=None):
        self.default_config = {'leaky_relu':             False,
                               'batch_norm':             True,
                               'trainNewModel':          True,
                               'concatPreviousLayers':   True,
                               'repeatInput':            True,
                               'unroll':                 True,
                               'splitInputs':            False,
                               'initial_lstm':           False,
                               'input_dense':            False,
                               'splitLayer':             False,
                               'concat_dense':           True,
                               'bidirectional':          True,
                               'concat_before_output':   True,
                               'draw_model':             True,
                               'gpu':                    True,
                               'neuron_list':            None,
                               'index_in':               False,
                               'class_neurons':          True,
                               'decode_output':          True,
                               'tpu':                    False,
                               'two_dimensional':        False,
                               'embedding':              False,
                               'inputs':                 10,
                               'neurons_per_layer':      20,
                               'layer_count':            4,
                               'epochs':                 1,
                               'kerasEpochsPerEpoch':    256,
                               'learning_rate':          0.005,
                               'outputs':                1,
                               'dropout':                0.35,
                               'batch_size':             16,
                               'valSplit':               0.1,
                               'verbose':                1,
                               'out_char_count':         512,
                               'change_per_keras_epoch': 0.25,
                               'steps':                  1000,
                               'activation':             'gelu',
                               'weight_folder_name':     'MLP_Weights',
                               'inputGenerator':         'text',
                               'loss':
                                                         'sparse_categorical_crossentropy',
                               'output_activation':      'softmax',
                               'metric':                 'sparse_categorical_accuracy',
                               'testString':             None,
                               }
        self.model = None
        if config_file_path is not None:
            import json
            with open(config_file_path, 'r') as configFile:
                config = configFile.read()
            config = json.loads(config)
        if config is not None:
            for key, value in config.items():
                self.default_config[key] = value
        else:
            print("No config found. Using default config.")
        if self.default_config['testString'] is None:
            self.default_config['testString'] = utils.get_test_string()

    def get_model(self, model_compile=True):
        if self.default_config['tpu']:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                self.model = model_creator.get_model(**self.default_config,
                                                     modelCompile=model_compile)
        else:
            self.model = model_creator.get_model(**self.default_config,
                                                 modelCompile=model_compile)

    @staticmethod
    def load_text(dataset_file_path=None, dataset_array=None):
        if dataset_file_path is not None:
            with open(dataset_file_path, 'rb') as dataset_file:
                dataset_array = dataset_file.read()
            dataset_array = np.frombuffer(dataset_array, dtype=np.uint8)
        if dataset_array is None:
            print("FATAL: No dataset given. Exiting.")
            exit()
        return dataset_array

    @staticmethod
    def get_dataset_from_gdrive(dataset_file_name):
        utils.mount_drive()
        utils.get_dataset_from_gdrive(dataset_file_name)

    def train(self, dataset_file_path=None, dataset_array=None):
        if self.default_config['inputGenerator'] == 'text':

            self.default_config['steps'] = int(
                    len(dataset_array) / self.default_config['batch_size'] /
                    self.default_config['kerasEpochsPerEpoch'])

        generate_chars_instance = generate_characters.GenerateChars(
                self.default_config['inputs'],
                self.default_config['testString'],
                self.default_config['out_char_count'],
                self.default_config['outputs'])
        if self.default_config['inputGenerator'] == 'text':
            input_generator = text_generator.Generator(
                    self.default_config['batch_size'],
                    dataset_array,
                    self.default_config['outputs'],
                    self.default_config['index_in'],
                    self.default_config['inputs'],
                    self.default_config['steps'],
                    self.default_config['change_per_keras_epoch'],
                    self.default_config['embedding'])
        else:
            input_generator = self.default_config['inputGenerator']
        self.model.fit(input_generator,
                       epochs=self.default_config['epochs'],
                       verbose=self.default_config['verbose'],
                       callbacks=[
                               tf.keras.callbacks.ModelCheckpoint(
                                       'gdrive/My Drive/' + self.default_config[
                                           'weight_folder_name'] + '/weights.{'
                                                                   'epoch:02d}.hdf5',
                                       monitor='val_loss', verbose=1,
                                       save_best_only=False,
                                       save_weights_only=False, mode='auto', period=1),
                               generate_characters.GenerateCharsCallback(
                                       generate_chars_instance,
                                       self.default_config['testString'],
                                       self.default_config['inputs'],
                                       self.default_config['decode_output'])
                               ])

    def run(self, datasetFilePath=None, dataset_array=None, prepareText=True,
            fromGDrive=False):
        if fromGDrive and datasetFilePath is not None:
            self.get_dataset_from_gdrive(datasetFilePath)
        dataset_array = self.load_text(datasetFilePath, dataset_array)
        self.get_model()
        self.train(dataset_array=dataset_array)
