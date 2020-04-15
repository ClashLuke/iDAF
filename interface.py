import itertools

import numpy as np
import tensorflow as tf

from .mlp import generate_characters, modelCreator, textGenerator, utils


class CharNet:
    def __init__(self, config=None, configFilePath=None):
        self.defaultConfig = {'leakyRelu':            False,
                              'batchNorm':            True,
                              'trainNewModel':        True,
                              'concatPreviousLayers': True,
                              'repeatInput':          True,
                              'unroll':               True,
                              'splitInputs':          False,
                              'initialLSTM':          False,
                              'inputDense':           False,
                              'splitLayer':           False,
                              'concatDense':          True,
                              'bidirectional':        True,
                              'concatBeforeOutput':   True,
                              'drawModel':            True,
                              'gpu':                  True,
                              'neuronList':           None,
                              'indexIn':              False,
                              'classNeurons':         True,
                              'decodeOutput':         True,
                              'tpu':                  False,
                              'twoDimensional':       False,
                              'embedding':            False,
                              'inputs':               60,
                              'neuronsPerLayer':      120,
                              'layerCount':           4,
                              'epochs':               1,
                              'kerasEpochsPerEpoch':  256,
                              'learningRate':         0.005,
                              'outputs':              1,
                              'dropout':              0.35,
                              'batchSize':            1024,
                              'valSplit':             0.1,
                              'verbose':              1,
                              'outCharCount':         512,
                              'changePerKerasEpoch':  0.25,
                              'steps':                1000,
                              'activation':           'gelu',
                              'weightFolderName':     'MLP_Weights',
                              'inputGenerator':       'text',
                              'loss':                 'sparse_categorical_crossentropy',
                              'outputActivation':     'softmax',
                              'metric':               'sparse_categorical_accuracy',
                              'testString':           None,
                              'charSet':              None
                              }
        self.model = None
        if configFilePath is not None:
            import json
            with open(configFilePath, 'r') as configFile:
                config = configFile.read()
            config = json.loads(config)
        if config is not None:
            for key, value in config.items():
                self.defaultConfig[key] = value
        else:
            print("No config found. Using default config.")
        if self.defaultConfig['charSet'] is None:
            self.defaultConfig['charSet'] = utils.getChars()
        if self.defaultConfig['testString'] is None:
            self.defaultConfig['testString'] = utils.getTestString()

    def prepareText(self, datasetFilePath=None, datasetString=None, prepareText=False):
        if datasetFilePath is not None:
            with open(datasetFilePath, 'r', errors='ignore') as datasetFile:
                datasetString = datasetFile.read()
        if datasetString is None:
            print("FATAL: No dataset given. Exiting.")
            exit()
        if prepareText:
            chars, _, _, _ = utils.getCharacterVars(self.defaultConfig['indexIn'],
                                                    self.defaultConfig['charSet'])
            print(
                    "WARNING: if your dataset is larger than 1GB and you have less "
                    "than "
                    "8GiB of available RAM, you will receive a memory error.")
            datasetString = utils.reformatString(datasetString, chars)
        return datasetString

    def getModel(self, modelCompile=True):
        if self.defaultConfig['tpu']:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                self.model = modelCreator.getModel(**self.defaultConfig,
                                                   modelCompile=modelCompile)
        else:
            self.model = modelCreator.getModel(**self.defaultConfig,
                                               modelCompile=modelCompile)

    def getDatasetFromGDrive(self, datasetFileName):
        utils.mountDrive()
        utils.getDatasetFromGDrive(datasetFileName)

    def train(self, datasetFilePath=None, datasetString=None):
        if self.defaultConfig['inputGenerator'] == 'text':
            if datasetFilePath is not None:
                with open(datasetFilePath, 'r', errors='ignore') as datasetFile:
                    datasetString = datasetFile.read()
            if datasetString is None:
                print("FATAL: No dataset given. Exiting.")
                return None
            self.defaultConfig['steps'] = int(
                    len(datasetString) / self.defaultConfig['batchSize'] /
                    self.defaultConfig['kerasEpochsPerEpoch'])

        chars, charDict, charDictList, classes = utils.getCharacterVars(
                self.defaultConfig['indexIn'] or self.defaultConfig['embedding'],
                self.defaultConfig['charSet'])

        self.defaultConfig['classes'] = classes

        generateCharsInstance = generate_characters.generateChars(
                self.defaultConfig['classes'],
                self.defaultConfig['inputs'],
                self.defaultConfig['testString'],
                self.defaultConfig['outCharCount'],
                self.defaultConfig['outputs'],
                chars,
                charDictList)
        gen = textGenerator.generator(self.defaultConfig['batchSize'],
                                      datasetString,
                                      self.defaultConfig['outputs'],
                                      self.defaultConfig['indexIn'],
                                      self.defaultConfig['inputs'],
                                      self.defaultConfig['steps'],
                                      charDictList,
                                      charDict,
                                      self.defaultConfig['classes'],
                                      self.defaultConfig['changePerKerasEpoch'],
                                      self.defaultConfig['embedding'])
        if self.defaultConfig['inputGenerator'] == 'text':
            inputGenerator = gen.inpGenerator()
        else:
            inputGenerator = self.defaultConfig['inputGenerator']
        if not self.defaultConfig['decodeOutput']:
            tmp = np.zeros(
                    (1, self.defaultConfig['classes'] * self.defaultConfig['inputs']))
            tmp[0][:] = list(itertools.chain.from_iterable(
                    [charDictList[self.defaultConfig['testString'][j]] for j in
                     range(self.defaultConfig['inputs'])]))
            self.defaultConfig['testString'] = tmp
        tfGenerator = utils.getTfGenerator(inputGenerator,
                                           self.defaultConfig['batchSize'],
                                           self.defaultConfig['outputs'])
        self.model.fit(tfGenerator,
                       epochs=self.defaultConfig['epochs'] * self.defaultConfig[
                           'kerasEpochsPerEpoch'],
                       verbose=self.defaultConfig['verbose'],
                       max_queue_size=2,
                       use_multiprocessing=True,
                       steps_per_epoch=self.defaultConfig['steps'],
                       callbacks=[
                               tf.keras.callbacks.ModelCheckpoint(
                                       'gdrive/My Drive/' + self.defaultConfig[
                                           'weightFolderName'] + '/weights.{'
                                                                 'epoch:02d}.hdf5',
                                       monitor='val_loss', verbose=1,
                                       save_best_only=False,
                                       save_weights_only=False, mode='auto', period=1),
                               generate_characters.GenerateCharsCallback(
                                       generateCharsInstance,
                                       self.defaultConfig['testString'],
                                       self.defaultConfig['inputs'],
                                       self.defaultConfig['decodeOutput'])
                               ])

    def run(self, datasetFilePath=None, datasetString=None, prepareText=True,
            fromGDrive=False):
        if fromGDrive and datasetFilePath is not None:
            self.getDatasetFromGDrive(datasetFilePath)
        datasetString = self.prepareText(datasetFilePath, datasetString, prepareText)
        self.getModel()
        self.train(datasetString=datasetString)
