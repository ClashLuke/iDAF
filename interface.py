import CharNet.mlp.generateCharacters as generateCharacters
import CharNet.mlp.textGenerator as textGenerator
import CharNet.mlp.modelCreator as modelCreator
import CharNet.mlp.utils as utils

import keras

class charnet():
  defaultConfig = {'leakyRelu':False, 'batchNorm':True, 'trainNewModel':True,
             'concatPreviousLayers':True, 'repeatInput':True, 'unroll':True,
             'splitInputs':False, 'initialLSTM':False,'inputDense':False,
             'splitLayer':False, 'concatDense':True, 'bidirectional':True,
             'concatBeforeOutput':True, 'drawModel':True, 'gpu':True, 
             'neuronList':None, 'indexIn':False, 'classNeurons':True,
             'inputs':60, 'neuronsPerLayer':120, 'layerCount':4, 'epochs': 1,
             'kerasEpochsPerEpoch': 256, 'learningRate':0.005, 'outputs':1,
             'dropout':0.35, 'batchSize': 1024, 'valSplit':0.1, 'verbose': 1,
             'outCharCount':512, 'changePerKerasEpoch': 0.25,
             'activation':'gelu', 'weightFolderName':'MLP_Weights',
             'testString':None, 'charSet': None}
  model = None

  def __init__(self, config=None, configFilePath=None):
    if configFilePath is not None:
      import json
      with open(configFilePath,'r') as configFile:
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
      chars, _, _, _ = utils.getCharacterVars(self.defaultConfig['indexIn'],self.defaultConfig['charSet'])
      print("WARNING: if your dataset is larger than 1GB and you have less than 8GiB of available RAM, you will receive a memory error.")
      datasetString = utils.reformatString(datasetString, chars)
    return datasetString

  def getModel(self):
    self.model = modelCreator.getModel(**self.defaultConfig)

  def getDatasetFromGDrive(self, datasetFileName):
    utils.mountDrive()
    utils.getDatasetFromGDrive(datasetFileName)

  def train(self, datasetFilePath=None, datasetString=None):
    if datasetFilePath is not None:
      with open(datasetFilePath, 'r', errors='ignore') as datasetFile:
        datasetString = datasetFile.read()
    if datasetString is None:
      print("FATAL: No dataset given. Exiting.")
      return None
    chars, charDict, charDictList, classes = utils.getCharacterVars(self.defaultConfig['indexIn'],self.defaultConfig['charSet'])

    self.defaultConfig['classes'] = classes
    self.defaultConfig['steps'] = int(len(datasetString)/self.defaultConfig['batchSize']/self.defaultConfig['kerasEpochsPerEpoch'])

    generateCharsInstance = generateCharacters.generateChars(
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
                      self.defaultConfig['valSplit'],
                      self.defaultConfig['changePerKerasEpoch'])
    inputGenerator = gen.inpGenerator()
    outputGenerator = gen.outGenerator()
    self.model.fit_generator(inputGenerator,
                    epochs=self.defaultConfig['epochs']*self.defaultConfig['kerasEpochsPerEpoch'],
                    verbose=self.defaultConfig['verbose'],
                    max_queue_size=2,
                    use_multiprocessing=True,
                    steps_per_epoch=self.defaultConfig['steps'],
                    callbacks=[
    keras.callbacks.ModelCheckpoint('gdrive/My Drive/'+self.defaultConfig['weightFolderName']+'/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False),
    generateCharacters.GenerateCharsCallback(generateCharsInstance,self.defaultConfig['testString'],self.defaultConfig['inputs'])               
                              ],
                   validation_data=outputGenerator,
                   validation_steps=self.defaultConfig['changePerKerasEpoch']*0.01)

  def run(self, datasetFilePath=None, datasetString=None, prepareText=True, fromGDrive=False):
    if fromGDrive and datasetFilePath is not None:
      self.getDatasetFromGDrive(datasetFilePath)
    datasetString = self.prepareText(datasetFilePath, datasetString, prepareText)
    self.getModel()
    self.train(datasetString=datasetString)
