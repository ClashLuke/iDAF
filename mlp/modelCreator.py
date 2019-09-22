import CharNet.mlp.utils as utils
import keras
import numpy as np
import tensorflow as tf


def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def addAdvancedLayers(layer, leakyRelu, batchNorm):
    if batchNorm:
      layer = keras.layers.BatchNormalization()(layer)
    if leakyRelu:
      layer = keras.layers.LeakyReLU(0.2)(layer)
    return layer

def getInitialBinaryLayer(initialLSTM, gpu, bidirectional,
                          inputs, unroll, classes, dropout):
  inp = keras.layers.Input(shape=(inputs*classes,))
  if initialLSTM:
    layer = keras.layers.Reshape((inputs,classes))(inp)
    if gpu:
      if bidirectional:
        layer = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(inputs, kernel_initializer=keras.initializers.lecun_normal(), return_sequences=True))(layer)
      else:
        layer = keras.layers.CuDNNLSTM(inputs, kernel_initializer=keras.initializers.lecun_normal(), return_sequences=True)(layer)
    else:
      if bidirectional:
        layer = keras.layers.Bidirectional(keras.layers.LSTM(units=inputs, activation='hard_sigmoid',recurrent_activation='hard_sigmoid', 
                              kernel_initializer=keras.initializers.lecun_normal(),
                              unroll=unroll, return_sequences=True))(layer)
      else:
        layer = keras.layers.LSTM(units=inputs, activation='hard_sigmoid',recurrent_activation='hard_sigmoid', 
                              kernel_initializer=keras.initializers.lecun_normal(),
                              unroll=unroll, return_sequences=True)(layer)
    layer = keras.layers.GaussianDropout(dropout)(layer)
    layer = keras.layers.Flatten()(layer)
  else:
    layer = keras.layers.GaussianDropout(dropout)(inp)
  return layer

def initialiseList(lenght, initValue, differingValuePosition, differingValue):
  initialList = [initValue]*lenght
  initialList[differingValuePosition] = differingValue
  return initialList

def splitLayer(layer, units, neurons, dimensions, splitAt, activation):
  end = [-1]*dimensions
  end[splitAt] = units
  inLayers = [
      keras.layers.Dense(units=units, activation=activation,
                         kernel_initializer=keras.initializers.lecun_normal())(
                           keras.layers.Lambda(lambda layer: keras.backend.slice(layer,
                                                                                initialiseList(dimensions, 0, splitAt, i),
                                                                                end)
                                                )(layer)
                            ) for i in range(0,neurons,units)]
  layer = keras.layers.concatenate(inLayers)
  return layer

def getInputLayer(splitInputs, layer, inputs, activation, classes, inputDense):
  if splitInputs:
      splitLayer(layer, classes, inputs, 3, 2, activation)
  elif inputDense:
    layer = keras.layers.Dense(units=inputs, activation=activation, kernel_initializer=keras.initializers.lecun_normal())(layer)
  return layer

def getHiddenLayers(layer, layerCount, neuronList, activation, leakyRelu, batchNorm, layerList, concatDense, splitLayer):
  for i in range(layerCount-1):
    n = neuronList[i]
    if splitLayer:
      inLayers = [keras.layers.Lambda( lambda layer: keras.backend.slice(layer, (0, i), (-1, 1)))(layer) for i in range(n)]
      inLayers = [keras.layers.Dense(units=1, activation=activation, kernel_initializer=keras.initializers.lecun_normal())(inLayers[i]) for i in range(n)]
      for i in range(n):
        layer = keras.layers.Dense(units=1, activation=activation, kernel_initializer=keras.initializers.lecun_normal())(keras.layers.concatenate([inLayers[i-1],inLayers[i]]))
        inLayers[i] = layer
      layer = keras.layers.concatenate(inLayers)
    else:
      layer = keras.layers.Dense(units=n, activation=activation, kernel_initializer=keras.initializers.lecun_normal())(layer)
    layer = addAdvancedLayers(layer, leakyRelu, batchNorm)
    layerList.append(layer)
    if concatDense:
      layer = keras.layers.concatenate(layerList)
  return layerList, layer

def getOutput(layer, concatBeforeOutput, layerList, outputs, classes):
  if concatBeforeOutput:
    layer = keras.layers.concatenate(layerList+[layer])
  layer = keras.layers.Dense(units=outputs*classes, activation='softmax', kernel_initializer=keras.initializers.lecun_normal())(layer)

  if outputs > 1:
    layer = keras.layers.Reshape((outputs,classes))(layer)
  return layer

def compileModel(inp, layer, learningRate, drawModel):
  model = keras.models.Model(inputs=[inp],outputs=[layer])
  model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(decay=2**-20,lr=learningRate), metrics=['sparse_categorical_accuracy'])
  model.summary()
  if drawModel:
    keras.utils.plot_model(model, to_file='model.png')
  return model

def getModel(leakyRelu=True, batchNorm=True, trainNewModel=True,
             concatPreviousLayers=True, repeatInput=True, unroll=True,
             splitInputs=False, initialLSTM=False,inputDense=False,
             splitLayer=False, concatDense=True, bidirectional=True,
             concatBeforeOutput=True, drawModel=True, gpu=True, 
             neuronList=None, indexIn=False, classNeurons=True,
             inputs=60, neuronsPerLayer=120, layerCount=4,
             learningRate=0.005, classes=30, outputs=1, dropout=0.35,
             activation='gelu', weightFolderName='MLP_Weights', **kwargs):
  if neuronList is None:
    neuronList = utils.getNeuronList(neuronsPerLayer,layerCount,classNeurons,classes)
  else:
    neuronsPerLayer = neuronList[0]
  if activation == 'gelu':
    activation = gelu

  if trainNewModel:
    layerList = []
    # Input layer
    if indexIn:
      inp = keras.layers.Input(shape=(inputs,))
      layer = keras.layers.GaussianDropout(dropout)(inp)
    else:
      layer = getInitialBinaryLayer(initialLSTM, gpu, bidirectional, inputs, unroll, classes, inputDense)
    layer = getInputLayer(splitInputs, layer, inputs, activation, classes, inputDense)
    layer = addAdvancedLayers(layer, leakyRelu, batchNorm)
    if repeatInput:
      layerList.append(layer)
    # Hidden layer
    layerList, layer = getHiddenLayers(layer, layerCount, neuronList, activation, leakyRelu, batchNorm, layerList, concatDense, splitLayer)
    # Output layer
    n = neuronList[-1]
    layer = keras.layers.Dense(units=n, activation=activation, kernel_initializer=keras.initializers.lecun_normal())(layer)
    layer = addAdvancedLayers(layer, leakyRelu, batchNorm)
    layer = getOutput(layer, concatBeforeOutput, layerList, outputs, classes)
    # Compiling and displaying model
    model = compileModel(inp, layer, learningRate, drawModel)
  else:
    utils.getPreviousWeightsFromGDrive(weightFolderName)
    lastUsedModel = utils.getLatestModelName(weightFolderName)
    model = keras.models.load_model(lastUsedModel,custom_objects={'gelu':gelu})
    model.summary()
  return model