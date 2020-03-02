import CharNet.mlp.utils as utils
import numpy as np
import tensorflow as tf
import os


DEPTH = 2  # To make use of the Universal Approximation Theorem

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def addAdvancedLayers(layer, leakyRelu, batchNorm):
    if batchNorm:
      layer = tf.keras.layers.BatchNormalization()(layer)
    if leakyRelu:
      layer = tf.keras.layers.LeakyReLU(0.2)(layer)
    return layer

def getInitialBinaryLayer(initialLSTM, gpu, bidirectional,
                          inputs, unroll, classes, dropout,
                          twoDimensional, embedding):
  if embedding:
    inp = tf.keras.layers.Input(shape=(inputs,))
    layer = tf.keras.layers.Embedding(input_dim=inputs, output_dim=classes)(inp)
  else:
    inp = tf.keras.layers.Input(shape=(inputs*classes,))
    layer = inp
  if initialLSTM:
    layer = tf.keras.layers.Reshape((inputs,classes))(layer)
    if gpu:
      if bidirectional:
        layer = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(classes, kernel_initializer=tf.keras.initializers.lecun_normal(), return_sequences=True))(layer)
      else:
        layer = tf.keras.layers.CuDNNLSTM(classes, kernel_initializer=tf.keras.initializers.lecun_normal(), return_sequences=True)(layer)
    else:
      if bidirectional:
        layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=classes, activation='hard_sigmoid',recurrent_activation='hard_sigmoid', 
                              kernel_initializer=tf.keras.initializers.lecun_normal(),
                              unroll=unroll, return_sequences=True))(layer)
      else:
        layer = tf.keras.layers.LSTM(units=classes, activation='hard_sigmoid',recurrent_activation='hard_sigmoid', 
                              kernel_initializer=tf.keras.initializers.lecun_normal(),
                              unroll=unroll, return_sequences=True)(layer)
    layer = tf.keras.layers.GaussianDropout(dropout)(layer)
    if not twoDimensional:
      layer = tf.keras.layers.Flatten()(layer)
  else:
    layer = tf.keras.layers.GaussianDropout(dropout)(layer)
  return layer, inp

def initialiseList(lenght, initValue, differingValuePosition, differingValue):
  initialList = [initValue]*lenght
  initialList[differingValuePosition] = differingValue
  return initialList

def getInputLayer(layer, inputs, activation, classes, inputDense):
  if inputDense:
    layer = tf.keras.layers.Dense(units=inputs, activation=activation, kernel_initializer=tf.keras.initializers.lecun_normal())(layer)
  return layer

def getHiddenLayers(layer, layerCount, neuronList, activation, leakyRelu, batchNorm, layerList, concatDense, twoDimensional):
  for i in range(layerCount-1):
    n = neuronList[i]
    for _ in range(DEPTH):
      if twoDimensional:
        layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=n, activation=activation, kernel_initializer=tf.keras.initializers.lecun_normal()))(layer)
      else:
        layer = tf.keras.layers.Dense(units=n, activation=activation, kernel_initializer=tf.keras.initializers.lecun_normal())(layer)
    layer = addAdvancedLayers(layer, leakyRelu, batchNorm)
    layerList.append(layer)
    if concatDense and len(layerList) > 1:
      layer = tf.keras.layers.concatenate(list(layerList))
  return layerList, layer

def getOutput(layer, concatBeforeOutput, layerList, outputs, classes, outputActivation, loss, twoDimensional):
  if concatBeforeOutput:
    layer = tf.keras.layers.concatenate(layerList+[layer])
  if twoDimensional:
    layer = tf.keras.layers.Flatten()(layer)
  if 'crossentropy' not in loss:
    classes = 1
  layer = tf.keras.layers.Dense(units=outputs*classes, activation=outputActivation, kernel_initializer=tf.keras.initializers.lecun_normal())(layer)

  if outputs > 1:
    layer = tf.keras.layers.Reshape((outputs,classes))(layer)
  return layer

def compileModel(inp, layer, learningRate, drawModel, loss, metric, modelCompile):
  model = tf.keras.Model(inputs=[inp],outputs=[layer])
  if modelCompile:
    model.compile(loss=loss, optimizer=tf.train.AdamOptimizer(learning_rate=learningRate), metrics=[metric])
    model.summary()
    if drawModel:
      tf.keras.utils.plot_model(model, to_file='model.png')
  return model

def getModel(leakyRelu=True, batchNorm=True, trainNewModel=True,
             concatPreviousLayers=True, repeatInput=True, unroll=True,
             initialLSTM=False, inputDense=False, concatDense=True,
             bidirectional=True, modelCompile=True,
             concatBeforeOutput=True, drawModel=True, gpu=True, 
             neuronList=None, indexIn=False, classNeurons=True,
             twoDimensional=True, embedding=False,
             inputs=60, neuronsPerLayer=120, layerCount=4,
             learningRate=0.005, classes=30, outputs=1, dropout=0.35,
             activation='gelu', weightFolderName='MLP_Weights', 
             outputActivation='softmax',loss='sparse_categorical_crossentropy',
             metric='sparse_categorical_accuracy',**kwargs):
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
      inp = tf.keras.layers.Input(shape=(inputs,))
      layer = tf.keras.layers.GaussianDropout(dropout)(inp)
    else:
      layer, inp = getInitialBinaryLayer(initialLSTM, gpu, bidirectional, inputs, unroll, classes, inputDense, twoDimensional, embedding)
    layer = getInputLayer(layer, inputs, activation, classes, inputDense)
    layer = addAdvancedLayers(layer, leakyRelu, batchNorm)
    if repeatInput:
      layerList.append(layer)
    # Hidden layer
    layerList, layer = getHiddenLayers(layer, layerCount, neuronList, activation, leakyRelu, batchNorm, layerList, concatDense, twoDimensional)
    # Output layer
    n = neuronList[-1]
    layer = tf.keras.layers.Dense(units=n, activation=activation, kernel_initializer=tf.keras.initializers.lecun_normal())(layer)
    layer = addAdvancedLayers(layer, leakyRelu, batchNorm)
    layer = getOutput(layer, concatBeforeOutput, layerList, outputs, classes, outputActivation, loss, twoDimensional)
    # Compiling and displaying model
    model = compileModel(inp, layer, learningRate, drawModel, loss, metric, modelCompile)
  else:
    utils.getPreviousWeightsFromGDrive(weightFolderName)
    lastUsedModel = utils.getLatestModelName(weightFolderName)
    model = tf.keras.models.load_model(lastUsedModel,custom_objects={'gelu':gelu})
    model.summary()
  return model
