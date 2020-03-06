import CharNet.mlp.utils as utils
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import os

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
                              kernel_initializer=tf.keras.initializers.Orthogonal(),
                              unroll=unroll, return_sequences=True))(layer)
      else:
        layer = tf.keras.layers.LSTM(units=classes, activation='hard_sigmoid',recurrent_activation='hard_sigmoid', 
                              kernel_initializer=tf.keras.initializers.Orthogonal(),
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

def getInputLayer(layer, inputs, classes, inputDense):
  if inputDense:
    layer = tf.keras.layers.Dense(units=inputs, kernel_initializer=tf.keras.initializers.Orthogonal())(layer)
    layer = tfa.layers.GELU()(layer)
  return layer

def getHiddenLayers(layer, layerCount, neuronList, leakyRelu, batchNorm, concatDense, twoDimensional, dropout, depth):
  if twoDimensional:
      dense = lambda *x, **y: tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(*x, **y))
  else:
      dense = lambda *x, **y: tf.keras.layers.Dense(*x, **y)
  for i in range(layerCount-1):
    n = neuronList[i]
    prev_in = layer
    for _ in range(depth):
        key_layer = dense(n, kernel_initializer=tf.keras.initializers.Orthogonal())(layer)
        query_layer = dense(n, kernel_initializer=tf.keras.initializers.Orthogonal())(layer)
        value_layer = tfa.layers.GELU()(key_layer)
        value_layer = dense(n, kernel_initializer=tf.keras.initializers.Orthogonal())(value_layer)
        value_layer = tf.keras.layers.Softmax()(value_layer)
        key_layer = tf.keras.layers.Multiply()([key_layer, value_layer])
        layer = tf.keras.layers.Add()([query_layer, key_layer])
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
        layer = tf.keras.layers.GaussianDropout(dropout)(layer)
        layer = tfa.layers.GELU()(layer)
    if concatDense:
      layer = tf.keras.layers.Concatenate(axis=-1)([prev_in, layer])
  return layer

def getOutput(layer, concatBeforeOutput, outputs, classes, outputActivation, loss, twoDimensional):
  if twoDimensional:
    layer = tf.keras.layers.Flatten()(layer)
  if 'crossentropy' not in loss:
    classes = 1
  layer = tf.keras.layers.Dense(units=outputs*classes, activation=outputActivation, kernel_initializer=tf.keras.initializers.Orthogonal())(layer)

  if outputs > 1:
    layer = tf.keras.layers.Reshape((outputs,classes))(layer)
  return layer

def compileModel(inp, layer, learningRate, drawModel, loss, metric, modelCompile):
  model = tf.keras.Model(inputs=[inp],outputs=[layer])
  if modelCompile:
    model.compile(loss=loss, optimizer=tfa.optimizers.LAMB(lr=learningRate,weight_decay_rate=learningRate/100), metrics=[metric])
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
             weightFolderName='MLP_Weights', outputActivation='softmax',
             loss='sparse_categorical_crossentropy', metric='sparse_categorical_accuracy',
             depth=1, **kwargs):
  if len(kwargs) > 0:
    print(f"Unused Keyword Arguments: {kwargs}")
  if neuronList is None:
    neuronList = utils.getNeuronList(neuronsPerLayer,layerCount,classNeurons,classes)
  else:
    neuronsPerLayer = neuronList[0]

  if trainNewModel:
    # Input layer
    if indexIn:
      inp = tf.keras.layers.Input(shape=(inputs,))
      layer = tf.keras.layers.GaussianDropout(dropout)(inp)
    else:
      layer, inp = getInitialBinaryLayer(initialLSTM, gpu, bidirectional, inputs, unroll, classes, inputDense, twoDimensional, embedding)
    layer = tf.keras.layers.Dense(inputs,kernel_initializer=tf.keras.initializers.Orthogonal())(layer)
    layer = tf.keras.layers.BatchNorm(axis=-1)(layer)
    layer = tfa.layers.GELU()(layer)
    layer = getHiddenLayers(layer, layerCount, neuronList, leakyRelu, batchNorm, concatDense, twoDimensional, dropout, depth)
    layer = getOutput(layer, concatBeforeOutput, outputs, classes, outputActivation, loss, twoDimensional)
    # Compiling and displaying model
    model = compileModel(inp, layer, learningRate, drawModel, loss, metric, modelCompile)
  else:
    utils.getPreviousWeightsFromGDrive(weightFolderName)
    lastUsedModel = utils.getLatestModelName(weightFolderName)
    model = tf.keras.models.load_model(lastUsedModel,custom_objects={'gelu':gelu})
    model.summary()
  return model
