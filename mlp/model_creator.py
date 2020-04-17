from tensorflow.keras import Model
from tensorflow.keras.initializers import orthogonal as initializer
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate, Dense,
                                     Embedding, Flatten, GaussianDropout,
                                     GlobalAveragePooling1D, Input, Multiply, Softmax)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow_addons.layers import GELU
from tensorflow_addons.optimizers import LAMB as OPTIMIZER
from tensorflow.keras.regularizers import L1L2

from . import utils


def add_advanced_layers(layer, leaky_relu, batch_norm):
    if batch_norm:
        layer = BatchNormalization()(layer)
    return layer


def get_initial_binary_layer(initial_lstm, gpu, bidirectional,
                             inputs, unroll, classes, dropout,
                             two_dimensional, embedding, class_neurons):
    inp = Input(shape=(inputs,))
    layer = Embedding(input_dim=256, output_dim=classes)(inp)
    if not class_neurons:
        layer = Flatten()(layer)
    if dropout:
        layer = GaussianDropout(dropout)(layer)
    return layer, inp


def initialise_list(length, init_value, differing_value_position, differing_value):
    initial_list = [init_value] * length
    initial_list[differing_value_position] = differing_value
    return initial_list


def get_hidden_layers(layer, layer_count, neuron_list, leaky_relu, batch_norm,
                      concat_dense, two_dimensional, dropout, depth, class_neurons):
    for neurons in neuron_list:
        def dense(in_layer):
            """
            Creates a new dense layer using keras' dense function. The parameters used
            to create it are given in the parent function call.
            This function exists as the initializer and the regularizer are both classes
            which have to be freshly instantiated when creating a new layer.
            :param in_layer: layer the new dense layer will be attached to in graph
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
            value_layer = Softmax(axis=-1 - class_neurons)(value_layer)
            key_layer = Multiply()([key_layer, value_layer])
            layer = Add()([query_layer, key_layer])
            layer = BatchNormalization(axis=1)(layer)
            layer = GaussianDropout(dropout)(layer)
            layer = GELU()(layer)
        if concat_dense:
            layer = Concatenate(axis=-1)([prev_in, layer])
    return layer


def get_output(layer, concat_before_output, outputs, classes, output_activation, loss,
               two_dimensional, class_neurons):
    if class_neurons:
        layer = GlobalAveragePooling1D()(layer)
    layer = Dense(units=256, activation=output_activation,
                  kernel_initializer=initializer())(layer)
    return layer


def compile_model(inp, layer, learning_rate, draw_model, loss, metric, model_compile,
                  global_l2):
    model = Model(inputs=[inp], outputs=[layer])
    if model_compile:
        model.compile(loss=loss,
                      optimizer=OPTIMIZER(lr=learning_rate,
                                          weight_decay_rate=1e-3),
                      metrics=[metric])
        model.summary()
        if draw_model:
            plot_model(model, to_file='model.png')
    return model


def get_model(leakyRelu=True, batchNorm=True, trainNewModel=True,
              concatPreviousLayers=True, repeatInput=True, unroll=True,
              initialLSTM=False, inputDense=False, concatDense=True,
              bidirectional=True, modelCompile=True,
              concatBeforeOutput=True, drawModel=True, gpu=True,
              neuronList=None, indexIn=False, classNeurons=True,
              twoDimensional=True, embedding=False, class_neurons=True,
              inputs=60, neurons_per_layer=120, layer_count=4,
              learningRate=0.005, classes=30, outputs=1, dropout=0.35,
              weightFolderName='MLP_Weights', outputActivation='softmax',
              local_l1=0.01, local_l2=0.01, global_l2=0.001,
              loss='sparse_categorical_crossentropy',
              metric='sparse_categorical_accuracy',
              depth=1, **kwargs):
    if len(kwargs) > 0:
        print(f"Unused Keyword Arguments: {kwargs}")
    if neuronList is None:
        neuronList = utils.get_neuron_list(neurons_per_layer, layer_count, classNeurons,
                                           classes)
    else:
        neurons_per_layer = neuronList[0]

    if trainNewModel:
        # Input layer
        if indexIn:
            inp = Input(shape=(inputs,))
            layer = GaussianDropout(dropout)(inp)
        else:
            layer, inp = get_initial_binary_layer(initialLSTM, gpu, bidirectional,
                                                  inputs,
                                                  unroll, classes, inputDense,
                                                  twoDimensional, embedding,
                                                  class_neurons)
        layer = get_hidden_layers(layer, layer_count, neuronList, leakyRelu, batchNorm,
                                  concatDense, twoDimensional, dropout, depth,
                                  class_neurons)
        layer = get_output(layer, concatBeforeOutput, outputs, classes,
                           outputActivation, loss, twoDimensional, class_neurons)
        # Compiling and displaying model
        model = compile_model(inp, layer, learningRate, drawModel, loss, metric,
                              modelCompile, global_l2)
    else:
        utils.get_previous_weights_from_gdrive(weightFolderName)
        last_used_model = utils.get_latest_model_name(weightFolderName)
        model = load_model(last_used_model)
        model.summary()
    return model
