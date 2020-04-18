from tensorflow.keras import Model
from tensorflow.keras.initializers import orthogonal as initializer
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate, Dense,
                                     Embedding, GaussianDropout, GlobalAveragePooling1D,
                                     Input, Multiply, Softmax)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow_addons.layers import GELU
from tensorflow_addons.optimizers import LAMB as OPTIMIZER

from . import utils


def get_model(trainNewModel=True, concat_dense=True,
              neuron_list=None, class_neurons=True, embedding=True,
              inputs=60, neurons_per_layer=120, layer_count=4,
              learning_rate=0.005, classes=30, dropout=0.35,
              input_dropout=0.1, weightFolderName='MLP_Weights',
              output_activation='softmax',
              loss='sparse_categorical_crossentropy',
              metric='sparse_categorical_accuracy',
              depth=1, **kwargs):
    if kwargs:
        print(f"Unused Keyword Arguments: {kwargs}")
    if neuron_list is None:
        neuron_list = [neurons_per_layer] * layer_count
    if trainNewModel:
        inp = Input(shape=(inputs,))
        layer = Embedding(256, classes)(inp) if embedding else inp
        if input_dropout:
            layer = GaussianDropout(input_dropout)(layer)

        for neurons in neuron_list:
            def dense(in_layer):
                """
                Creates a new dense layer using keras' dense function. The parameters
                used
                to create it are given in the parent function call.
                This function exists as the initializer and the regularizer are both
                classes
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

        if class_neurons:
            layer = GlobalAveragePooling1D()(layer)
        layer = Dense(units=256, activation=output_activation,
                      kernel_initializer=initializer())(layer)

        model = Model(inputs=[inp], outputs=[layer])
        model.compile(loss=loss,
                      optimizer=OPTIMIZER(lr=learning_rate,
                                          weight_decay_rate=1e-3),
                      metrics=[metric])
        model.summary()
        plot_model(model, to_file='model.png')
    else:
        utils.get_previous_weights_from_gdrive(weightFolderName)
        last_used_model = utils.get_latest_model_name(weightFolderName)
        model = load_model(last_used_model)
        model.summary()
    return model
