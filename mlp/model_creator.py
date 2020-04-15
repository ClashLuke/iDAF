import tensorflow as tf
import tensorflow_addons as tfa

from . import utils


def add_advanced_layers(layer, leaky_relu, batch_norm):
    if batch_norm:
        layer = tf.keras.layers.BatchNormalization()(layer)
    if leaky_relu:
        layer = tf.keras.layers.LeakyReLU(0.2)(layer)
    return layer


def get_initial_binary_layer(initial_lstm, gpu, bidirectional,
                             inputs, unroll, classes, dropout,
                             two_dimensional, embedding):
    if embedding:
        inp = tf.keras.layers.Input(shape=(inputs,))
        layer = tf.keras.layers.Embedding(input_dim=inputs, output_dim=classes)(inp)
    else:
        inp = tf.keras.layers.Input(shape=(inputs * classes,))
        layer = inp
    if dropout:
        layer = tf.keras.layers.GaussianDropout(dropout)(layer)
    return layer, inp


def initialise_list(length, init_value, differing_value_position, differing_value):
    initial_list = [init_value] * length
    initial_list[differing_value_position] = differing_value
    return initial_list


def get_input_layer(layer, inputs, classes, input_dense):
    if input_dense:
        layer = tf.keras.layers.Dense(units=inputs,
                                      kernel_initializer=tf.keras.initializers.Orthogonal())(
                layer)
        layer = tfa.layers.GELU()(layer)
    return layer


def get_hidden_layers(layer, layer_count, neuron_list, leaky_relu, batch_norm,
                      concat_dense,
                      two_dimensional, dropout, depth):
    if two_dimensional:
        def dense(*args, **kwargs):
            return tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(*args,
                                                                         **kwargs))
    else:
        def dense(*args, **kwargs):
            return tf.keras.layers.Dense(*args, **kwargs)
    for i in range(layer_count - 1):
        n = neuron_list[i]
        prev_in = layer
        for _ in range(depth):
            key_layer = dense(n, kernel_initializer=tf.keras.initializers.Orthogonal())(
                    layer)
            query_layer = dense(n,
                                kernel_initializer=tf.keras.initializers.Orthogonal())(
                    layer)
            value_layer = tfa.layers.GELU()(key_layer)
            value_layer = dense(n,
                                kernel_initializer=tf.keras.initializers.Orthogonal())(
                    value_layer)
            value_layer = tf.keras.layers.Softmax()(value_layer)
            key_layer = tf.keras.layers.Multiply()([key_layer, value_layer])
            layer = tf.keras.layers.Add()([query_layer, key_layer])
            layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
            layer = tf.keras.layers.GaussianDropout(dropout)(layer)
            layer = tfa.layers.GELU()(layer)
        if concat_dense:
            layer = tf.keras.layers.Concatenate(axis=-1)([prev_in, layer])
    return layer


def get_output(layer, concat_before_output, outputs, classes, output_activation, loss,
               two_dimensional):
    if two_dimensional:
        layer = tf.keras.layers.Flatten()(layer)
    if 'crossentropy' not in loss:
        classes = 1
    layer = tf.keras.layers.Dense(units=outputs * classes, activation=output_activation,
                                  kernel_initializer=tf.keras.initializers.Orthogonal())(
            layer)

    if outputs > 1:
        layer = tf.keras.layers.Reshape((outputs, classes))(layer)
    return layer


def compile_model(inp, layer, learning_rate, draw_model, loss, metric, model_compile):
    model = tf.keras.Model(inputs=[inp], outputs=[layer])
    if model_compile:
        model.compile(loss=loss,
                      optimizer=tfa.optimizers.LAMB(lr=learning_rate,
                                                    weight_decay_rate=1e-3),
                      metrics=[metric])
        model.summary()
        if draw_model:
            tf.keras.utils.plot_model(model, to_file='model.png')
    return model


def get_model(leakyRelu=True, batchNorm=True, trainNewModel=True,
              concatPreviousLayers=True, repeatInput=True, unroll=True,
              initialLSTM=False, inputDense=False, concatDense=True,
              bidirectional=True, modelCompile=True,
              concatBeforeOutput=True, drawModel=True, gpu=True,
              neuronList=None, indexIn=False, classNeurons=True,
              twoDimensional=True, embedding=False,
              inputs=60, neuronsPerLayer=120, layerCount=4,
              learningRate=0.005, classes=30, outputs=1, dropout=0.35,
              weightFolderName='MLP_Weights', outputActivation='softmax',
              loss='sparse_categorical_crossentropy',
              metric='sparse_categorical_accuracy',
              depth=1, **kwargs):
    if len(kwargs) > 0:
        print(f"Unused Keyword Arguments: {kwargs}")
    if neuronList is None:
        neuronList = utils.get_neuron_list(neuronsPerLayer, layerCount, classNeurons,
                                           classes)
    else:
        neuronsPerLayer = neuronList[0]

    if trainNewModel:
        # Input layer
        if indexIn:
            inp = tf.keras.layers.Input(shape=(inputs,))
            layer = tf.keras.layers.GaussianDropout(dropout)(inp)
        else:
            layer, inp = get_initial_binary_layer(initialLSTM, gpu, bidirectional,
                                                  inputs,
                                                  unroll, classes, inputDense,
                                                  twoDimensional, embedding)
        layer = tf.keras.layers.Dense(inputs,
                                      kernel_initializer=tf.keras.initializers.Orthogonal())(
                layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tfa.layers.GELU()(layer)
        layer = get_hidden_layers(layer, layerCount, neuronList, leakyRelu, batchNorm,
                                  concatDense, twoDimensional, dropout, depth)
        layer = get_output(layer, concatBeforeOutput, outputs, classes, outputActivation,
                           loss, twoDimensional)
        # Compiling and displaying model
        model = compile_model(inp, layer, learningRate, drawModel, loss, metric,
                              modelCompile)
    else:
        utils.get_previous_weights_from_gdrive(weightFolderName)
        last_used_model = utils.get_latest_model_name(weightFolderName)
        model = tf.keras.models.load_model(last_used_model)
        model.summary()
    return model
