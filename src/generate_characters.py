import numpy as np
import tensorflow as tf


class GeneratorCallback(tf.keras.callbacks.Callback):
    """
    Keras callback to generate and print a string using the previously trained model
    when the epoch ends.
    """

    def __init__(self, input_string, inputs, output_characters):
        self.input_string = np.array([ord(input_string[i])
                                      for i in range(inputs)])
        self.inputs = inputs
        self.output_characters = output_characters
        super().__init__()

    def _generate_key(self, inp):
        return np.argmax(self.model.predict(inp.reshape(1, -1))[0])

    def _generate_string(self):
        inp = self.input_string.copy()
        for i in range(self.output_characters):
            inp = np.append(inp, self._generate_key(inp[i:]))
        out = ''.join(map(chr, inp[self.inputs:]))
        return out

    def on_epoch_end(self, epoch, logs=None):
        """
        Function called when a training epoch ends. Generates and prints a string.
        :param epoch: Index of the current epoch, ignored
        :param logs: Logs generated during training, ignores
        :return: None
        """
        print(self._generate_string())
