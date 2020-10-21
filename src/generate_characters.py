import numpy as np
import tensorflow as tf


class GeneratorCallback(tf.keras.callbacks.Callback):
    """
    Keras callback to generate and print a string using the previously trained model
    when the epoch ends.
    """

    def __init__(self, input_string, inputs, output_characters, dtype):
        self.input_string = np.array([ord(input_string[i])
                                      for i in range(inputs)],
                                     dtype=np.int32)
        self.inputs = inputs
        self.output_characters = output_characters
        super().__init__()

#    @tf.function(experimental_relax_shapes=True)
    def _generate_string(self):
        inp = self.input_string.copy()
        model = self.model
        for i in range(self.output_characters):
            output_probabilities = model.predict_on_batch((inp.reshape(1, -1),)[0]
            possible_indices = np.arange(output_probabilities.shape[0])
            output_index = np.random.choice(possible_indices, p=output_probabilities)
            inp = np.append(inp, output_index)[1:]               
        out = ''.join(map(chr, inp))
        return out

    def on_epoch_end(self, epoch, logs=None):
        """
        Function called when a training epoch ends. Generates and prints a string.
        :param epoch: Index of the current epoch, ignored
        :param logs: Logs generated during training, ignores
        :return: None
        """
        print(self._generate_string())
