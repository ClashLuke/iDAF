import numpy as np
import tensorflow as tf


class GenerateChars:
    def __init__(self, inputs, input_string, out_char_count, outputs):
        self.inputs = inputs
        self.outputs = outputs

        self.inputString = input_string
        self.outCharCount = out_char_count

    def genKey(self, inp, model):
        return np.argmax(model.predict(inp.reshape(1, -1))[0])

    def gen_recurse(self, instr, model):
        inp = np.array([ord(self.inputString[i]) for i in range(self.inputs)])
        for i in range(self.outCharCount):
            inp = np.append(inp, self.genKey(inp[i:], model))
        return inp

    def gen_str(self, instr, model):
        rec_out = self.gen_recurse(instr, model)
        out = ''.join(map(chr, rec_out))
        return out


class GenerateCharsCallback(tf.keras.callbacks.Callback):
    def __init__(self, generate_chars_instance, input_string, inputs, decode_output):
        self.generateCharsInstance = generate_chars_instance
        self.inputString = input_string
        self.inputs = inputs
        self.decodeOutput = decode_output

    def on_epoch_end(self, batch, logs=None):
        if self.decodeOutput:
            print(self.generateCharsInstance.gen_str(self.inputString, self.model)[
                  self.inputs:])
        else:
            print(self.model.predict(self.inputString))

        return None
