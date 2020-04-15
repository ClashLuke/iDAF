import itertools

import numpy as np
import tensorflow as tf


class GenerateChars:
    def __init__(self, classes, inputs, input_string, out_char_count, outputs, chars,
                 char_dict):
        self.classes = classes
        self.inputs = inputs
        self.outputs = outputs
        self.chars = chars
        self.char_dict = char_dict
        self.inputString = input_string
        self.outCharCount = out_char_count

    def genKey(self, inp, model):
        topred = np.array(inp).reshape(1, -1)
        pred = np.argmax(model.predict(topred)[0])
        return pred

    def gen_recurse(self, instr, model):
        inp = [self.char_dict[self.inputString[i]] for i in range(self.inputs)]
        for i in range(self.outCharCount):
            inp.append(self.genKey(inp[i:], model))
        return inp

    def gen_str(self, instr, model):
        rec_out = self.gen_recurse(instr, model)
        out = ''.join(self.chars[np.argmax(rec_out[i * self.classes:
                                                   (i + 1) * self.classes])]
                      for i in range(self.outCharCount))
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
