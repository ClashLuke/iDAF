import tensorflow as tf
import numpy as np
import itertools

class generateChars():
  def __init__(self, classes, inputs, inputString, outCharCount, outputs, chars, charDictList):
    self.classes = classes
    self.inputs = inputs
    self.outputs = outputs
    self.chars = chars
    self.charDictList = charDictList
    self.inputString = inputString
    self.outCharCount = outCharCount

  def genKey(self, inp, model):
    topred = np.zeros((1,self.classes*self.inputs))
    topred[0][:] = inp[:]
    if self.outputs == 1:
      pred = np.argmax(model.predict(topred)[0])
      pred = [self.chars[pred]]
    else:
      pred = model.predict(topred)[0]
      pred = [np.argmax(p) for p in pred]
      pred = [self.chars[p] for p in pred]
    return pred
    #return CHARS[np.argmax(pred)]

  def genRecurse(self, instr, model):
    # initial input
    inp = list(itertools.chain.from_iterable(
        [self.charDictList[self.inputString[i]] for i in range(self.inputs)]
    ))
    for i in range(self.outCharCount):
      res = self.genKey(inp[i*self.classes*self.outputs:], model)
      inp = inp+list(itertools.chain.from_iterable([self.charDictList[r] for r in res]))
    return inp

  def genStr(self, instr, model):
    recOut = self.genRecurse(instr, model)
    out = ''.join(self.chars[np.argmax(recOut[i*self.classes:(i+1)*self.classes])] for i in range(self.outCharCount))
    return out

class GenerateCharsCallback(tf.keras.callbacks.Callback):
  def __init__(self, generateCharsInstance, inputString, inputs, decodeOutput):
    self.generateCharsInstance = generateCharsInstance
    self.inputString = inputString
    self.inputs = inputs
    self.decodeOutput = decodeOutput
  def on_epoch_end(self, batch, logs = {}):
    if self.decodeOutput:
      print(self.generateCharsInstance.genStr(self.inputString, self.model)[self.inputs:])
    else:
      print(self.model.predict(self.inputString))

    return None
