import numpy as np
import itertools

class generator():
  def __init__(self, batchsize, txt, outputs, indexIn, inputs, steps, charDictList, charDict, classes, valSplit, changePerKerasEpoch, tpu):
    self.batchsize = batchsize
    self.txt = txt
    self.txtLen = len(txt)-inputs-2
    self.outputs = outputs
    self.indexIn = indexIn
    self.inputs = inputs
    self.steps = steps
    self.charDictList = charDictList
    self.charDict = charDict
    self.classes = classes
    self.changePerKerasEpoch = changePerKerasEpoch
    self.tpu = tpu
  def inpGenerator(self):
    out = self.inputs+self.outputs
    inputsTimesClasses = self.inputs*self.classes
    n = 0
    if self.outputs == 1:
      tmpOut = np.zeros((1),dtype=np.float32)
      if self.indexIn:
        tmpIn = np.zeros((self.inputs),dtype=np.float32)
        tmpIn[:] = [self.charDictList[self.txt[j]] for j in range(self.inputs)]
        tmpOut[0] = self.charDict[self.txt[self.inputs]] 
        yield (tmpIn, tmpOut)
        while True:
          tmpIn[:] = np.append(tmpIn[1:],self.charDictList[self.txt[self.inputs+n]])
          tmpOut[0] = self.charDict[self.txt[self.inputs+1+n]]
          n+=1
          yield (tmpIn, tmpOut)
          if n >= self.txtLen:
            n = 0
      else:
        tmpIn = np.zeros((inputsTimesClasses),dtype=np.float32)
        tmpIn[:] = list(itertools.chain.from_iterable(
            [self.charDictList[self.txt[j]] for j in range(self.inputs)]))
        tmpOut[0] = self.charDict[self.txt[self.inputs]] 
        yield (tmpIn, tmpOut)
        while True:
          tmpIn[:] = np.append(tmpIn[self.classes:],self.charDictList[self.txt[self.inputs+n]])
          tmpOut[0] = self.charDict[self.txt[self.inputs+1+n]]
          n+=1
          yield (tmpIn, tmpOut)
          if n >= self.txtLen:
            n = 0