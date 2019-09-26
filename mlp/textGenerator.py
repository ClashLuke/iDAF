import numpy as np
import itertools

class generator():
  def __init__(self, batchsize, txt, outputs, indexIn, inputs, steps, charDictList, charDict, classes, changePerKerasEpoch):
    self.batchsize = batchsize
    self.txt = txt
    self.txtLen = len(txt)-inputs-batchsize-2
    self.outputs = outputs
    self.indexIn = indexIn
    self.inputs = inputs
    self.steps = steps
    self.charDictList = charDictList
    self.charDict = charDict
    self.classes = classes
    self.changePerKerasEpoch = changePerKerasEpoch
  def inpGenerator(self):
    n = 0
    tmpOut = np.zeros((self.batchsize,1),dtype=np.float32)
    # Using lists instead of numpy arrays is about seven times slower
    if self.indexIn:
      
      tmpOut = np.zeros((self.batchsize,1),dtype=np.float32)
      tmpIn = [None]*self.batchsize
      tmpIn[0][:] = [self.charDictList[self.txt[j]] for j in range(self.inputs)]
      tmpOut[0][0] = self.charDict[self.txt[self.inputs]] 
      while True:
        for b in range(self.batchsize):
          tmpIn[b][:] = np.append(tmpIn[b-1][1:],self.charDictList[self.txt[self.inputs+b]])
          tmpOut[b][0] = self.charDict[self.txt[self.inputs+1+b]]
        for b in range(self.batchsize):
          yield (tmpIn[b], tmpOut[b])
        if n >= self.txtLen:
          n = 0
    else:
      tmpIn = [None]*self.batchsize
      tmpIn[0][:] = list(itertools.chain.from_iterable(
          [self.charDictList[self.txt[j]] for j in range(self.inputs)]))
      tmpOut[0][0] = self.charDict[self.txt[self.inputs]] 
      while True:
        for b in range(self.batchsize):
          tmpIn[b][:] = np.append(tmpIn[b-1][self.classes:],self.charDictList[self.txt[self.inputs+b]])
          tmpOut[b][0] = self.charDict[self.txt[self.inputs+1+b]]
        for b in range(self.batchsize):
          yield (tmpIn[b], tmpOut[b])
        if n >= self.txtLen:
          n = 0