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
    tmpOut = np.zeros((1),dtype=np.float32)
    if self.indexIn:
      tmpIn = [None]*self.batchsize
      tmpIn[0] = [self.charDictList[self.txt[j]] for j in range(self.inputs)]
      tmpOut[0] = self.charDict[self.txt[self.inputs]] 
      while True:
        for b in range(1,self.batchsize):
          tmpIn[b] = tmpIn[b-1][1:]+[self.charDictList[self.txt[self.inputs+b]]]
          tmpOut[b][0] = self.charDict[self.txt[self.inputs+1+b]]
        for b in range(self.batchsize):
          yield (np.array(tmpIn[b]), np.array(tmpOut[b]))
        if n >= self.txtLen:
          n = 0
    else:
      tmpIn = [None]*self.batchsize
      tmpIn[0] = list(itertools.chain.from_iterable(
          [self.charDictList[self.txt[j]] for j in range(self.inputs)]))
      tmpOut[0] = self.charDict[self.txt[self.inputs]] 
      while True:
        for b in range(1,self.batchsize):
          tmpIn[b] = tmpIn[b-1][1:]+[self.charDictList[self.txt[self.inputs+b]]]
          tmpOut[b][0] = self.charDict[self.txt[self.inputs+1+b]]
        for b in range(self.batchsize):
          yield (np.array(tmpIn[b]), np.array(tmpOut[b]))
        if n >= self.txtLen:
          n = 0