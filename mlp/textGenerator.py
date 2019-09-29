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
    # Using lists instead of numpy arrays is about seven times slower
    if self.outputs == 1:
      tmpOut = np.zeros((self.batchsize,1),dtype=np.float32)
      if self.indexIn:
        tmpIn = np.zeros((self.batchsize,self.inputs),dtype=np.float32)
        tmpIn[-1][:] = [self.charDictList[self.txt[j]] for j in range(self.inputs)]
        tmpOut[0][0] = self.charDict[self.txt[self.inputs]] 
        while True:
          for b in range(self.batchsize):
            tmpIn[b][:] = np.append(tmpIn[b-1][1:],self.charDictList[self.txt[self.inputs+b+n]])
            tmpOut[b][0] = self.charDict[self.txt[self.inputs+1+b+n]]
          for b in range(self.batchsize):
            yield (tmpIn[b], tmpOut[b])
          n+=self.batchsize
          if n >= self.txtLen:
            n = 0
      else:
        tmpIn = np.zeros((self.batchsize,self.inputs*self.classes),dtype=np.float32)
        tmpIn[-1][:] = list(itertools.chain.from_iterable(
            [self.charDictList[self.txt[j]] for j in range(self.inputs)]))
        tmpOut[0][0] = self.charDict[self.txt[self.inputs]] 
        while True:
          for b in range(self.batchsize):
            tmpIn[b][:] = np.append(tmpIn[b-1][self.classes:],self.charDictList[self.txt[self.inputs+b+n]])
            tmpOut[b][0] = self.charDict[self.txt[self.inputs+1+b+n]]
          for b in range(self.batchsize):
            yield (tmpIn[b], tmpOut[b])
          n+=self.batchsize
          if n >= self.txtLen:
            n = 0
    else:
      tmpOut = np.zeros((self.batchsize,self.outputs),dtype=np.float32)
      if self.indexIn:
        tmpIn = np.zeros((self.batchsize,self.inputs),dtype=np.float32)
        tmpIn[-1][:] = [self.charDictList[self.txt[j]] for j in range(self.inputs)]
        tmpOut[0][:] = [self.charDict[self.txt[self.inputs]] for j in range(self.outputs)]
        while True:
          for b in range(self.batchsize):
            tmpIn[b][:] = np.append(tmpIn[b-1][1:],self.charDictList[self.txt[self.inputs+b+n]])
            tmpOut[b][:] = np.append(tmpIn[b-1][1:],self.charDict[self.txt[self.inputs+1+b+n]])
          for b in range(self.batchsize):
            yield (tmpIn[b], tmpOut[b])
          n+=self.batchsize
          if n >= self.txtLen:
            n = 0
      else:
        tmpIn = np.zeros((self.batchsize,self.inputs*self.classes),dtype=np.float32)
        tmpIn[-1][:] = list(itertools.chain.from_iterable(
            [self.charDictList[self.txt[j]] for j in range(self.inputs)]))
        tmpOut[0][:] = [self.charDict[self.txt[self.inputs]] for j in range(self.outputs)]
        while True:
          for b in range(self.batchsize):
            tmpIn[b][:] = np.append(tmpIn[b-1][self.classes:],self.charDictList[self.txt[self.inputs+b+n]])
            tmpOut[b][:] = np.append(tmpIn[b-1][1:],self.charDict[self.txt[self.inputs+1+b+n]])
          for b in range(self.batchsize):
            yield (tmpIn[b], tmpOut[b])
          n+=self.batchsize
          if n >= self.txtLen:
            n = 0
       