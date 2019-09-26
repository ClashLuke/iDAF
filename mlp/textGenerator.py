import numpy as np
import itertools

class generator():
  def __init__(self, batchsize, txt, outputs, indexIn, inputs, steps, charDictList, charDict, classes, valSplit, changePerKerasEpoch, tpu):
    self.batchsize = batchsize
    self.txt = txt
    self.txtLen = len(txt)
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
    ik = 0
    currentBatchsize = self.batchsize
    if self.outputs == 1:
      while True:
        tmpIn = np.zeros((currentBatchsize,inputsTimesClasses),dtype=np.float32)
        tmpOut = np.zeros((currentBatchsize,1),dtype=np.float32)
        if self.indexIn:
          for _ in range(self.steps):
            tmpIn[0][:] = [self.charDictList[self.txt[j]] for j in range(ik,ik+self.inputs)]
            tmpOut[0] = self.charDict[self.txt[ik+self.inputs]] 
            n = 0
            for i in range(1,currentBatchsize):
              tmpIn[i][:] = np.append(tmpIn[i-1][1:],self.charDictList[self.txt[self.inputs+n+ik]])
              tmpOut[i] = self.charDict[self.txt[self.inputs+1+n+ik]]
              n+=1
            yield (tmpIn, tmpOut)
            ik+=currentBatchsize
            if ik >= self.txtLen:
              ik = 0
        else:
          for _ in range(self.steps):
            tmpIn[0][:] = list(itertools.chain.from_iterable(
                [self.charDictList[self.txt[j]] for j in range(ik,ik+self.inputs)]))
            tmpOut[0] = self.charDict[self.txt[ik+self.inputs]] 
            n = 0
            for i in range(1,currentBatchsize):
              tmpIn[i][:] = np.append(tmpIn[i-1][self.classes:],self.charDictList[self.txt[self.inputs+n+ik]])
              tmpOut[i] = self.charDict[self.txt[self.inputs+1+n+ik]]
              n+=1
            yield (tmpIn, tmpOut)
            ik+=currentBatchsize
            if ik >= self.txtLen:
              ik = 0
              
        if not self.tpu:
          currentBatchsize += int(self.batchsize*self.changePerKerasEpoch)
    if self.outputs == 1:
      while True:
        tmpIn = np.zeros((currentBatchsize,inputsTimesClasses),dtype=np.float32)
        tmpOut = np.zeros((currentBatchsize,1),dtype=np.float32)
        if self.indexIn:
          for _ in range(self.steps):
            tmpIn[0][:] = [self.charDictList[self.txt[j]] for j in range(ik,ik+self.inputs)]
            tmpOut[0] = self.charDict[self.txt[ik+self.inputs]] 
            n = 0
            for i in range(1,currentBatchsize):
              tmpIn[i][:] = np.append(tmpIn[i-1][1:],self.charDictList[self.txt[self.inputs+n+ik]])
              tmpOut[i] = self.charDict[self.txt[self.inputs+1+n+ik]]
              n+=1
            yield (tmpIn, tmpOut)
            ik+=currentBatchsize
            if ik >= self.txtLen:
              ik = 0
        else:
          for _ in range(self.steps):
            tmpIn[0][:] = list(itertools.chain.from_iterable(
                [self.charDictList[self.txt[j]] for j in range(ik,ik+self.inputs)]))
            tmpOut[0] = self.charDict[self.txt[ik+self.inputs]] 
            n = 0
            for i in range(1,currentBatchsize):
              tmpIn[i][:] = np.append(tmpIn[i-1][self.classes:],self.charDictList[self.txt[self.inputs+n+ik]])
              tmpOut[i] = self.charDict[self.txt[self.inputs+1+n+ik]]
              n+=1
            yield (tmpIn, tmpOut)
            ik+=currentBatchsize
            if ik >= self.txtLen:
              ik = 0
        if not self.tpu:
          currentBatchsize += int(self.batchsize*self.changePerKerasEpoch)
    else:
      while True:
        tmpIn = np.zeros((currentBatchsize,inputsTimesClasses),dtype=np.float32)
        tmpOut = np.zeros((currentBatchsize,self.outputs,1),dtype=np.float32)
        if self.indexIn:
          for _ in range(self.steps):
            tmpIn[0][:] = [self.charDictList[self.txt[j]] for j in range(ik,ik+self.inputs)]
            tmpOut[0][:] = np.array([self.charDict[self.txt[j]] for j in range(ik+self.inputs, ik+out)]).reshape(self.outputs,1)
            n = 0
            for i in range(1,currentBatchsize):
              tmpIn[i][:] = np.append(tmpIn[i-1][1:],self.charDictList[self.txt[self.inputs+n+ik]])
              tmpOut[i][:] = np.append(tmpOut[i-1][1:],self.charDict[self.txt[out+n+ik]]).reshape(self.outputs,1)
              n+=1
            yield (tmpIn, tmpOut)
            ik+=currentBatchsize
            if ik >= self.txtLen:
              ik = 0
        else:
          for _ in range(self.steps):
            tmpIn[0][:] = list(itertools.chain.from_iterable([self.charDictList[self.txt[j]] for j in range(ik,ik+self.inputs)]))
            tmpOut[0][:] = np.array([self.charDict[self.txt[j]] for j in range(ik+self.inputs, ik+out)]).reshape(self.outputs,1)
            n = 0
            for i in range(1,currentBatchsize):
              tmpIn[i][:] = np.append(tmpIn[i-1][self.classes:],self.charDictList[self.txt[self.inputs+n+ik]])
              tmpOut[i][:] = np.append(tmpOut[i-1][1:],self.charDict[self.txt[out+n+ik]]).reshape(self.outputs,1)
              n+=1
            yield (tmpIn, tmpOut)
            ik+=currentBatchsize
            if ik >= self.txtLen:
              ik = 0
          
        if not self.tpu:
          currentBatchsize += int(self.batchsize*self.changePerKerasEpoch)