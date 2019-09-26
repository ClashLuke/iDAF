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
    if self.tpu:
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
      else:
        if self.indexIn:
          tmpOut = np.zeros((self.outputs),dtype=np.float32)
          tmpIn = np.zeros((self.inputs),dtype=np.float32)
          tmpIn[:] = [self.charDictList[self.txt[j]] for j in range(self.inputs)]
          tmpOut[:] = [self.charDictList[self.txt[j]] for j in range(self.inputs,out)]
          yield (tmpIn, tmpOut)
          while True:
            tmpIn[:] = np.append(tmpIn[1:],self.charDictList[self.txt[self.inputs+n]])
            tmpOut[:] = np.append(tmpIn[1:],self.charDictList[self.txt[out+n]])
            n+=1
            yield (tmpIn, tmpOut)
            if n >= self.txtLen:
              n = 0
        else:
          tmpOut = np.zeros((self.outputs*self.classes),dtype=np.float32)
          tmpIn = np.zeros((inputsTimesClasses),dtype=np.float32)
          tmpIn[:] = list(itertools.chain.from_iterable(
              [self.charDictList[self.txt[j]] for j in range(self.inputs)]))
          tmpOut[:] = list(itertools.chain.from_iterable(
             [self.charDictList[self.txt[j]] for j in range(self.inputs,out)]))
          yield (tmpIn, tmpOut)
          while True:
            tmpIn[:] = np.append(tmpIn[self.classes:],self.charDictList[self.txt[self.inputs+n]])
            tmpOut[:] = np.append(tmpOut[self.classes:],self.charDictList[self.txt[out+n]])
            n+=1
            yield (tmpIn, tmpOut)
            if n >= self.txtLen:
              n = 0
    else:
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
          currentBatchsize += int(self.batchsize*self.changePerKerasEpoch)