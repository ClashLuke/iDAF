import numpy as np
import itertools

class generator():
  def __init__(self, batchsize, txt, outputs, indexIn, inputs, steps, charDictList, charDict, classes, valSplit, changePerKerasEpoch):
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
    self.valBegin = int(self.txtLen*valSplit)
    self.changePerKerasEpoch = changePerKerasEpoch
  def inpGenerator(self):
    out = self.inputs+self.outputs
    inputsTimesClasses = self.inputs*self.classes
    ik = 0
    currentBatchsize = self.batchsize
    if self.outputs == 1:
      while True:
        tmpIn = np.zeros((currentBatchsize,inputsTimesClasses),dtype=np.bool)
        tmpOut = np.zeros((currentBatchsize,1),dtype=np.float)
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
            if ik >= self.valBegin:
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
            if ik >= self.valBegin:
              ik = 0
        currentBatchsize += int(self.batchsize*self.changePerKerasEpoch)
    else:
      while True:
        tmpIn = np.zeros((currentBatchsize,inputsTimesClasses),dtype=np.bool)
        tmpOut = np.zeros((currentBatchsize,self.outputs,1),dtype=np.float)
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
            if ik >= self.valBegin:
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
            if ik >= self.valBegin:
              ik = 0
          
        currentBatchsize += int(self.batchsize*self.changePerKerasEpoch)
      
      
  def outGenerator(self):
    out = self.inputs+self.outputs
    inputsTimesClasses = self.inputs*self.classes
    ok = self.valBegin
    tmpIn = np.zeros((self.batchsize,inputsTimesClasses),dtype=np.bool)
    if self.outputs == 1:
      tmpOut = np.zeros((self.batchsize,self.outputs),dtype=np.float)
      while True:
        if self.indexIn:
          tmpIn[0][:] = [self.charDictList[self.txt[j]] for j in range(ok,ok+self.inputs)]
        else:
          tmpIn[0][:] = list(itertools.chain.from_iterable(
              [self.charDictList[self.txt[j]] for j in range(ok,ok+self.inputs)]))
        tmpOut[0] = self.charDict[self.txt[ok+self.inputs]]
        n = 0
        try:
          for i in range(1,self.batchsize):
            tmpIn[i][:] = np.append(tmpIn[i-1][1 if self.indexIn else self.classes:],self.charDictList[self.txt[ok+self.inputs+n]])
            tmpOut[i] = self.charDict[self.txt[ok+self.inputs+1+n]]
            n+=1
          yield (tmpIn, tmpOut)
        except:
          pass
        ok+=self.batchsize
        if ok >= self.txtLen:
            ok = self.valBegin
    else:
      tmpOut = np.zeros((self.batchsize,self.outputs,1),dtype=np.float)
      while True:
        if self.indexIn:
          tmpIn[0][:] = [self.charDictList[self.txt[j]] for j in range(ok,ok+self.inputs)]
        else:
          tmpIn[0][:] = list(itertools.chain.from_iterable(
              [self.charDictList[self.txt[j]] for j in range(ok,ok+self.inputs)]))
        tmpOut[0][:] = np.array([self.charDict[self.txt[j]] for j in range(ok+self.inputs,ok+out)]).reshape(self.outputs,1)
        n = 0
        for i in range(1,self.batchsize):
          tmpIn[i][:] = np.append(tmpIn[i-1][1 if self.indexIn else self.classes:],self.charDictList[self.txt[ok+self.inputs+n]])
          tmpOut[i][:] = np.append(tmpOut[i-1][1:],self.charDict[self.txt[ok+out+n]]).reshape(self.outputs,1)
          n+=1
        yield (tmpIn, tmpOut)
        ok+=self.batchsize
        if ok > self.txtLen:
          ok = self.valBegin
