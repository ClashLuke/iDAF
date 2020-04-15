import itertools

import numpy as np


class Generator:
    def __init__(self, batchsize, txt, outputs, index_in, inputs, steps, char_dict_list,
                 char_dict, classes, change_per_keras_epoch, embedding):
        self.batchsize = batchsize
        self.txt = txt
        self.txtLen = len(txt) - inputs - batchsize - 2
        self.outputs = outputs
        self.indexIn = index_in or embedding
        self.inputs = inputs
        self.steps = steps
        self.charDictList = char_dict_list
        self.charDict = char_dict
        self.classes = classes
        self.changePerKerasEpoch = change_per_keras_epoch

    def inp_generator(self):
        n = 0
        # Using lists instead of numpy arrays is about seven times slower
        if self.outputs == 1:
            tmp_out = np.zeros((self.batchsize, 1), dtype=np.float32)
            if self.indexIn:
                tmp_in = np.zeros((self.batchsize, self.inputs), dtype=np.float32)
                tmp_in[-1][:] = [self.charDictList[self.txt[j]] for j in
                                 range(self.inputs)]
                tmp_out[0][0] = self.charDict[self.txt[self.inputs]]
                while True:
                    for b in range(self.batchsize):
                        tmp_in[b][:] = np.append(tmp_in[b - 1][1:], self.charDictList[
                            self.txt[self.inputs + b + n]])
                        tmp_out[b][0] = self.charDict[self.txt[self.inputs + 1 + b + n]]
                    for b in range(self.batchsize):
                        yield tmp_in[b], tmp_out[b]
                    n += self.batchsize
                    if n >= self.txtLen:
                        n = 0
            else:
                tmp_in = np.zeros((self.batchsize, self.inputs * self.classes),
                                  dtype=np.float32)
                tmp_in[-1][:] = list(itertools.chain.from_iterable(
                        [self.charDictList[self.txt[j]] for j in range(self.inputs)]))
                tmp_out[0][0] = self.charDict[self.txt[self.inputs]]
                while True:
                    for b in range(self.batchsize):
                        tmp_in[b][:] = np.append(tmp_in[b - 1][self.classes:],
                                                 self.charDictList[
                                                     self.txt[self.inputs + b + n]])
                        tmp_out[b][0] = self.charDict[self.txt[self.inputs + 1 + b + n]]
                    for b in range(self.batchsize):
                        yield tmp_in[b], tmp_out[b]
                    n += self.batchsize
                    if n >= self.txtLen:
                        n = 0
        else:
            out = self.inputs + self.outputs
            tmp_out = np.zeros((self.batchsize, self.outputs, 1), dtype=np.float32)
            if self.indexIn:
                tmp_in = np.zeros((self.batchsize, self.inputs), dtype=np.float32)
                tmp_in[-1][:] = [self.charDictList[self.txt[j]] for j in
                                 range(self.inputs)]
                tmp_out[0][:] = [[self.charDict[self.txt[self.inputs + j]]] for j in
                                 range(self.outputs)]
                while True:
                    for b in range(self.batchsize):
                        tmp_in[b][:] = np.append(tmp_in[b - 1][1:], self.charDictList[
                            self.txt[self.inputs + b + n]])
                        tmp_out[b][:] = np.append(tmp_out[b - 1][1:], self.charDict[
                            self.txt[out + 1 + b + n]]).reshape(self.outputs, 1)
                    for b in range(self.batchsize):
                        yield tmp_in[b], tmp_out[b]
                    n += self.batchsize
                    if n >= self.txtLen:
                        n = 0
            else:
                tmp_in = np.zeros((self.batchsize, self.inputs * self.classes),
                                  dtype=np.float32)
                tmp_in[-1][:] = list(itertools.chain.from_iterable(
                        [self.charDictList[self.txt[j]] for j in range(self.inputs)]))
                tmp_out[0][:] = [[self.charDict[self.txt[self.inputs + j]]] for j in
                                 range(self.outputs)]
                while True:
                    for b in range(self.batchsize):
                        tmp_in[b][:] = np.append(tmp_in[b - 1][self.classes:],
                                                 self.charDictList[
                                                     self.txt[self.inputs + b + n]])
                        tmp_out[b][:] = np.append(tmp_out[b - 1][1:], self.charDict[
                            self.txt[out + 1 + b + n]]).reshape(self.outputs, 1)
                    for b in range(self.batchsize):
                        yield tmp_in[b], tmp_out[b]
                    n += self.batchsize
                    if n >= self.txtLen:
                        n = 0
