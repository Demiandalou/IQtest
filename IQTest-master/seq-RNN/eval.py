import os
import shutil
import argparse
import random
import numpy as np
import torch
import time
import json
import math
from modeling import SeqRNN
from modeling import SeqData
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
data_file={'train':'seq-train.json','test':'seq-test.json'}

INPUT_SIZE = 3
OUTPUT_SIZE = 1
HIDDEN_SIZE = 4
seqrnn = SeqRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
seqrnn.load_state_dict(torch.load('RNNmodel-50witer.pkl'))

data_dir='../data'
# print(data_dir, data_file['test'])

option_file=open(data_dir+"/seq-public.json", 'r')
option_file = json.load(option_file)
seqdict={}
for i in range(len(option_file)):
    curid=str(option_file[i]["id"])
    seqdict[curid]={"stem": option_file[i]["stem"], "options": option_file[i]["options"], "category": option_file[i]["category"]}


def oneObs(seq_train_data):
    data=seq_train_data.__getitem__(0)
    real_value=data[0][-1][2]
    recovered=seq_train_data.recover(data[-1])
    recover_value=recovered[-1][-1]
    hidden = seqrnn.initHidden()

    for i in range(data.size()[0]):
        output, hidden = seqrnn(data[i].float(), hidden)
    # print(seq_train_data.min_)
    print('predicted value:',output.item())
    print('real value:',real_value.item())

    data[-1][-1][-1]=output[0][0].item()
    recover_output=seq_train_data.recover(data[-1])[-1][-1]

    print('predicted recovered',recover_output)
    print('realval recovered',recover_value)
# 4.9634e-05    ----100w iter
# -0.0006=6e-4  ----10w iter

def cal_accuracy(seq_train_data):
    setlen=seq_train_data.__len__()
    right=0
    cnt=0
    for i in range(setlen):
    # for i in range(10):
        data=seq_train_data.gen_data(i)
        curID=seq_train_data.get_key(i)
        op=seqdict[curID]["options"]

        try:
            real_value=data[0][-1][2]
        except:
            continue
        recovered=seq_train_data.recover(data[-1])
        recover_value=recovered[-1][-1]
        hidden = seqrnn.initHidden()
        for i in range(data.size()[0]):
            output, hidden = seqrnn(data[i].float(), hidden)
        output=output[0][0].item()
        data[-1][-1][-1]=output
        recover_output=seq_train_data.recover(data[-1])[-1][-1]
        print('predicted recovered',recover_output)
        print('realval recovered',recover_value)
        print(op)

        if len(op)==0:
            if abs(int(recover_output)-recover_value)<1:
                right+=1
        else:
            diff=[]
            for j in range(len(op)):
                try:
                    if "/" in op[j]:
                        field=op[j].split('/')
                        op[j]=float(field[0])/float(field[1])
                    diff.append(abs(recover_output-float(op[j])))
                except:
                    print(op)
            if len(diff)==0:
                print(op)
                print(len(op))
            approx=min(diff)
            cnt+=1
            idx=diff.index(approx)
            ans=op[idx]
            if abs(float(ans)-recover_value)<0.05:
            # if int(float(ans))==int(recover_value):
                right+=1
    print("accuracy=",right/setlen)
    return right/setlen



def main():
    seq_train_data = SeqData(data_dir, data_file['test'])
    # oneObs(seq_train_data)
    cal_accuracy(seq_train_data)

if __name__ == "__main__":
    main()
