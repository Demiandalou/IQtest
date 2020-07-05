'''
This file contains two data classes, which uses MinMaxScaler or divide by 10**MaxNumberLength to normalize data
This file also contains an RNN model called SeqRNN to solve seqence problem in IQ-test
'''
import torch
from torch import nn
import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 

class SeqData_minmax(Dataset):
    def __init__(self, data_dir, filename):
        files=os.path.join(data_dir, filename)
        inputdic=open(files, 'r')
        self.seqdict = json.load(inputdic)
        self.seqlist=[]
        self.consistent=[]
        self.keylist=[]
        for key in self.seqdict:
            onelist=[]
            stem=self.seqdict[key]['stem']
            # skip all next operations in for if sequence is illegal
            if stem[0][0].isalpha()==True or stem[0][0].isdigit()==False:
                continue 
            tmplist=stem.split(",")
            if len(tmplist)==1:
                tmplist=stem.split(" ")
            for i in range(len(tmplist)):
                if "/" in tmplist[i]:
                    field=tmplist[i].split('/')
                    tmplist[i]=float(field[0])/float(field[1])
                elif "√" in tmplist[i]:
                    field=tmplist[i].strip(' ').split('√')
                    if field[0]=='':
                        tmplist[i]=math.sqrt(float(field[1]))
                    else:
                        tmplist[i]=float(field[0])*math.sqrt(float(field[1]))
                elif "^" in tmplist[i]:
                    field=tmplist[i].strip(' ').split('^')
                    tmplist[i]=float(field[0])**float(field[1])
            for i in range(len(tmplist)-2):
                try:
                    tmplist=list(map(float,tmplist))
                    onelist.append(tmplist[i:i+3])
                    if tmplist[i]>1000000:
                        print(tmplist[i])
                    self.consistent.append(tmplist[i:i+3])
                except:
                    pass
            self.seqlist.append(onelist)
            self.keylist.append(key)
        self.df=df=pd.DataFrame(self.consistent)
        self.scaler=MinMaxScaler()
        self.scaler.fit_transform(self.df)
    def get_key(self,idx):
        return self.keylist[idx]
    def __getitem__(self):
        res=[]
        while len(res)<=1 or len(res[0])==0:
            index=np.random.randint(0,len(self.seqlist))
            res=self.seqlist[index][:]
        for i in range(len(res)):
            if len(res[i])!=1:
                res[i]=self.scaler.transform(np.array([res[i]]))
            else:
                res[i]=self.scaler.transform(res[i][0])
        res=torch.tensor(res,dtype=torch.double)
        return res
    def gen_data(self,index):
        res=self.seqlist[index][:]
        for i in range(len(res)):
            if len(res[i])!=1:
                res[i]=self.scaler.transform(np.array([res[i]]))
            else:
                res[i]=self.scaler.transform(res[i][0])
        res=torch.tensor(res,dtype=torch.double)
        return res

    def __len__(self):
        return len(self.seqlist)

    def recover(self,data):
        return self.scaler.inverse_transform(data)
    
    def cal_accuracy(self,seqrnn,seqdict):
        setlen=self.__len__()
        right=0
        cnt=0
        for i in range(setlen):
        # for i in range(10):
            data=self.gen_data(i)
            curID=self.get_key(i)
            op=seqdict[curID]["options"]

            try:
                real_value=data[0][-1][2]
            except:
                continue
            recovered=self.recover(data[-1])
            recover_value=recovered[-1][-1]
            hidden = seqrnn.initHidden()
            data=data[:-1]
            for i in range(data.size()[0]):
                output, hidden = seqrnn(data[i].float(), hidden)
            output=output[0][0].item()
            data[-1][-1][-1]=output
            recover_output=self.recover(data[-1])[-1][-1]

            if len(op)==0:
                try:
                    if abs(int(recover_output)-recover_value)<0.01:
                        right+=1
                except:
                    # continue
                    print(recover_output)
                    print(int(recover_output))
                    print(recover_value)
            else:
                diff=[]
                for j in range(len(op)):
                    # try:
                    if isinstance(op[j], str) and "/" in op[j]:
                        field=op[j].split('/')
                        op[j]=float(field[0])/float(field[1])
                    elif isinstance(op[j], str) and "," in op[j]:
                        field=op[j].split(',')
                        op[j]=int(field[-1])
                    # print(float(op[j]))
                    diff.append(abs(recover_output-float(op[j])))
                    # except:
                        # print(op)
                if len(diff)==0:
                    print(op)
                approx=min(diff)
                idx=diff.index(approx)
                ans=op[idx]
                cnt+=1
                if abs(float(ans)-recover_value)<0.01:
                    right+=1
        return right/cnt


class SeqData_normbylen(Dataset):
    def __init__(self, data_dir, filename):
        files=os.path.join(data_dir, filename)
        inputdic=open(files, 'r')
        self.seqdict = json.load(inputdic)
        self.seqlist=[]
        self.consistent=[]
        self.keylist=[]
        for key in self.seqdict:
            onelist=[]
            stem=self.seqdict[key]['stem']
            # skip all next operations in for if sequence is illegal
            if stem[0][0].isalpha()==True or stem[0][0].isdigit()==False:
                continue 
            tmplist=stem.split(",")
            if len(tmplist)==1:
                tmplist=stem.split(" ")
            for i in range(len(tmplist)):
                if "/" in tmplist[i]:
                    field=tmplist[i].split('/')
                    tmplist[i]=float(field[0])/float(field[1])
                elif "√" in tmplist[i]:
                    # print(tmplist[i])
                    field=tmplist[i].strip(' ').split('√')
                    # print(field)
                    if field[0]=='':
                        tmplist[i]=math.sqrt(float(field[1]))
                    else:
                        tmplist[i]=float(field[0])*math.sqrt(float(field[1]))
                elif "^" in tmplist[i]:
                    field=tmplist[i].strip(' ').split('^')
                    tmplist[i]=float(field[0])**float(field[1])
            for i in range(len(tmplist)-2):
                # print(tmplist[i:i+4])
                try:
                    # self.seqlist.append(torch.Tensor(tmplist[i:i+4]))
                    tmplist=list(map(float,tmplist))
                    onelist.append(tmplist[i:i+3])
                    if tmplist[i]>1000000:
                        print(tmplist[i])
                    self.consistent.append(tmplist[i:i+3])
                except:
                    pass
            self.seqlist.append(onelist)
            self.keylist.append(key)
        self.df=df=pd.DataFrame(self.consistent)
        self.scaler=MinMaxScaler()
        self.scaler.fit_transform(self.df)
    def get_key(self,idx):
        return self.keylist[idx]
    def __getitem__(self):
        tmp=[]
        while len(tmp)<=1 or len(tmp[0])==0:
            index=np.random.randint(0,len(self.seqlist))
            tmp=np.copy(self.seqlist[index])
        res=[]
        for i in range(len(tmp)):
            if len(tmp[i])!=1:
                tmp[i]=np.array([tmp[i]])
                lengthes=[len(str(int(arr))) for arr in tmp[i]]
                maxlen=max(lengthes)
                if maxlen!=0:
                    for j in range(len(tmp[i])):
                        tmp[i][j]=tmp[i][j]/(10**maxlen)
                res.append([tmp[i]])
            else:
                lengthes=[len(arr) for arr in tmp[i][0]]
                maxlen=max(lengthes)
                if maxlen!=0:
                    for j in len(tmp[i][0]):
                        tmp[i][0][j]=tmp[i][0][j]/(10**maxlen)
                res.append(tmp[i])
        res=torch.tensor(res,dtype=torch.double)
        return res

    def gen_data(self,index):
        tmp=np.copy(self.seqlist[index])
        real_value=tmp[-1][-1]
        res=[]
        for i in range(len(tmp)):
            if len(tmp[i])!=1:
                tmp[i]=np.array([tmp[i]])
                lengthes=[len(str(int(arr))) for arr in tmp[i]]
                maxlen=max(lengthes)
                if maxlen!=0:
                    for j in range(len(tmp[i])):
                        tmp[i][j]=tmp[i][j]/(10**maxlen)
                res.append([tmp[i]])
            else:
                lengthes=[len(arr) for arr in tmp[i][0]]
                maxlen=max(lengthes)
                if maxlen!=0:
                    for j in len(tmp[i][0]):
                        tmp[i][0][j]=tmp[i][0][j]/(10**maxlen)
                res.append(tmp[i])
        res=torch.tensor(res,dtype=torch.double)
        return res,maxlen,real_value

    def __len__(self):
        return len(self.seqlist)
    #norm by len
    def cal_accuracy(self,seqrnn,seqdict):
        setlen=self.__len__()
        right=0
        cnt=0
        for i in range(setlen):
            if len(self.seqlist[i])==0:
                continue
            data,maxlen,real_value=self.gen_data(i)
            curID=self.get_key(i)
            op=seqdict[curID]["options"]
            
            hidden = seqrnn.initHidden()
            data=data[:-1]
            for i in range(data.size()[0]):
                output, hidden = seqrnn(data[i].float(), hidden)
            output=output[0][0].item()
            recover_output=output*(10**maxlen)

            if len(op)==0:
                try:
                    if abs(int(recover_output)-real_value)<0.01:
                        right+=1
                    cnt+=1
                except:
                    # continue
                    print(recover_output)
                    print(int(recover_output))
                    print(real_value)
            else:
                diff=[]
            
                for j in range(len(op)):
                    # try:
                    if isinstance(op[j], str) and "/" in op[j]:
                        field=op[j].split('/')
                        op[j]=float(field[0])/float(field[1])
                    elif isinstance(op[j], str) and "," in op[j]:
                        field=op[j].split(',')
                        op[j]=int(field[-1])
                    # print(float(op[j]))
                    diff.append(abs(recover_output-float(op[j])))
                if len(diff)==0:
                    print(op)
                approx=min(diff)
                idx=diff.index(approx)
                ans=op[idx]
                cnt+=1
                if abs(float(ans)-real_value)<0.01:
                    right+=1
        # print(cnt)
        return right/cnt


class SeqRNN(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(SeqRNN, self).__init__()

        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # print(input,hidden)
        # print('input shape',input.shape)
        # print('hidden shape',hidden.shape)
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        # output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

