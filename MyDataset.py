from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

#load data
df = pd.read_csv('FinalProject/data.csv',header = None)
inputs = torch.from_numpy(np.array(df)[:,0:188])
targets = torch.from_numpy(np.array(df)[:,188].astype(np.int64)) - 1 #方便起见，这里把target改成了0123，为了后面onehot的时候可以是1x4的向量

#divide dataset as five pieces, and concrete them to realise 5-fold cross validation
data1 = np.array(df)[0:1705*1,0:188]
data2 = np.array(df)[1705*1:1705*2,0:188]
data3 = np.array(df)[1705*2:1705*3,0:188]
data4 = np.array(df)[1705*3:1705*4,0:188]
data5 = np.array(df)[1705*4:1705*5,0:188]
data_train1 =  np.concatenate((data1,data2,data3,data4),axis = 0)
data_test1 = data5
data_train2 =  np.concatenate((data5,data1,data2,data3),axis = 0)
data_test2 = data4
data_train3 =  np.concatenate((data4,data5,data1,data2),axis = 0)
data_test3 = data3
data_train4 =  np.concatenate((data3,data4,data5,data1),axis = 0)
data_test4 = data2
data_train5 =  np.concatenate((data2,data3,data4,data5),axis = 0)
data_test5 = data1

target1 = np.array(df)[0:1705*1,188]
target2 = np.array(df)[1705*1:1705*2,188]
target3 = np.array(df)[1705*2:1705*3,188]
target4 = np.array(df)[1705*3:1705*4,188]
target5 = np.array(df)[1705*4:1705*5,188]
target_train1 =  np.concatenate((target1,target2,target3,target4),axis = 0)
target_test1 = target5
target_train2 =  np.concatenate((target5,target1,target2,target3),axis = 0)
target_test2 = target4
target_train3 =  np.concatenate((target4,target5,target1,target2),axis = 0)
target_test3 = target3
target_train4 =  np.concatenate((target3,target4,target5,target1),axis = 0)
target_test4 = target2
target_train5 =  np.concatenate((target2,target3,target4,target5),axis = 0)
target_test5 = target1


label_train1 = torch.from_numpy(target_train1.astype(np.int64)) - 1
label_test1 = torch.from_numpy(target_test1.astype(np.int64)) - 1
label_train2 = torch.from_numpy(target_train2.astype(np.int64)) - 1
label_test2 = torch.from_numpy(target_test2.astype(np.int64)) - 1
label_train3 = torch.from_numpy(target_train3.astype(np.int64)) - 1
label_test3 = torch.from_numpy(target_test3.astype(np.int64)) - 1
label_train4 = torch.from_numpy(target_train4.astype(np.int64)) - 1
label_test4 = torch.from_numpy(target_test4.astype(np.int64)) - 1
label_train5 = torch.from_numpy(target_train5.astype(np.int64)) - 1
label_test5 = torch.from_numpy(target_test5.astype(np.int64)) - 1


def count(y,x):
    y = y.flatten()
    c = 0
    for i in y:
        if i == x:
            c = c + 1
    return c


class GetData1(Dataset):
    def __init__(self):
        self.data = data_train1  #取data_train集
        self.label = label_train1

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)


    def getTarget(self):
        return label_train1

    def getWeight(self):
        maximum = max(count(label_train1,0), count(label_train1,1), count(label_train1,2), count(label_train1,3))
        return torch.from_numpy(np.array([maximum/count(label_train1,0),maximum/count(label_train1,1),
        maximum/count(label_train1,2),maximum/count(label_train1,3)])).float()

class GetData2(Dataset):
    def __init__(self):
        self.data = data_train2  #取data_train集
        self.label = label_train2

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)


    def getTarget(self):
        return label_train2

    def getWeight(self):
        maximum = max(count(label_train2,0), count(label_train2,1), count(label_train2,2), count(label_train2,3))
        return torch.from_numpy(np.array([maximum/count(label_train2,0),maximum/count(label_train2,1),
        maximum/count(label_train2,2),maximum/count(label_train2,3)])).float()

class GetData3(Dataset):
    def __init__(self):
        self.data = data_train3  #取data_train集
        self.label = label_train3

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)


    def getTarget(self):
        return label_train3

    def getWeight(self):
        maximum = max(count(label_train3,0), count(label_train3,1), count(label_train3,2), count(label_train3,3))
        return torch.from_numpy(np.array([maximum/count(label_train3,0),maximum/count(label_train3,1),
        maximum/count(label_train3,2),maximum/count(label_train3,3)])).float()

class GetData4(Dataset):
    def __init__(self):
        self.data = data_train4  #取data_train集
        self.label = label_train4

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)


    def getTarget(self):
        return label_train4

    def getWeight(self):
        maximum = max(count(self.label,0), count(self.label,1), count(self.label,2), count(self.label,3))
        return torch.from_numpy(np.array([maximum/count(self.label,0),maximum/count(self.label,1),
        maximum/count(self.label,2),maximum/count(self.label,3)])).float()
class GetData5(Dataset):
    def __init__(self):
        self.data = data_train5  #取data_train集
        self.label = label_train5

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)

    def getTarget(self):
        return target_train5

    def getWeight(self):
        # print(count(label_train,0))
        # print(count(label_train,1))
        # print(count(label_train,2))
        # print(count(label_train,3))
        maximum = max(count(self.label,0), count(self.label,1), count(self.label,2), count(self.label,3))
        return torch.from_numpy(np.array([maximum/count(self.label,0),maximum/count(self.label,1),
        maximum/count(self.label,2),maximum/count(self.label,3)])).float()
    
class TestSet1(Dataset):
    def __init__(self):
        self.data = data_test1  #取data_train集
        self.label = label_test1

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)

    def getTrainNum(self):
        return 1705
    
    def getData(self):
        return self.data

    def getTarget(self):
        return self.label

class TestSet2(Dataset):
    def __init__(self):
        self.data = data_test2  #取data_train集
        self.label = label_test2

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)

    def getTrainNum(self):
        return 1705

    def getData(self):
        return self.data
    
    def getTarget(self):
        return self.label

class TestSet3(Dataset):
    def __init__(self):
        self.data = data_test3  #取data_train集
        self.label = label_test3

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)

    def getTrainNum(self):
        return 1705
    
    def getData(self):
        return self.data
    
    def getTarget(self):
        return self.label

class TestSet4(Dataset):
    def __init__(self):
        self.data = data_test4  #取data_train集
        self.label = label_test4

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)

    def getTrainNum(self):
        return 1705

    def getData(self):
        return self.data 
    
    def getTarget(self):
        return self.label

class TestSet5(Dataset):
    def __init__(self):
        self.data = data_test5  #取data_train集
        self.label = label_test5

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)

    def getTrainNum(self):
        return 1705
    
    def getData(self):
        return self.data

    def getTarget(self):
        return self.label