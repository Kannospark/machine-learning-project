import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from MyDataset import GetData1,GetData2,GetData3,GetData4,GetData5,TestSet1,TestSet2,TestSet3,TestSet4,TestSet5
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def set_seed(seed):
    torch.manual_seed(seed)  # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # gpu 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
def Recall(x,y,correct):
    x = x.flatten()  #x是个高维数组，把他变成一维
    xsum = 0
    ysum = 0
    # print(x.shape)
    # print(y.shape)
    j = 0
    for i in y:
        if i == correct:
            ysum = ysum + 1
            if x[j] == correct:
                xsum = xsum + 1
        j = j + 1
    return float(xsum/ysum)
def Recallplus(x,y,correct1,correct2):
    x = x.flatten()  #x是个高维数组，把他变成一维
    xsum = 0
    ysum = 0
    j = 0
    for i in y:
        if i == (correct1 or correct2):
            ysum = ysum + 1
            if x[j] == (correct1 or correct2):
                xsum = xsum + 1
        j = j + 1
    return float(xsum/ysum)
def Precision(x,y,correct):
    x = x.flatten()
    xsum = 0
    ysum = 0
    j = 0
    for i in y:
        if i == correct:
            ysum = ysum + 1
            if x[j] == correct:
                xsum = xsum + 1
        j = j + 1
    return float(xsum/ysum)
def Precisionplus(x,y,correct1,correct2):
    x = x.flatten()
    xsum = 0
    ysum = 0
    j = 0
    for i in y:
        if i == (correct1 or correct2):
            ysum = ysum + 1
            if x[j] == (correct1 or correct2):
                xsum = xsum + 1
        j = j + 1
    return float(xsum/ysum)
def F1(x,y,correct):  #大数组写前面
    r = Recall(x,y,correct)
    p = Precision(x,y,correct)
    return 2*p*r/(p+r)
def F1plus(x,y,correct1,correct2):  #大数组写前面
    r = Recallplus(x,y,correct1,correct2)
    p = Precisionplus(x,y,correct1,correct2)
    return 2*p*r/(p+r)
def check(x,y):
    j = 0
    y = y.flatten()
    correct = 0
    for i in x:
        if y[j] == i:
            correct = correct + 1
        j = j + 1
    return correct/len(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(188, 100)
        self.fc2 = nn.Linear(100, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 

for i in range(5):
    if i == 0 : dataset = GetData1()
    testset = TestSet1()
    if i == 1 : dataset = GetData2()
    testset = TestSet2()
    if i == 2 : dataset = GetData3()
    testset = TestSet3()
    if i == 3 : dataset = GetData4()
    testset = TestSet4()
    if i == 4 : dataset = GetData5()
    testset = TestSet5()
    
    print()
    print("it is training set of",i+1)
    batch_size = 20
    dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = False, drop_last = True)
    learning_rate = 1e-4
    epochs = 100  #40
    set_seed(1000)

    model = Net().double()
    
    # create a stochastic gradient descent optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # create a loss function
    weight = dataset.getWeight()
    criterion = nn.CrossEntropyLoss(weight = weight)
    i_iter = np.array([])
    loss_iter = np.array([])

    
    # run the main training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            net_out = model(data)
            target2 = torch.zeros(batch_size, 4).scatter_(1, target.unsqueeze(1), 1)
            loss = criterion(net_out.to(torch.float32), target2.to(torch.float32))
            loss.backward()
            optimizer.step()
        i_iter  = np.append(i_iter, epoch)
        loss_iter = np.append(loss_iter, loss.item())
        if epoch % 5 == 4:
                print('Epoch [{}/{}],loss: {:.4f}'.format(epoch+1,epochs,loss.item())) 
        

    output = model(torch.from_numpy(testset.getData()))

    idx = output.detach().numpy().argmax(axis=1)
    out = np.zeros_like(output.detach().numpy(),dtype=float)
    out[np.arange(output.shape[0]), idx] = 1
    output = [one_label.tolist().index(1) for one_label in out]
    output = np.array(output).flatten()
    # print(output)
    # print(dataset.getTarget())
    # print(check(output,dataset.getTarget()))


    print("this is F1 of normal:",F1(testset.getTarget(), output, 0))
    print("this is F1 of AF:",F1(testset.getTarget(), output, 1))
    print("this is F1 of other rhythms:",F1plus(testset.getTarget(), output, 2, 3))
    print("this is total F1:",(1/3)*(F1(testset.getTarget(), output, 0)+F1(testset.getTarget(), output, 1)+F1plus(testset.getTarget(), output, 2, 3)))
    plt.subplot(5,1,i+1)
    plt.plot(i_iter,loss_iter)
    plt.title("loss of Regression")
plt.show()