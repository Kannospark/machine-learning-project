from Attention import SelfAttention
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from MyDataset import GetData1,GetData2,GetData3,GetData4,GetData5,TestSet1,TestSet2,TestSet3,TestSet4,TestSet5
from torch.utils.data import DataLoader
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

# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers) # utilize the LSTM model in torch.nn 
        self.forwardCalculation = nn.Linear(hidden_size, output_size)
 
    def forward(self, _x):
        x, _ = self.lstm(_x)  
        x = self.forwardCalculation(x)
        return x

def check(x,y):  #计算
    j = 0
    y = y.flatten()
    correct = 0
    for i in x:
        if y[j] == i:
            correct = correct + 1
        j = j + 1
    return correct/len(x)

data_len = 200
inputNum = 188
outputNum = 4
batch_size = 50
learning_rate = 1e-3
max_epochs = 60  #60
set_seed(1000)
 

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
    lstm_model = LstmRNN(inputNum, 16, output_size=outputNum, num_layers=1).double() # 16 hidden units
    attention = SelfAttention(num_attention_heads = 1, input_size = 188, hidden_size = 1, hidden_dropout_prob = 0.3).double()
    dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = False, drop_last = True)
    weight = dataset.getWeight()
    loss_function = nn.CrossEntropyLoss(weight = weight)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

    i_iter = np.array([])
    loss_iter = np.array([])
    for epoch in range(max_epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            data = attention(data)
            optimizer.zero_grad()
            output = lstm_model(data)
            target2 = torch.zeros(batch_size, 4).scatter_(1, target.unsqueeze(1), 1)
            loss = loss_function(output.to(torch.float32), target2.to(torch.float32))
            loss.backward()
            optimizer.step()
            
        i_iter  = np.append(i_iter, epoch)
        loss_iter = np.append(loss_iter, loss.item())
        if (epoch+1) % 5 == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch+1, max_epochs, loss.item()))
    
    # prediction on training dataset
    dataloader_test = DataLoader(dataset = testset, batch_size = batch_size, shuffle = False, drop_last = True)
    sum = 0
    prediction = np.ones(1)
    for data, target in dataloader_test:    
        output = lstm_model(data)
        idx = output.detach().numpy().argmax(axis=1)
        out = np.zeros_like(output.detach().numpy(),dtype=float)
        out[np.arange(output.shape[0]), idx] = 1
        output = [one_label.tolist().index(1) for one_label in out]
        prediction = np.append(prediction, np.array(output).flatten())
        
        # sum = sum + check(output,target)
        # print(output)


    prediction = np.asarray(prediction).flatten()
    prediction = np.delete(prediction, 0)  

    print("this is F1 of normal:",F1(testset.getTarget(), prediction, 0))
    print("this is F1 of AF:",F1(testset.getTarget(), prediction, 1))
    print("this is F1 of other rhythms:",F1plus(testset.getTarget(), prediction, 2, 3))
    print("this is total F1:",(1/3)*(F1(testset.getTarget(), prediction, 0)+F1(testset.getTarget(), prediction, 1)+F1plus(testset.getTarget(), prediction, 2, 3)))
    # plt.figure()
    plt.subplot(5,1,i+1)
    plt.plot(i_iter,loss_iter)
    plt.title("loss")
plt.show()

