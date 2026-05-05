#多维输入
import torch
import torch.nn.functional as F  
import numpy as np

xy = np.loadtxt('D:/apytorchLearning/diabetes.csv', delimiter=',', dtype=np.float32)
#糖尿病的数据
x_data = torch.from_numpy(xy[:-1,:-1])
#，前是行，最后一列取为测试，，后是列，最后一列不要，是结果
y_data = torch.from_numpy(xy[:-1,[-1]])
#列只要最后一列，加[]为了拿出的是矩阵

#模型代码

class DiabetesModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.activate = torch.nn.Sigmoid()
        #这里Sigmoid大写是因为是一个类，之后可以使用里面callable的函数sigmoid
        #尝试多种激活函数比如relu，但注意最后一次要是sigmoid，才能保持0-1

    def forward(self,x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))

        return x
    
model = DiabetesModel()

criterion = torch.nn.BCELoss(reduction='sum')
#BCELoss是交叉熵损失
#参数reduction:sum求和，mean求均值，none逐个输出loss

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


#训练代码

for epoch in range(300):

    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    #print('w = ',model.linear.weight.item())
    #print('b = ',model.linear.bias.item())
    #之后复杂的参数就不需要都看了，选择打印即可

    x_test = torch.from_numpy(xy[[-1],:-1])
    y_test = model(x_test)
    print('y_test = ',y_test.data)


y_correct = torch.from_numpy(xy[[-1],[-1]])
print('y_correct = ',y_correct.data)


#训练会发现这个准确率不怎么高，可能是由于Sigmoid 的导数最大只有 0.25，导致向前传播梯度消失，之后会加入relu进行尝试