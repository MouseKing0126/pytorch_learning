#二分类问题
import torch  
import torch.nn.functional as F  
#F = torch.nn.functional它是 PyTorch 的函数式接口，专门放激活函数、损失函数等


#训练集构建

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0],[0],[1]])
#1，0两种分类


#模型代码

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred= F.sigmoid(self.linear(x))
        #把x做线性运算之后的结果z用sigmoid线性映射到0-1，激活函数
        return y_pred
    
model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(reduction='sum')
#BCELoss是交叉熵损失
#参数reduction:sum求和，mean求均值，none逐个输出loss

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


#训练代码

for epoch in range(1000):

    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    print('w = ',model.linear.weight.item())
    print('b = ',model.linear.bias.item())

    x_test = torch.Tensor([[4.0]])
    y_test = model(x_test)
    print('y_pred = ',y_test.data)