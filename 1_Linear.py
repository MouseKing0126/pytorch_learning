import torch  # 导入pytorch库

#训练集构建

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])


#模型代码

class LinearModel(torch.nn.Module):#父类是torch.nn.Module，里面有自动反馈求导的方法，也可以自己设计
    def __init__(self):
        #构造函数
        #super(LinearModel,self).__init__()调用父类（super就是父类）构造  PyTorch 所有自定义模型都要这行
        super().__init__()
        #现在这样写就可以
        self.linear = torch.nn.Linear(1,1)
        #构造对象，self.linear的类型是torch.nn这个模块下的Linear类
        #nn是neural network神经网络
        #（1，1）输入输出都是一维

    def forward(self,x):
    #前馈，函数重写覆盖掉父类里的forward函数
        y_pred= self.linear(x)
        #y hat定义运算，将x矩阵运算成y hat
        return y_pred
    
model = LinearModel()
#实例化类
#这个model是callable的，因为Module里定义了call，使得调用这个类的时候自动调用forward

criterion = torch.nn.MSELoss(reduction='sum')
#MSELoss均方误差损失
#参数reduction:sum求和，mean求均值，none逐个输出loss

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#实例化optimizer优化器，parameters梯度搜索找出要求的权重
#SGD是随机梯度下降


#训练代码

for epoch in range(100):
#训练轮次100
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch, loss.item())
    #这里自动做str(),不用担心生成计算图
    #必须加 item()，只取出数值，不保留计算图，防止显存泄露

    optimizer.zero_grad()
    #做梯度归零，把上一轮的梯度清零
    loss.backward()
    optimizer.step()
    #step函数是update更新梯度

    print('w = ',model.linear.weight.item())
    print('b = ',model.linear.bias.item())
    #输出权重w和偏置b

    x_test = torch.Tensor([[4.0]])
    #给一个测试的x的值4
    y_test = model(x_test)
    #输出训练结果得出的y
    print('y_pred = ',y_test.data)







