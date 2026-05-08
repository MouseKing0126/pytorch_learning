import torch
from torchvision import transforms
#图像处理
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim



#===========================数据加载=============================
batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307),(0.3081))])
#对图像进行处理Compose把图像转变成w*h*c的张量，取值变为0-1；
# Normalize标准化：（（均值），（标准差））自己算，使其满足正态分布
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               download=True,
                               transform=transform)

train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)
#这里已经设置好一批多少样本，而且datase拿到的mnist已经分好（图片，标签）

test_dataset = datasets.MNIST(root='./data',
                               train=False,
                               download=True,
                               transform=transform)

test_loader = DataLoader(test_dataset,
                          shuffle=False,
                          batch_size=batch_size)

#模型
class Net(torch.nn.Module):
        def __init__(self):
            super(Net,self).__init__()
            self.conv1 = torch.nn.Conv2d(1,10,kernel_size=5)
            self.conv2 = torch.nn.Conv2d(10,20,kernel_size=5)
            self.pooling = torch.nn.MaxPool2d(2)
            self.fc  = torch.nn.Linear(320,10)
            #展开，自己算一下20*4*4

        def forward(self, x):
            batch_size = x.size(0)
            x = F.relu(self.pooling(self.conv1(x)))
            x = F.relu(self.pooling(self.conv2(x)))
            #先做relu再做池化也可以
            x = x.view(batch_size,-1)
            #展开，-1是不确定多少自己算
            x = self.fc(x)
            return x

model=Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
     print('GPU ok')
model.to(device)
#把模型都放到gpu

#==============损失和优化器======================
criterion = torch.nn.CrossEntropyLoss()
#交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#===============训练过程=======================

def train(epoch):
     running_loss =0.0
     for batch_idx,data in enumerate(train_loader,0):
          inputs,target =data
          inputs,target = inputs.to(device),target.to(device)
          #运算的数据也要放在同一块显卡
          optimizer.zero_grad()

          outputs = model(inputs)
          loss = criterion(outputs,target)
          loss.backward()
          optimizer.step()

          running_loss += loss.item()

          if batch_idx %300 == 299:
               print('[%d,%5d] loss :%.3f' % (epoch +1 ,batch_idx+1,running_loss/300))
     
#=====================测试=========================
def test():
    correct = 0
    total =0
    with torch.no_grad():
          for data in test_loader:
               inputs, target = data
               inputs,target = inputs.to(device),target.to(device)
          #运算的数据也要放在同一块显卡
               outputs= model(inputs)
               _, predictd = torch.max(outputs.data,dim=1)#沿着第一个维度，列是第0个维度
               #_是最大值是多少，predicate是最大值的下标是多少
               total += target.size(0)
               #size()返回几行几列（0）返回第0维大小，也就是列
               correct += (predictd == target).sum().item()
               #张量之间的比较运算，每一个都比较，对了就是1，然后相加
    print('Accuracy on test set:%d %%' % (100 *correct /total))

#封装了训练和测试
if __name__ == '__main__':
     for epoch in range(10):
          train(epoch)
          test()
