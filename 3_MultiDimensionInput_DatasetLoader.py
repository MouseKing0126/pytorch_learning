import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader#加载数据
#dataset是抽象类，不能实例化，只能被子类继承


#加载数据集
class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        #可以加载所有数据，一个个传出单个数据;也可以只做初始化，把文件名做成列表，在getitem再读出来
        xy = np.loadtxt(filepath, delimiter =',', dtype=np.float32)
        self.len = xy.shape[0]
        #取(N,9)的第0个元素N
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])
        #加self的时候整个类都能用这个变量，不加只有init能用
    def __getitem__(self,index):#获取第index个数据
        return self.x_data[index],self.y_data[index]

    def __len__(self):#获取数据集条数
        return self.len
#魔法方法

if __name__ == '__main__':  
#  多进程加载数据 num_workers>0时，Windows 系统跑多进程必须把主程序包在这里面
    dataset = DiabetesDataset('D:/apytorchLearning/diabetes.csv')
    train_loader = DataLoader(dataset=dataset,
                          batch_size=32,#一个batch大小，一组32条数据
                          shuffle=True,
                          num_workers=0)
                          #这里设置成0的时候，实际跑的会更快，因为数据量很小
    #加载器，dataset传递数据集，批量，打乱，几个并行的数据，看cpu核心有几个
    #shuffle是打乱样本之后分割batch

    #模型
    class DiabetesModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.net = torch.nn.Sequential(
                torch.nn.Linear(8, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 8),
                torch.nn.ReLU(),
                torch.nn.Linear(8, 1),
            )

        def forward(self, x):
            
            return self.net(x)

    
    model = DiabetesModel()

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #训练
    for epoch in range(1000):
        epoch_loss = 0.0#为了打印每轮的loss
        for i, data in enumerate(train_loader,0):
            #自动组成两个矩阵的元组
            #把loader里的batch一组一组拿到data里面
            inputs, labels = data#inputs=x,labels=y标签，两个矩阵
        
            y_pred = model(inputs)
            loss = criterion(y_pred,labels)
            epoch_loss += loss.item()
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, epoch_loss)

           
