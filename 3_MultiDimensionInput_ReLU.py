# 多维输入：ReLU 版本
import torch
import numpy as np


# 固定随机种子：每次运行时，Linear 层的随机初始化尽量保持一致，方便对比训练曲线。
torch.manual_seed(0)

xy = np.loadtxt("D:/apytorchLearning/diabetes.csv", delimiter=",", dtype=np.float32)

# 最后一行当测试数据，其余数据用于训练
x_data = torch.from_numpy(xy[:-1, :-1])
y_data = torch.from_numpy(xy[:-1, [-1]])
x_test = torch.from_numpy(xy[[-1], :-1])
y_correct = torch.from_numpy(xy[[-1], [-1]])


class DiabetesModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # torch.nn.Sequential 可以把多层网络按顺序打包成一个“net 块”。
        # forward 里调用 self.net(x) 时，数据会按下面的顺序依次经过：
        # Linear(8,16) -> ReLU -> Linear(16,8) -> ReLU -> Linear(8,1)
        # 这样比手动写 self.linear1、self.linear2、self.linear3 更简洁。
        self.net = torch.nn.Sequential(
            # 输入数据有 8 个特征，所以第一层输入维度是 8。
            # 16 是隐藏层神经元个数，可以自己调整，不是固定答案。
            torch.nn.Linear(8, 16),

            # ReLU 会把负数变成 0，正数保持不变。
            # 它通常比 Sigmoid 更不容易在隐藏层出现梯度消失。
            torch.nn.ReLU(),

            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),

            # 最后一层输出 1 个值，用来表示二分类的原始分数。
            # 这里先不接 Sigmoid，因为下面的 BCEWithLogitsLoss 会一起处理。
            torch.nn.Linear(8, 1),
        )

    def forward(self, x):
        # logits：还没有经过 Sigmoid 的原始输出。
        # 训练时把 logits 交给 BCEWithLogitsLoss，比“Sigmoid + BCELoss”数值更稳定。
        return self.net(x)


model = DiabetesModel()

# BCEWithLogitsLoss = Sigmoid + BCELoss 的稳定版本。
# 所以模型最后一层不要再手动加 Sigmoid，否则相当于重复处理。
criterion = torch.nn.BCEWithLogitsLoss()

# Adam 会自动调节每个参数的学习步长（动量），通常比普通 SGD 更容易先跑出平稳曲线。
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


for epoch in range(300):
    logits = model(x_data)
    loss = criterion(logits, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    # 测试/打印概率时不需要计算梯度，用 no_grad 可以省内存和计算。
    with torch.no_grad():
        # 训练时输出 logits；真正想看 0-1 概率时，再手动做 Sigmoid。
        prob = torch.sigmoid(model(x_test))
        print(epoch, "loss =", loss.item(), "y_test =", prob.item())

print("y_correct =", y_correct.item())
