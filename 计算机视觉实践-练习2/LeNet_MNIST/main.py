import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import LeNet5
from train import train_model
from test import test_model

# --------------step1: 定义超参数-------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 是否用GPU
EPOCHS = 10  # 数据集的训练次数
BATCH_SIZE = 16  # 每批处理的数据 16/32/64/128

# -------------step2: 构建transform（对图像做处理）---------
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转成成tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化 => x' = （x-μ）/σ
])


def load_data():
    # -------------step3: 下载并加载数据集------------------
    # 下载数据集
    train_set = datasets.MNIST("data_sets", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("data_sets", train=False, download=True, transform=transform)
    # 加载数据集
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = load_data()
    # ----------------step:5 定义优化器--------------------------
    model = LeNet5().to(DEVICE)
    # optimizer = optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)
    # --------------step8: 训练模型----------------------------
    for epoch in range(1, EPOCHS + 1):
        train_model(model, DEVICE, train_loader, optimizer, epoch)
        test_model(model, DEVICE, test_loader)
