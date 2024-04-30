import torch
import torch.nn.functional as F
#----------------step6: 定义训练方法-----------------------
def train_model(my_model, device, trains_loader, optimizers, epoches):
    # 模型训练
    my_model.train()
    for batch_idx, (data, target) in enumerate(trains_loader):
        # 将data和target部署到DEVICE上去
        data, target = data.to(device), target.to(device)
        # 将梯度初始化为0
        optimizers.zero_grad()
        # 训练所得的结果
        output = my_model(data)
        # 计算交叉熵损失
        loss = F.cross_entropy(output, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizers.step()
        # 每100batch_size打印一次log
        if batch_idx % 1000 == 0:
            print("Training Epoch:{} \t Loss:{:.5f}".format(epoches, loss.item()))