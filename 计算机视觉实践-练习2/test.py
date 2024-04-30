import torch
import torch.nn.functional as F
# ----------------step7: 定义测试方法------------------------
def test_model(my_model, device, test_loder):
    my_model.eval()  # 模型验证
    correct = 0.0    # 正确率
    test_loss = 0.0   # 测试损失
    with torch.no_grad():  # 测试时不计算梯度，也不进行反向传播
        for data, target in test_loder:
            # 将data和target部署到device上
            data, target = data.to(device), target.to(device)
            # 测试所得的结果
            output = my_model(data)
            # 计算交叉熵损失
            test_loss += F.cross_entropy(output, target).item()
            # 找到概率最大的下标
            predict = output.argmax(dim=1)
            # predict = torch.max(output, dim=1)
            correct += predict.eq(target.view_as(predict)).sum().item()  # 累计正确的值
        # 计算平均损失
        avg_loss = test_loss / len(test_loder.dataset)
        # 计算准确率
        correct_ratio = 100 * correct / len(test_loder.dataset)
        print("Average_loss in test:{:.5f}\t Accuracy:{:.5f}\n".format(
            avg_loss, correct_ratio
        ))
