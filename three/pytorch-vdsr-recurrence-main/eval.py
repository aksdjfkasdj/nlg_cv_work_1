import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np

# 创建一个解析命令行参数的ArgumentParser对象，设置描述信息
parser = argparse.ArgumentParser(description="PyTorch VDSR Eval")
# 添加一个命令行参数，表示是否使用CUDA进行加速
parser.add_argument("--cuda", action="store_true", help="use cuda?")
# 添加一个命令行参数，表示模型的路径
parser.add_argument("--model", default="model/model_epoch_50.pth", type=str, help="model path")
# 添加一个命令行参数，表示数据集的名称
parser.add_argument("--dataset", default="Set5", type=str, help="dataset name, Default: Set5")
# 添加一个命令行参数，表示GPU的ID，默认为0
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")



def SSIM(pred, gt, shave_border=0, data_range=None):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    ssim = compare_ssim(pred, gt, multichannel=True, data_range=data_range)
    return ssim


# 定义一个函数，用于计算峰值信噪比
def PSNR(pred, gt, shave_border=0):
    # 计算图像的高度和宽度
    height, width = pred.shape[:2]
    # 对预测图像和原始图像进行裁剪，去除边界
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    # 计算预测图像与原始图像的差值
    imdff = pred - gt
    # 计算均方根误差
    rmse = math.sqrt(np.mean(imdff ** 2))
    # 如果均方根误差为0，则返回一个较大的值（100），避免除以0的错误
    if rmse == 0:
        return 100
    # 计算峰值信噪比并返回
    return 20 * math.log10(255.0 / rmse)


# 解析命令行参数，并将结果存储在opt对象中
opt = parser.parse_args()
# 检查是否使用CUDA
cuda = opt.cuda

# 如果使用CUDA
if cuda:
    # 打印正在使用的GPU ID
    print("=> use gpu id: '{}'".format(opt.gpus))
    # 设置CUDA可见的设备
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    # 如果没有找到可用的GPU，则抛出异常
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

# 加载预训练的模型
model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

# 定义图像的放大比例
scales = [2, 3, 4]

# 使用glob模块匹配指定数据集文件夹中的所有图像文件
image_list = glob.glob(opt.dataset + "_mat/*.*")


# 遍历每个放大比例
for scale in scales:
    # 初始化预测的平均峰值信噪比、基线方法（双三次插值）的平均峰值信噪比、处理每张图像所需的平均时间和图像计数器
    avg_psnr_predicted = 0.0
    avg_psnr_bicubic = 0.0
    avg_ssim_predicted = 0.0
    avg_elapsed_time = 0.0
    count = 0.0
    # 遍历数据集中的每张图像
    for image_name in image_list:
        # 如果当前图像的名称包含当前放大比例
        if str(scale) in image_name:
            # 增加图像计数器
            count += 1
            # 打印当前正在处理的图像名称
            print("Processing ", image_name)
            # 加载原始图像和低分辨率图像
            im_gt_y = sio.loadmat(image_name)['im_gt_y']
            im_b_y = sio.loadmat(image_name)['im_b_y']

            # 将图像转换为浮点类型
            im_gt_y = im_gt_y.astype(float)
            im_b_y = im_b_y.astype(float)

            # 计算基线方法（双三次插值）的峰值信噪比
            psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=scale)
            avg_psnr_bicubic += psnr_bicubic

            # 将低分辨率图像归一化到0-1之间
            im_input = im_b_y / 255.

            # 将图像转换为PyTorch Variable对象
            im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

            # 如果使用CUDA，则将模型和输入数据移动到GPU上
            if cuda:
                model = model.cuda()
                im_input = im_input.cuda()
            else:
                model = model.cpu()

            # 计时模型处理图像所需的时间
            start_time = time.time()
            HR = model(im_input)
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time

            # 将处理后的图像从GPU移到CPU
            HR = HR.cpu()

            # 将处理后的图像转换为NumPy数组并进行后处理
            im_h_y = HR.data[0].numpy().astype(np.float32)
            im_h_y = im_h_y * 255.
            im_h_y[im_h_y < 0] = 0
            im_h_y[im_h_y > 255.] = 255.
            im_h_y = im_h_y[0, :, :]

            # 计算处理后图像的峰值信噪比
            psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=scale)
            avg_psnr_predicted += psnr_predicted

            # 计算处理后图像的SSIM
            #ssim_predicted = SSIM(im_gt_y, im_h_y, shave_border=scale)
            ssim_predicted = SSIM(im_gt_y, im_h_y, shave_border=scale, data_range=1.0)
            avg_ssim_predicted += ssim_predicted

    # 打印每个放大比例下的评估结果
    print("Scale=", scale)
    print("Dataset=", opt.dataset)
    print("PSNR_predicted=", avg_psnr_predicted / count)
    print("PSNR_bicubic=", avg_psnr_bicubic / count)
    print("SSIM_predicted=", avg_ssim_predicted / count)
    print("It takes average {}s for processing".format(avg_elapsed_time / count))



# # 遍历每个放大比例
# for scale in scales:
#     # 初始化预测的平均峰值信噪比、基线方法（双三次插值）的平均峰值信噪比、处理每张图像所需的平均时间和图像计数器
#     avg_psnr_predicted = 0.0
#     avg_psnr_bicubic = 0.0
#     avg_elapsed_time = 0.0
#     count = 0.0
#     # 遍历数据集中的每张图像
#     for image_name in image_list:
#         # 如果当前图像的名称包含当前放大比例
#         if str(scale) in image_name:
#             # 增加图像计数器
#             count += 1
#             # 打印当前正在处理的图像名称
#             print("Processing ", image_name)
#             # 加载原始图像和低分辨率图像
#             im_gt_y = sio.loadmat(image_name)['im_gt_y']
#             im_b_y = sio.loadmat(image_name)['im_b_y']
#
#             # 将图像转换为浮点类型
#             im_gt_y = im_gt_y.astype(float)
#             im_b_y = im_b_y.astype(float)
#
#             # 计算基线方法（双三次插值）的峰值信噪比
#             psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=scale)
#             avg_psnr_bicubic += psnr_bicubic
#
#             # 将低分辨率图像归一化到0-1之间
#             im_input = im_b_y / 255.
#
#             # 将图像转换为PyTorch Variable对象
#             im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
#
#             # 如果使用CUDA，则将模型和输入数据移动到GPU上
#             if cuda:
#                 model = model.cuda()
#                 im_input = im_input.cuda()
#             else:
#                 model = model.cpu()
#
#             # 计时模型处理图像所需的时间
#             start_time = time.time()
#             HR = model(im_input)
#             elapsed_time = time.time() - start_time
#             avg_elapsed_time += elapsed_time
#
#             # 将处理后的图像从GPU移到CPU
#             HR = HR.cpu()
#
#             # 将处理后的图像转换为NumPy数组并进行后处理
#             im_h_y = HR.data[0].numpy().astype(np.float32)
#             im_h_y = im_h_y * 255.
#             im_h_y[im_h_y < 0] = 0
#             im_h_y[im_h_y > 255.] = 255.
#             im_h_y = im_h_y[0, :, :]
#
#             # 计算处理后图像的峰值信噪比
#             psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=scale)
#             avg_psnr_predicted += psnr_predicted
#
#     # 打印每个放大比例下的评估结果
#     print("Scale=", scale)
#     print("Dataset=", opt.dataset)
#     print("PSNR_predicted=", avg_psnr_predicted / count)
#     print("PSNR_bicubic=", avg_psnr_bicubic / count)
#     print("It takes average {}s for processing".format(avg_elapsed_time / count))
