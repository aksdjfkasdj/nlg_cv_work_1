# 导入所需的库
import argparse, os
import torch
from torch.autograd import Variable
from imageio import imread
from PIL import Image
import numpy as np
import time, math
import matplotlib.pyplot as plt

# 定义命令行参数
parser = argparse.ArgumentParser(description="PyTorch VDSR Demo")
parser.add_argument("--cuda", default=True, help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--image", default="butterfly_GT", type=str, help="image name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

# 定义PSNR计算函数
def PSNR(pred, gt, shave_border=0):
    # PSNR是一种评价图像质量的指标，值越大表示图像质量越好
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

# 定义图像色彩空间转换函数
def colorize(y, ycbcr):
    # 这个函数将Y通道和CbCr通道合并，得到YCbCr格式的图像，然后将其转换为RGB格式
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img

# 解析命令行参数
opt = parser.parse_args()
cuda = opt.cuda

# 检查是否使用CUDA，并设置GPU设备
if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

# 加载预训练的模型
model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

# 加载图像，并将其转换为YCbCr格式
im_gt_ycbcr = imread("Set5/" + opt.image + ".bmp", pilmode="YCbCr")
im_b_ycbcr = imread("Set5/"+ opt.image + "_scale_"+ str(opt.scale) + ".bmp", pilmode="YCbCr")

# 提取Y通道
im_gt_y = im_gt_ycbcr[:,:,0].astype(float)
im_b_y = im_b_ycbcr[:,:,0].astype(float)

# 计算bicubic插值的PSNR
psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=opt.scale)

# 将图像转换为模型的输入格式
im_input = im_b_y/255.
im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

# 将模型和输入数据移动到GPU
if cuda:
    model = model.cuda()
    im_input = im_input.cuda()
else:
    model = model.cpu()

# 记录开始时间，然后进行模型推理
start_time = time.time()
out = model(im_input)
# 记录结束时间，然后计算推理时间
elapsed_time = time.time() - start_time

# 将输出数据移动到CPU
out = out.cpu()

# 处理输出图像
im_h_y = out.data[0].numpy().astype(np.float32)
im_h_y = im_h_y * 255.
im_h_y[im_h_y < 0] = 0
im_h_y[im_h_y > 255.] = 255.

# 计算VDSR预测的PSNR
psnr_predicted = PSNR(im_gt_y, im_h_y[0,:,:], shave_border=opt.scale)

# 将Y通道和CbCr通道合并，得到彩色图像
im_h = colorize(im_h_y[0,:,:], im_b_ycbcr)
im_gt = Image.fromarray(im_gt_ycbcr, "YCbCr").convert("RGB")
im_b = Image.fromarray(im_b_ycbcr, "YCbCr").convert("RGB")

# 打印结果
print("Scale=",opt.scale)
print("PSNR_predicted=", psnr_predicted)
print("PSNR_bicubic=", psnr_bicubic)
print("It takes {}s for processing".format(elapsed_time))

# 显示图像
fig = plt.figure()
ax = plt.subplot(131)
ax.imshow(im_gt)
ax.set_title("GT")

ax = plt.subplot(132)
ax.imshow(im_b)
ax.set_title("Input(bicubic)")

ax = plt.subplot(133)
ax.imshow(im_h)
ax.set_title("Output(vdsr)")
plt.show()
