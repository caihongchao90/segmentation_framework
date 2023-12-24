import torch
import torchvision
from skimage import io
import os
import warnings
import numpy as np
from osgeo import gdal, gdalconst
warnings.filterwarnings("ignore")
totensor = torchvision.transforms.ToTensor()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def readTif(imagePath):
    '''
    读取路径上的影像，并返回图像瓦片数组、坐标信息等
    imagePath : 指定路径
    '''
    # 打开图像
    input_image = gdal.Open(imagePath, gdalconst.GA_ReadOnly)

    # 获取原始坐标信息
    original_projection = input_image.GetProjection()
    original_geo_transform = input_image.GetGeoTransform()

    # 获取图像的宽度和高度
    width = input_image.RasterXSize
    height = input_image.RasterYSize
    bands = input_image.RasterCount
    tile_data = np.zeros((width, height, bands), dtype=np.float32)
    for k in range(1, bands  + 1):
        band_data = input_image.GetRasterBand(k).ReadAsArray()
        tile_data[:, :, k - 1] = band_data
    # 关闭图片
    input_image = None
    return tile_data, (width, height, bands), (original_projection, original_geo_transform)

def save_preLabel(path_name, image, original_projection, original_geo_transform):
    '''
    保存带有坐标系的预测图片
    path_name : 图片路径 + 名字
    image : 保存的瓦片数据数组
    original_projection : 原始投影
    original_geo_transform : 原始地理变换信息
    '''
    (width, height, bands) = image.shape
    # 创建输出数据集
    driver = gdal.GetDriverByName('GTiff')
    output_merged_image = driver.Create(path_name, width, height,
                                        bands, gdalconst.GDT_Byte)
    # 设置输出数据集的坐标信息
    output_merged_image.SetProjection(original_projection)
    output_merged_image.SetGeoTransform(original_geo_transform)
    # 写入小图像块的数据
    for k in range(1, bands + 1):
        output_merged_image.GetRasterBand(k).WriteArray(image[:, :, k - 1])
    output_merged_image = None

def get_classify(last_output):
    '''
    对分类结果进行分类
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=="cuda":
        last_output = last_output.cpu().detach().numpy()
    else:
        last_output = last_output.detach().numpy()
    last_output[last_output>=0.2]=1
    last_output[last_output<0.2]=0
    return last_output

def fileExists(path):
    '''
    判断是否存在文件夹如果不存在则创建为文件夹
    path : 检查目标
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def perdiction_nomal(model, clip_result,pre_result):
    '''
    适用于预测 [没有坐标系] 的图片
    model : 加载好的模型
    clip_result : 切割好的图片存放路径
    pre_result : 预测结果存放路径
    '''
    rgbfolders = clip_result                             # 分割 rgb 结果路径，因为分割后自动生成该文件夹，所以不需要修改
    folders =  os.listdir(rgbfolders)
    for folder in folders:
        img = io.imread(rgbfolders + folder).astype(np.float32)
        # 转图片格式
        img = totensor(img)
        img_shape = img.shape
        # 1为batch值
        img = img.reshape(1,img_shape[0],img_shape[1],img_shape[2]) 
        img = img.to(device)

        output = model(img)

        output_class = get_classify(output)

        b,c,w,h = output_class.shape
        output_class = output_class.reshape(w,h,c)

        pre_unet_result_path = pre_result
        fileExists(pre_unet_result_path)

        # 保存结果图片
        io.imsave(pre_unet_result_path + folder, output_class)

def perdiction_withCoord(model, clip_result,pre_result):
    '''
    适用于预测 [含坐标系] 的图片
    model : 加载好的模型
    clip_result : 切割好的图片存放路径
    pre_result : 预测结果存放路径
    '''
    rgbfolders = clip_result                             # 分割 rgb 结果路径，因为分割后自动生成该文件夹，所以不需要修改
    folders =  os.listdir(rgbfolders)
    for folder in folders:
        img, (width, height, bands), (original_projection, original_geo_transform) = readTif(rgbfolders + folder)
        # 转图片格式
        img = totensor(img)
        img_shape = img.shape
        # 1为batch值
        img = img.reshape(1,img_shape[0],img_shape[1],img_shape[2]) 
        img = img.to(device)

        output = model(img)

        output_class = get_classify(output)

        b,c,w,h = output_class.shape
        output_class = output_class.reshape(w,h,c)

        pre_unet_result_path = pre_result
        fileExists(pre_unet_result_path)

        # 保存结果图片
        save_preLabel(pre_unet_result_path + folder, output_class, original_projection, original_geo_transform)