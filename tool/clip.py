import numpy as np
import os
from skimage import io
from osgeo import gdal, gdalconst
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")
class MyCustomError(Exception):
    def __init__(self, message="发生自定义错误"):
        self.message = message
        super().__init__(self.message)

def isEven(value):
    if (value != 0)&(value % 2 != 0):
        raise MyCustomError("重叠区像素个数必须为偶数")

def clipImage_nomal(pic_path, pic_target, size = 256, overlap = 0):
    '''
    将影像裁剪为指定大小，[没有] 坐标系，也可以用来裁剪标签（不建议这么使用）
    Parameters:
    - pic_path   : 目标影像位置
    - pic_target : 保存路径
    - size       : 裁剪出来的图片大小
    - overlap    : 重叠像素数，默认为0，即不重叠

    Returns:
    没有返回值
    '''
    try:
        isEven(overlap)
    except MyCustomError as e:
        print(f"错误: {e}")
        return
    if not os.path.exists(pic_target):  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(pic_target)
    #要分割后的尺寸
    cut_width = size
    cut_length = size
    # 读取要分割的图片，以及其尺寸等数据
    picture = io.imread(pic_path)
    # 如果形状只长宽，即是标签，else是RGB图片
    if len(picture.shape) == 2:
        (width, length) = picture.shape
        depth = 1
    else:
        (width, length, depth) = picture.shape
    # 预处理生成0矩阵
    pic = np.zeros((cut_width, cut_length, depth))
    # 计算可以划分的横纵的个数(width - overlap) // (tile_size[0] - overlap)
    num_width = int((width-overlap) / (cut_width-overlap))
    num_length = int((length-overlap) / (cut_length-overlap))
    # for循环迭代生成
    for i in range(num_width):
        for j in range(num_length):
            if depth == 1:
                pic = picture[i*(cut_width-overlap) : i*(cut_width-overlap)+cut_width, j*(cut_length-overlap) : j*(cut_length-overlap)+cut_length]
            else:
                 pic = picture[i*(cut_width-overlap) : i*(cut_width-overlap)+cut_width, j*(cut_length-overlap) : j*(cut_length-overlap)+cut_length, 0:3]  
            result_path = pic_target + '{}_{}.tif'.format(i+1, j+1)
            io.imsave(result_path, pic)
    picture = None
 
def clipImage_withCoord(input_image_path, output_folder, tile_size = 256, overlap = 0, bands = 3):
    '''
    带坐标系的方法进行裁剪，[含] 坐标系，该函数不可用于标签裁剪
    Parameters:
    - input_image_path : 影像路径
    - output_folder    : 裁剪图片保存路径
    - tile_size        : 裁剪图片目标大小
    - overlap          : 图片间的重叠像素个数
    - no_data_value    : 没有数据的部分设置为该值

    Returns:
    没有返回值
    '''
    # 输入图像路径
    input_image_path = input_image_path

    # 输出文件夹路径
    output_folder = output_folder
    # 定义裁剪的小图像块大小
    tile_size = (tile_size, tile_size)

    # 打开图像
    input_image = gdal.Open(input_image_path, gdalconst.GA_ReadOnly)
    bands = bands
    
    # 获取原始坐标信息
    original_projection = input_image.GetProjection()
    original_geo_transform = input_image.GetGeoTransform()

    # 获取图像的宽度和高度
    width = input_image.RasterXSize
    height = input_image.RasterYSize

    # 计算裁剪小图像块的数量
    num_tiles_x = (width - overlap) // (tile_size[0] - overlap)
    num_tiles_y = (height - overlap) // (tile_size[1] - overlap)

    # 循环裁剪小图像块
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            # 计算当前小图像块的左上角坐标
            ulx = i * (tile_size[0] - overlap)
            uly = j * (tile_size[1] - overlap)

            # 计算当前小图像块的右下角坐标
            lrx = ulx + tile_size[0]
            lry = uly + tile_size[1]

            # 读取小图像块的数据
            tile_data = np.zeros((tile_size[0], tile_size[1], bands), dtype=np.float32)
            if bands == 1:
                print(f"裁剪影像的通道数为 {bands}，该函数不可用于标签裁剪")
                return "Function ended early."
            for k in range(1, bands + 1):
                # 获取到数据之后，指定无数据像素的值
                band_data = input_image.GetRasterBand(k)
                band_data = band_data.ReadAsArray(ulx, uly, tile_size[0], tile_size[1])
                tile_data[:, :, k - 1] = band_data

            # 创建输出数据集
            driver = gdal.GetDriverByName('GTiff')
            output_tile_path = f'{output_folder}/{j}_{i}.tif'
            output_tile = driver.Create(output_tile_path, tile_size[0], tile_size[1], bands, gdalconst.GDT_Float32)

            # 设置输出数据集的坐标信息
            output_tile.SetProjection(original_projection)
            output_tile.SetGeoTransform((original_geo_transform[0] + ulx * original_geo_transform[1],
                                         original_geo_transform[1],
                                         original_geo_transform[2],
                                         original_geo_transform[3] + uly * original_geo_transform[5],
                                         original_geo_transform[4],
                                         original_geo_transform[5]))

            # 写入小图像块的数据
            for k in range(1, bands + 1):
                output_tile.GetRasterBand(k).WriteArray(tile_data[:, :, k - 1])

            # 关闭输出数据集
            output_tile = None
    
    # 关闭输入数据集
    input_image = None


