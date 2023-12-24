import numpy as np
import os
from skimage import io
from osgeo import gdal, gdalconst
import os
import numpy as np

# 读取文件夹下所有图片的名称
def mergeImage_normal(pic_path, pic_target, target_size, overlap = 0 ):
    '''
    合并 [不含] 坐标系的图片，包括样本rgb和标签
    Parameters:
    - pic_path : 裁剪图片所在路径
    - pic_target : 输出图片路径 + 名字
    - target_size : 元组，如(1024, 1024),因为 [有重叠] 的裁剪之后，影像范围不一定裁剪完整，为得到和裁剪的影像一样大小,所以需传入影像大小
    - overlap : 图片间的重叠像素个数，必须是偶数

    Returns:
    没有返回值
    ''' 
    half_overlap = int(overlap / 2)
    pic_target = pic_target
    num_width_list = []
    num_lenght_list = []
    picture_names =  os.listdir(pic_path)
    if len(picture_names)==0:
        print("没有文件")
    else:
        # 获取分割后图片的尺寸
        img_1_1 = io.imread(pic_path + picture_names[0])
        (width, length, depth) = img_1_1.shape
        img_1_1 = None
        # 分割名字获得行数和列数，通过数组保存分割后图片的列数和行数
        for picture_name in picture_names:
            num_width_list.append(int(picture_name.split("_")[0]))
            num_lenght_list.append(int((picture_name.split("_")[-1]).split(".")[0]))
        # 取其中的最大值
        num_width = max(num_width_list)
        num_length = max(num_lenght_list)
        
        # 预生成拼接后的图片
        splicing_pic = np.zeros((target_size[0], target_size[1], depth),dtype="u8")
        need_width = width - overlap
        need_length = length - overlap
        # 循环复制
        for i in range(1, num_width+1):
            for j in range(1, num_length+1):
                img_part = io.imread(pic_path + '{}_{}.tif'.format(i, j))
                splicing_pic[need_width*(i-1)+half_overlap: need_width*i+half_overlap, need_length*(j-1)+half_overlap : need_length*j+half_overlap, :] = img_part[half_overlap:width-half_overlap, half_overlap:length-half_overlap,:]
                img_part = None
        # 保存图片，大功告成
        splicing_pic = splicing_pic.astype("uint8")
        io.imsave(pic_target, splicing_pic)

def mergeImage_withCoord(pic_path, pic_target, overlap = 0):
    '''
    合并 [带] 坐标系的图片，包括样本rgb和标签
    pic_path   : 裁剪图片所在路径
    pic_target : 输出图片路径 + 名字
    overlap    : 图片间的重叠像素个数，必须是偶数
    ''' 
    if (overlap !=0) & (overlap % 2 != 0):
        print("overlap must be an even number")
        return
    half_overlap = int(overlap / 2)
    # 输入小图像块文件夹路径
    input_folder = pic_path

    # 输出合并后的图像路径
    output_merged_image_path = pic_target

    # 获取小图像块文件列表
    input_tiles = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

    # 获取小图像块的行号和列号
    tile_indices = [(int(name.split('_')[0]), int(name.split('_')[1].split('.')[0])) for name in input_tiles]

    # 根据行号和列号排序小图像块
    sorted_input_tiles = sorted(zip(tile_indices, input_tiles))

    # 打开第一个小图像块获取坐标信息
    first_tile_path = os.path.join(input_folder, sorted_input_tiles[0][1])
    first_tile = gdal.Open(first_tile_path, gdalconst.GA_ReadOnly)
    original_projection = first_tile.GetProjection()
    original_geo_transform = first_tile.GetGeoTransform()
    width = first_tile.RasterXSize
    height = first_tile.RasterYSize
    bands = first_tile.RasterCount

    # 计算输出数据集的大小
    num_rows = len(set([i for (i, j), name in sorted_input_tiles]))
    num_cols = len(set([j for (i, j), name in sorted_input_tiles]))

    # 计算合并后的图像大小
    merged_width = num_cols * (width - overlap)  
    merged_height = num_rows * (height - overlap)

    # 创建输出数据集
    driver = gdal.GetDriverByName('GTiff')
    output_merged_image = driver.Create(output_merged_image_path, merged_width, merged_height,
                                        bands, gdalconst.GDT_Byte)
    # 设置输出数据集的坐标信息
    output_merged_image.SetProjection(original_projection)
    output_merged_image.SetGeoTransform((original_geo_transform[0] + half_overlap * original_geo_transform[1],
                                        original_geo_transform[1],
                                        original_geo_transform[2],
                                        original_geo_transform[3] + half_overlap * original_geo_transform[5],
                                        original_geo_transform[4],
                                        original_geo_transform[5]))

    # 逐个读取并写入小图像块的数据
    for i, (indices, input_tile) in enumerate(sorted_input_tiles, start=1):
        input_tile_path = os.path.join(input_folder, input_tile)
        input_tile_data = gdal.Open(input_tile_path, gdalconst.GA_ReadOnly)

        # 读取小图像块的数据
        for j in range(1, bands + 1):
            data = input_tile_data.GetRasterBand(j).ReadAsArray()
            output_merged_image.GetRasterBand(j).WriteArray(data[(half_overlap):(width - half_overlap),(half_overlap):(height - half_overlap)],
                                                            xoff=(indices[1]) * (width - overlap),
                                                            yoff=(indices[0]) * (height - overlap))

        # 关闭小图像块数据集
        input_tile_data = None

    # 关闭输出数据集
    first_tile = None
    output_merged_image = None