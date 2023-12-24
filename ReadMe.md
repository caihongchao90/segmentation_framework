# 介绍：

​	该项目集成了深度学习训练一个语义分割模型所需要的内容，从样本数据的获取、模型的训练、再到模型最终的应用，均有详细代码

项目中用到的库，及版本号如下：

`gdal`的安装，本项目采用`whl`文件进行安装

`torch`和`gdal`依赖的`numpy`版本有可能不一样，这或许会导致一部分问题，此处提供了我们使用的库版本，供参考！

| 库          | 项目中作用                               | 版本号（使用 {库名}.\__version__ 命令输出） |
| ----------- | ---------------------------------------- | ------------------------------------------- |
| python      | ......                                   | 3.9.13                                      |
| **torch**   | 搭建神经网络、训练等                     | 1.12.1+cu116                                |
| numpy       | ......                                   | 1.26.2                                      |
| torchvision | ......                                   | 0.13.1+cu116                                |
| **gdal**    | 主要用来处理影像的坐标系                 | 3.4.3                                       |
| **skimage** | 因为最开始没考虑坐标系，所以使用该库处理 | 0.19.3                                      |
| matplotlib  | 绘制分类结果                             | 3.5.3                                       |
| random      | 用来随机采样，获取样本数据集             | 随便吧                                      |

# 一、目录结构

​	由于训练期间是学习样本和标签之间的映射关系，所以该阶段不需要考虑图片坐标

目录结构如下：

```js
Tree_seg/
|-- dataset    	
|-- model
|-- prediction
|-- tool
|-- Train.ipynb
```

`dataset` : 训练期间使用到的样本

`model` : 训练好的模型保存路径

`prediction` : 预测阶段，分为`生产`、`验证`

`tool` : 预测阶段需要用到的一些工具

`Train.ipynb` : 训练的主文件

# 二、制作样本

​	`dataset` 文件夹里的`MakeDataset.ipynb`文件，用来制作训练样本

# 三、训练

`Train.ipynb` 文件里有模型训练需要的内容

`model`文件夹，用来存放训练好的模型

# 四、预测
`prediction` 文件夹里，<验证>文件夹是用来验证模型效果的，为256*256小图的分割效果、和大图效果（大图大小可以自己定义）；<生产>文件是用来直接预测的,分割、预测、合并一条龙。
这一阶段需要考虑到影像的坐标
`tool`文件夹里，有分割、预测、合并的代码，分为两套，一套是无坐标的，一套是有坐标的；
**无坐标：**
```js
tool.clipImage_nomal()
tool.perdiction_nomal()
tool.mergeImage_normal()
```
**有坐标：**
```js
tool.clipImage_withCoord()
tool.perdiction_withCoord()
tool.mergeImage_withCoord()
```

分为两种：

​	因为没有重叠的话，会在两张 `256*256` 图片拼接的地方出现很明显的拼接缝隙，拼接痕迹明显。

- 有重叠的方法，按下图方式进行重叠

**无坐标** : 该方法在裁剪影像时，不一定把影像裁剪完整；所以预测完成，拼接时，可能会缺失边缘很少一部分（影像大的话）；所以需要提前获取到该影像的大小，以便在拼接的时候输出和该影像大小一致的掩膜（边缘有黑边）

**有坐标** : 该方法不需要考虑原图大小，因为图片本身带有坐标

![图片描述](.\image\重叠.png)

- 无重叠的方式

​	无坐标：略

​	有坐标：略

![图片描述](.\image\预测结果.jpg)

总结：摸索摸索应该都能整明白，如果实在不懂欢迎找我讨论
邮箱：caihongchao0220@163.com

