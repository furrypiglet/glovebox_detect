# 手套箱危险检测

## 前言

基于SSD-MobileNet_v3-small预训练模型并使用自制的监控视频图像帧数据集完成训练，可部署于Linux和Windows平台，包括桌面电脑与嵌入式设备（树莓派4B等）。训练环境依赖简单，利用DNN进行加速，能够完成对监控画面指定区域的实时监测并报告异常。

## 检测结果与对比

| 操作系统              | 硬件                    | 检测每帧耗时 |
| --------------------- | ----------------------- | ------------ |
| Windows11             | CPU：AMD  R7 4800HS     | 0.018s       |
| Linux（Raspberry os） | 树莓派4B（Cortex A72 ） | 0.1s         |

## 训练环境与依赖

Windows11

Python

[TensorFlow-Object-Detection-API](https://github.com/tensorflow/models)

```
git clone https://github.com/tensorflow/models.git
```

[CUDA](https://developer.nvidia.com/cuda-downloads)

[cuDNN](https://developer.nvidia.com/cudnn-downloads)

## 检测环境与依赖

Python

OpenCV（Version>=3.3）

OpenCV-Contrib（与OpenCV版本相同）

```
pip3 install opencv-contrib-python
```

## 模型训练

详细过程附文档：[视觉检测模型训练教程](train_tutorials.md)

## 检测

设置`glovebox_detect/ssd-MobileNet_v3/detect.py`文件参数

​	1.设置检测区域

​	使用画图工具查询检测区域角点坐标像素，除以图像总像素数，得到相对坐标比例

```
 yy1 = 2.5 / 10 # 监测区域左上点坐标比例

 xx1 = 4.2 / 10 

 yy2 = 5.1 / 10 # 监测区域右下点坐标比例

 xx2 = 5.4 / 10 
```

​	2.设置检测内容

​	可检测图像文件、视频文件、摄像头

```
# 视频检测
cap = cv2.VideoCapture('normal.mp4')

# 图像检测
# img = cv2.imread('test1.jpg')
# cv2.imshow('test1', img)
# cv2.waitKey(0)
 
# 摄像头检测
# cap = cv2.VideoCapture(0)
```

​	3.开始检测

​	进入文件地址

```
cd glovebox_detect/ssd-MobileNet_v3
```

​	运行`detect.py`文件

```
python detecct.py
```

​	4.结束检测

```
#点击叉号或按“q”退出
if cv2.getWindowProperty('Output',1) > 0:
    break
if key & 0xFF == ord('q'):
    break
```



## 参考链接

1.Jetson目标检测SSD-MobileNet应用实战教程：https://blog.csdn.net/weixin_47407066/article/details/126256850

2.Install OpenCV on Raspberry Pi 5：https://qengineering.eu/install%20opencv%20on%20raspberry%20pi%205.html

3.OpenCV Tutorials：https://docs.opencv.org/4.x/d9/df8/tutorial_root.html