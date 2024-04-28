import cv2
import numpy as np
import time


def readclassname(path):    # 读取标签名
    classNames = []
    classFile = path
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    return classNames

if __name__ == '__main__':
    # 获得类别信息
    calssnamePath = 'data/class.names'
    classNames = readclassname(calssnamePath)
    
    #打印类别信息
    # print(len(classNames))  
    # print(classNames)

    # 定义模型
    configPath = 'data/ssd-mobilenetv3small.pbtxt'
    weightsPath = 'data/tflite_graph.pb'

    # 定义模型
    mobileNet_SSD_Model = cv2.dnn_DetectionModel(weightsPath, configPath)
    # dnn
    mobileNet_SSD_Model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # 使用CPU
    mobileNet_SSD_Model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # 参数处理
    mobileNet_SSD_Model.setInputSize(320, 320)  # 设置输入的数据大小
    mobileNet_SSD_Model.setInputScale(1.0 / 127.5)  # 设置数值缩放，归一化
    mobileNet_SSD_Model.setInputMean([127.5, 127.5, 127.5])  # 设置各个通道的平均值
    mobileNet_SSD_Model.setInputSwapRB(True)  # BGR转RGB

    # 图像检测
    # img = cv2.imread('test1.jpg')
    # cv2.imshow('test1', img)
    # cv2.waitKey(0)

    # 视频检测
    cap = cv2.VideoCapture('sample/normal.mp4')
    # 摄像头检测
    # cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
    yy1 = 2.5 / 10 # 监测区域左上点坐标比例
    xx1 = 4.2 / 10 
    yy2 = 5.1 / 10 # 监测区域右下点坐标比例
    xx2 = 5.4 / 10 
    #1，2分别是检测区域左上点和右下点的坐标
    y1 = int(height * yy1)
    x1 = int(width * xx1)
    y2 = int(height * yy2)
    x2 = int(width * xx2)  

    while(cap.isOpened()):
        #开始计时
        start = time.time()
        # 获取每一帧图像
        ret, img = cap.read()
        # 获取检测到的类别、conf、置信度
        classIds, confs, bbox = mobileNet_SSD_Model.detect(img, confThreshold=0.5, nmsThreshold=0.5)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        # 在检测区域内则画框
        if len(classIds) != 0 and bbox[0][0] >= x1 and bbox[0][1] >= y1 and bbox[0][0] + bbox[0][2] <= x2 and bbox[0][1] + bbox[0][3] <= y2:
            for classID, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # 画出矩形检测框
                cv2.rectangle(img, box, color=(255, 0, 0), thickness=2)
            # 画出类名和置信度
                cv2.putText(img, text="{}:{}%".format(classNames[classID - 1].upper(),
                                                      "%.2f" % (conf * 100)),
                            org=(box[0], box[1]-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=0.6, color=(0, 0, 255), thickness=2)
                
        #结束计时
        finish = time.time()
        print(f"每帧检测时间：{finish - start}")

        #生成窗口并展示检测结果
        cv2.namedWindow("Output",cv2.WINDOW_GUI_NORMAL) 
        cv2.imshow('Output', img)
        key = cv2.waitKey(1)
        
        #点击叉号或按“q”退出
        if cv2.getWindowProperty('Output',1) > 0:
            break
        if key & 0xFF == ord('q'):
            break    

    # 源视频帧率与画质
    fps=cap.get(cv2.CAP_PROP_FPS)
    size=(cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("fps: {}\nsize: {}".format(fps,size))

    






