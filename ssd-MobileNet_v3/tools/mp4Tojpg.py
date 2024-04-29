import cv2
import os
 
def extract_frames(video_path, output_path, frame_interval):
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
 
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return
 
    # 获取视频的帧率和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
    # 计算每隔多少帧进行抽取
    frames_to_skip = int(fps * frame_interval)
 
    # 逐帧抽取并保存图像
    frame_count = 0
    save_count = 0
 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
 
        if frame_count % frames_to_skip == 0:
            save_path = os.path.join(output_path, f"frame_{save_count}.jpg")
            cv2.imwrite(save_path, frame)
            save_count += 1
 
        frame_count += 1
 
    cap.release()
    print(f"抽取了 {save_count} 帧图像并保存到 {output_path} 文件夹中")
 
# 测试代码
video_path = "C:/Users/ASUS/Desktop/intern/glovebox/正常片段.mp4"  # 输入的视频文件路径
output_path = "C:/Users/ASUS/Desktop/intern/glovebox/start/test_data/normal"  # 图像输出路径
frame_interval = 1  # 每隔多少帧抽取一张图像
 
extract_frames(video_path, output_path, frame_interval)