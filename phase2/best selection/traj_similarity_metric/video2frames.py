import cv2
import os
import argparse

def extract_frames(video_path, output_folder):
    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建输出文件夹: {output_folder}")

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return

    frame_count = 0
    print(f"开始从视频 '{video_path}' 提取帧到 '{output_folder}'...")

    while True:
        # 逐帧读取视频
        ret, frame = cap.read()

        # 如果没有帧了，或者读取失败，就退出循环
        if not ret:
            break

        # 构造输出图像的文件名（例如：frame_00000.jpg, frame_00001.jpg）
        # 使用 zfill(5) 确保帧号有前导零，方便排序
        frame_filename = os.path.join(output_folder, f"frame_{str(frame_count).zfill(5)}.jpg")

        # 保存帧为JPG图像
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # 释放视频捕获对象
    cap.release()
    print(f"完成！共提取了 {frame_count} 帧。")

if __name__ == "__main__":

    # 视频所在目录
    parser = argparse.ArgumentParser(description="批量提取视频帧")
    parser.add_argument("--video_dir", type=str, required=True, help="包含视频文件的目录")
    parser.add_argument("--image_dir", type=str, required=True, help="输出帧的目录")
    args = parser.parse_args()

    video_names = os.listdir(args.video_dir)
    for video_name in video_names:
        video_path = os.path.join(args.video_dir, video_name)

        if not video_name.lower().endswith((".mp4")):
            continue

        output_dir = os.path.join(args.image_dir, video_name.split('.')[0], "images")
        extract_frames(video_path, output_dir)

