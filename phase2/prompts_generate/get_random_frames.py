import os
import random
import cv2

# ================== 配置部分 ==================
VIDEO_DIR = ""  # 视频文件夹路径
OUTPUT_FRAME_DIR = ""  # 输出帧图片文件夹路径
PREVIOUS_FRAME_DIRS = []  # 之前已采样帧的文件夹列表
NUM_FRAMES = 1000  # 本次采样帧数量
MAX_ATTEMPTS = 5   # 每个视频采样最大尝试次数

os.makedirs(OUTPUT_FRAME_DIR, exist_ok=True)

# ================== 主体部分 ==================

def load_processed_video_names(previous_dirs):
    processed = set()
    for p_dir in previous_dirs:
        if not os.path.exists(p_dir):
            print(f"警告: 上一个帧目录 '{p_dir}' 不存在，跳过。")
            continue
        for fname in os.listdir(p_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                name = os.path.splitext(fname)[0]
                idx = name.find("_from_")
                if idx != -1:
                    processed.add(name[idx + len("_from_"):])
    print(f"已加载 {len(processed)} 个已处理视频名称。")
    return processed

def get_available_video_files(video_dir, processed_names):
    all_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]
    available = [f for f in all_files if os.path.splitext(f)[0] not in processed_names]
    return available

def extract_random_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        raise ValueError(f"视频 {video_path} 没有可读取的帧")
    frame_idx = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"无法读取视频 {video_path} 的帧 {frame_idx}")
    return frame, frame_idx

def sample_frames_from_videos():
    processed_names = load_processed_video_names(PREVIOUS_FRAME_DIRS)
    available_videos = get_available_video_files(VIDEO_DIR, processed_names)

    if len(available_videos) < NUM_FRAMES:
        print(f"警告: 可用视频数量 ({len(available_videos)}) 小于目标采样数量 ({NUM_FRAMES})，将全部采样。")
        selected_videos = available_videos
    else:
        selected_videos = random.sample(available_videos, NUM_FRAMES)

    output_paths = []
    failed_count = 0

    for video_file in selected_videos:
        video_path = os.path.join(VIDEO_DIR, video_file)
        for attempt in range(MAX_ATTEMPTS):
            try:
                frame, frame_idx = extract_random_frame(video_path)
                video_name = os.path.splitext(video_file)[0]
                out_name = f"frame{frame_idx}_from_{video_name}.png"
                out_path = os.path.join(OUTPUT_FRAME_DIR, out_name)
                cv2.imwrite(out_path, frame)
                output_paths.append(out_path)
                print(f"成功采样: {out_path}")
                processed_names.add(video_name)
                break
            except Exception as e:
                if attempt == MAX_ATTEMPTS - 1:
                    print(f"视频 {video_file} 采样失败 (尝试 {attempt+1}/{MAX_ATTEMPTS}): {e}")
                    failed_count += 1

    print(f"采样完成: 成功 {len(output_paths)}/{len(selected_videos)}, 失败 {failed_count}")
    print(f"输出帧已保存至: {OUTPUT_FRAME_DIR}")
    return output_paths

def main():
    sample_frames_from_videos()
    print(f"\n'{OUTPUT_FRAME_DIR}' 目录内容:") # 可选：输出目录内容
    if os.path.exists(OUTPUT_FRAME_DIR):
        for f in os.listdir(OUTPUT_FRAME_DIR):
            print(f)
    else:
        print("输出目录未创建或已删除。")

if __name__ == "__main__":
    main()