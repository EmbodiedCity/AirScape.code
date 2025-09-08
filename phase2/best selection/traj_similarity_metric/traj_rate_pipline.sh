#!/bin/bash

# params
VIDEO_DIR="" # src videos
IMAGE_DIR="" # frames from src videos
OUTPUT_DIR=""
PROMPT_CSV_PATH=""
SCORE_CSV_PATH=""

# GPU选择
AVAILABLE_GPUS="" 

declare -a GPU_IDS_ARRAY # 数组存储最终确定的GPU

echo "用户手动指定 GPU：$AVAILABLE_GPUS"

IFS=',' read -ra GPU_IDS_ARRAY <<< "$AVAILABLE_GPUS"
NUM_GPUS=${#GPU_IDS_ARRAY[@]}

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "手动指定的 GPU 列表为空。请提供有效的 GPU ID。"
    exit 1
fi

echo "将使用 $NUM_GPUS 个指定 GPU：${GPU_IDS_ARRAY[@]}"

MAX_CONCURRENT_JOBS=$NUM_GPUS
echo "3D-reconstruction 任务将以最多 $MAX_CONCURRENT_JOBS 个并行任务运行。"


# 一、批量提取视频所有帧

# -video-dir/
#     -video1.mp4
#     -video2.mp4
#     ...

#      |  | 
#      |  | 
#      |  |
#     \    /
#      \  /
#       \/

# -image-dir/
#     -video1/
#         -image/ 
#             -image1.jpg
#             ...
#     -video2/
#         -image/ 
#             -image1.jpg
#             ...
#     ...

for video_path in "$VIDEO_DIR"/*; do
    video_name=$(basename "$video_path" .mp4)
    image_output="$IMAGE_DIR/$video_name/images"

    if [ -d "$image_output" ] && [ "$(ls -A "$image_output")" ]; then
        echo "Skipping frame extraction for $video_name (already done)"
    else
        echo "Extracting frames for $video_name..."
        python video2frames.py --video_dir "$VIDEO_DIR" --image_dir "$IMAGE_DIR"
        break 
    fi
done


#  二、进行vggt并以colmap形式输出
# -image-dir/
#     -video1/
#         -image/ 
#             -image1.jpg
#             ...
#     -video2/
#         -image/ 
#             -image1.jpg
#             ...
#     ...

#      |  | 
#      |  | 
#      |  |
#     \    /
#      \  /
#       \/

# -image-dir/
#     -video1/
#         -image/ 
#             -image1.jpg
#             ...
#         -sparse/ 
#     -video2/
#         -image/ 
#             -image1.jpg
#             ...
#         -sparse/ 
#     ...

# 临时文件存储所有需要进行 VGGT 处理的场景目录路径
SCENE_DIRS_TO_PROCESS=$(mktemp)
SCENE_DIRS_FOR_CONVERSION=$(mktemp)

# 遍历图像根目录下的所有子目录 (每个子目录对应一个视频的图像)
for SCENE_DIR in "$IMAGE_DIR"/*; do
    # 检查是否有效
    if [ -d "$SCENE_DIR" ] && [ -d "$SCENE_DIR/images" ]; then

        # 如果 images.bin 不存在，需做 VGGT
        if [ ! -f "$SCENE_DIR/sparse/images.bin" ]; then
            echo "$SCENE_DIR" >> "$SCENE_DIRS_TO_PROCESS"
        else
            echo "跳过重建：$(basename "$SCENE_DIR") (已完成)"
        fi

        # 如果 images.txt 不存在，需转换
        if [ ! -f "$SCENE_DIR/sparse/images.txt" ]; then
            echo "$SCENE_DIR" >> "$SCENE_DIRS_FOR_CONVERSION"
        else
            echo "跳过模型转换：$(basename "$SCENE_DIR") (已完成)"
        fi
    fi
done

# 执行 VGGT 
if [ -s "$SCENE_DIRS_TO_PROCESS" ]; then
    echo "开始并行处理VGGT任务..."

    GPU_ARRAY_IDX=0
    CURRENT_JOBS_COUNT=0

    while IFS= read -r SCENE_DIR_PATH; do
        SCENE_NAME=$(basename "$SCENE_DIR_PATH")
        TARGET_GPU_ID="${GPU_IDS_ARRAY[$GPU_ARRAY_IDX]}"

        echo "正在为 $SCENE_NAME 在GPU $TARGET_GPU_ID 上运行VGGT..."
        python /vggt/traj_colmap.py --scene_dir "$SCENE_DIR_PATH" --cuda_device "$TARGET_GPU_ID" &
        PID=$!

        CURRENT_JOBS_COUNT=$((CURRENT_JOBS_COUNT + 1))
        GPU_ARRAY_IDX=$(( (GPU_ARRAY_IDX + 1) % NUM_GPUS ))

        if [ "$CURRENT_JOBS_COUNT" -ge "$MAX_CONCURRENT_JOBS" ]; then
            wait -n
            CURRENT_JOBS_COUNT=$((CURRENT_JOBS_COUNT - 1))
            echo "一个VGGT任务完成，当前剩余 $CURRENT_JOBS_COUNT 个并行任务。"
        fi
    done < "$SCENE_DIRS_TO_PROCESS"

    wait
    echo "所有VGGT处理已完成。"
else
    echo "没有新的场景需要进行VGGT处理。"
fi

# 执行模型转换（仅对没有 images.txt 的）
if [ -s "$SCENE_DIRS_FOR_CONVERSION" ]; then
    echo "开始进行 COLMAP 模型转换..."
    while IFS= read -r SCENE_DIR_PATH; do
        SCENE_NAME=$(basename "$SCENE_DIR_PATH")
        echo "[ ] 正在转换 $SCENE_NAME 的 COLMAP 模型..."
        colmap model_converter --input_path "$SCENE_DIR_PATH/sparse" --output_path "$SCENE_DIR_PATH/sparse" --output_type TXT
    done < "$SCENE_DIRS_FOR_CONVERSION"
else
    echo "模型均已转换。"
fi

rm "$SCENE_DIRS_TO_PROCESS"
rm "$SCENE_DIRS_FOR_CONVERSION"


# 三、利用image.txt形成一个原始矩阵A，将矩阵A逐行向上差分得到轨迹矩阵B，利用B进行聚类（每组视频做一次）
if [ -d "$OUTPUT_DIR" ] && compgen -G "$OUTPUT_DIR/*" > /dev/null; then
    echo "跳过聚类"
else
    echo "正在聚类"
    python outliers_isolation.py --image_dir "$IMAGE_DIR" --output_dir "$OUTPUT_DIR"
fi


# 四、聚类后得到的视频轨迹与instruction对比，得到score（采用余弦相似度）
python compare_score.py --video_dir $VIDEO_DIR --image_dir $IMAGE_DIR --outlier_dir $OUTPUT_DIR --prompt_csv $PROMPT_CSV_PATH --output_csv $SCORE_CSV_PATH
# 初始化输出
if [ ! -f "$SCORE_CSV_PATH" ]; then
    echo "video_name,score" > "$SCORE_CSV_PATH"
fi

# 跳过已评视频
scored_videos=$(tail -n +2 "$SCORE_CSV_PATH" | cut -d',' -f1)

# 遍历所有视频评分（跳过已评）
for VIDEO_PATH in "$IMAGE_DIR"/*; do
    if [ -d "$VIDEO_PATH" ]; then
        VIDEO_NAME=$(basename "$VIDEO_PATH")

        if echo "$scored_videos" | grep -Fxq "$VIDEO_NAME"; then
            echo "已评分: $VIDEO_NAME"
            continue
        fi

        echo "评分中： $VIDEO_NAME"
        sleep 0.01
    fi
done

echo "所有评分完成"