#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Model Configuration
MODEL_ARGS=(
    --model_path "your_path/CogVideoX-5b-I2V"
    --model_name "cogvideox-i2v"  # ["cogvideox-i2v"]
    --model_type "i2v"
    --training_type "sft"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "your_path"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --data_root "your_path"
    --caption_column "prompts.txt"
    --video_column "videos.txt"
    # --image_column "images.txt"  # comment this line will use first frame of video as image conditioning
    --train_resolution "49x480x720"  # (frames x height x width), frames should be 8N+1 and height, width should be multiples of 16
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 5 # number of training epochs
    --seed 42 # random seed

    #########   Please keep consistent with deepspeed config file ##########
    --batch_size 2
    --gradient_accumulation_steps 8
    --mixed_precision "bf16"  # ["no", "fp16"] Only CogVideoX-2B supports fp16 training
    ########################################################################
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 25 # save checkpoint every x steps
    --checkpointing_limit 10 # maximum number of checkpoints to keep, after which the oldest one is deleted
    --resume_from_checkpoint "your_path"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation false  # ["true", "false"]
    --validation_dir "/absolute/path/to/validation_set"
    --validation_steps 20  # should be multiple of checkpointing_steps
    --validation_prompts "prompts.txt"
    --validation_images "images.txt"
    --gen_fps 16
)

# Combine all arguments and launch training
accelerate launch --num_processes 8 --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"
