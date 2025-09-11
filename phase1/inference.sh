#!/bin/bash

# default
OUTPUT_FILE="output.mp4"
LOAD_TYPE="cpu_model_offload"
DTYPE="bfloat16"
NUM_INFERENCE_STEPS=50
ROLLOUT=false

# Help function
show_help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -p, --prompt TEXT          Required: Video description prompt"
  echo "  -m, --model PATH           Required: Pre-trained model path"
  echo "  -o, --output FILE          Output video file path (default: output.mp4)"
  echo "  -i, --image FILE           Input image file (for I2V)"
  echo "  -t, --transformer PATH     Transformer model path"
  echo "  -l, --lora PATH            LoRA weights path"
  echo "  -d, --dtype TYPE           Computation precision: bfloat16 or float16 (default: bfloat16)"
  echo "  --load-type TYPE           Model loading type: cuda, cpu_model_offload, sequential_cpu_offload (default: cpu_model_offload)"
  echo "  --height NUMBER            Output video height"
  echo "  --width NUMBER             Output video width"
  echo "  --steps NUMBER             Inference steps (default: 50)"
  echo "  --seed NUMBER              Random seed"
  echo "  --rollout                  Enable rollout long-range generation mode"
  echo "  -h, --help                 Show this help information"
  exit 1
}

# Parameter parsing
while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--prompt)
      PROMPT="$2"
      shift 2
      ;;
    -m|--model)
      MODEL_PATH="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    -i|--image)
      IMAGE_FILE="$2"
      shift 2
      ;;
    -t|--transformer)
      TRANSFORMER_PATH="$2"
      shift 2
      ;;
    -l|--lora)
      LORA_PATH="$2"
      shift 2
      ;;
    -d|--dtype)
      DTYPE="$2"
      shift 2
      ;;
    --load-type)
      LOAD_TYPE="$2"
      shift 2
      ;;
    --height)
      HEIGHT="$2"
      shift 2
      ;;
    --width)
      WIDTH="$2"
      shift 2
      ;;
    --steps)
      NUM_INFERENCE_STEPS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --rollout)
      ROLLOUT=true
      shift
      ;;
    -h|--help)
      show_help
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      ;;
  esac
done

# Check required parameters
if [ -z "$PROMPT" ] || [ -z "$MODEL_PATH" ]; then
  echo "Error: Must provide prompt (-p) and model path (-m)"
  show_help
fi

# Build command
CMD="cogkit inference \"$PROMPT\" \"$MODEL_PATH\" --output_file \"$OUTPUT_FILE\" --dtype $DTYPE --load_type $LOAD_TYPE --num_inference_steps $NUM_INFERENCE_STEPS"

# Add optional parameters
if [ ! -z "$IMAGE_FILE" ]; then
  CMD="$CMD --image_file \"$IMAGE_FILE\""
fi

if [ ! -z "$TRANSFORMER_PATH" ]; then
  CMD="$CMD --transformer_path \"$TRANSFORMER_PATH\""
fi

if [ ! -z "$LORA_PATH" ]; then
  CMD="$CMD --lora_model_id_or_path \"$LORA_PATH\""
fi

if [ ! -z "$HEIGHT" ]; then
  CMD="$CMD --height $HEIGHT"
fi

if [ ! -z "$WIDTH" ]; then
  CMD="$CMD --width $WIDTH"
fi

if [ ! -z "$SEED" ]; then
  CMD="$CMD --seed $SEED"
fi

if [ "$ROLLOUT" = true ]; then
  CMD="$CMD --rollout"
fi

# Output and execute command
# echo "Executing command: $CMD"
# echo "------------------------------"
eval $CMD

# Check execution result
if [ $? -eq 0 ]; then
  echo "------------------------------"
  echo "Inference successful!"
  echo "Generated video saved to: $(realpath "$OUTPUT_FILE")"
else
  echo "------------------------------"
  echo "Inference failed, please check error messages."
fi