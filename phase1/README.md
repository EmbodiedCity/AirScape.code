# Phase-1: Configuration, Training, and Inference

This document provides detailed instructions for configuring the environment, training, and running inference in **Phase-1** of the project.

---

## 1. Requirements

* **Hardware**

  * An NVIDIA GPU with CUDA support is required.
  * Tested on: **8 × A800 40GB GPUs**.
  * **Minimum** requirements:

    * 40GB GPU memory for training
    * 20GB GPU memory for inference

* **Operating System**

  * Tested on **Linux**

---

## 2. Dependencies and Installation

Clone the repository:

```bash
git clone https://github.com/EmbodiedCity/AirScape.code.git
```

Create and activate the environment:

```bash
cd phase1
conda create -n AirScape python=3.11
conda activate AirScape
pip install -r requirements.txt
```

---

## 3. Model Weights

### 3.1 Pretrained CogVideo Weights (Required)

Install Hugging Face CLI if not already available:

```bash
pip install huggingface_hub
```

Login to Hugging Face (only needed once):

```bash
huggingface-cli login
```

Download the **CogVideoX-5b-I2V pretrained weights**:

```bash
huggingface-cli download zai-org/CogVideoX-5b-I2V --local-dir ./checkpoints/CogVideoX-5b-I2V
```

These weights can be used either as initialization for training or directly for inference.

---

### 3.2 Phase1 Checkpoint (Optional)

If you don’t want to train, you can directly download a *phase1 checkpoint* and use it for inference (this corresponds to the `-t` argument in `inference.sh`):

```bash
huggingface-cli download your_org/your phase1-model phase1.ckpt --local-dir ./checkpoints
```

Now you can specify it in inference with:

```bash
-t ./checkpoints phase1.ckpt
```

---

## 4. Training

### Data Preparation

Preprocess training data:

```bash
python process_data.py
```

### Training

Run training:

```bash
bash finetune/train.sh
```

**Key arguments**:

* `--output_dir`: Output directory
* `--data_root`: Root path of training data
* `--train_resolution`: Training resolution
* `--resume_from_checkpoint`: Path to resume training from a checkpoint

---

## 5. Inference

### Weight Conversion

Convert trained weights into diffusion format before inference:

```bash
python zero2diffusers.py <input_checkpoint_path> <output_diffusers_path>
```

### Running Inference

Example:

```bash
bash inference.sh -o 00847_urbanvideo_test.mp4 \
-p "The video is egocentric/first-person perspective, captured from a camera mounted on a drone. The drone rotated to the right slightly while maintaining its altitude, capturing urban buildings and streets, and eventually reached a position overlooking the high-rise area near a major road." \
-i 00847_urbanvideo_test.jpg \
-m checkpoints/CogVideoX-5b-I2V \
-t checkpoints phase1.ckpt
```

### Arguments

* `-p, --prompt TEXT` *(required)*: Text prompt describing the video
* `-m, --model PATH` *(required)*: Path to CogVideo pretrained weights
* `-o, --output FILE`: Output video file (default: `output.mp4`)
* `-i, --image FILE` *(required)*: Input image file
* `-t, --transformer PATH`: Path to phase1 weights (or pretrained weights if skipping training)
* `--steps NUMBER`: Number of inference steps (default: 50)
* `-h, --help`: Show help information

