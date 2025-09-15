# Video Benchmark Metrics

A comprehensive video quality evaluation system based on VBench metrics, designed for the AirScape.code project. This module provides both single-process and parallel evaluation capabilities for assessing video quality across four key dimensions.

## 🎯 Overview

This module evaluates videos using four core VBench metrics:

- **Imaging Quality**: Technical image quality assessment using MUSIQ architecture
- **Motion Smoothness**: Motion continuity and fluidity evaluation
- **Dynamic Degree**: Dynamic content and motion richness analysis
- **Aesthetic Quality**: Visual appeal and aesthetic assessment

## 📊 Key Features

- **Multi-dimensional Evaluation**: Comprehensive assessment across 4 VBench metrics
- **Parallel Processing**: Efficient multi-GPU parallel evaluation
- **Intelligent Resource Management**: Automatic GPU allocation and memory management
- **Resume Capability**: Robust checkpoint and resume functionality
- **Standardized Output**: Consistent CSV output with normalized scores
- **Progress Tracking**: Real-time progress monitoring and logging

## 🛠️ Installation

### Prerequisites

1. **Python Environment**: Python 3.8+ with PyTorch
2. **NVIDIA GPUs**: CUDA-compatible GPUs with sufficient memory
3. **VBench Framework**: VBench library and pretrained models

### Dependencies

```bash
# Core dependencies
pip install torch torchvision pandas numpy tqdm
pip install opencv-python pillow

# VBench specific dependencies
pip install clip-by-openai
pip install pyiqa
pip install sentence-transformers
```

### VBench Setup

1. **Install VBench**:
```bash
# Clone and install VBench
git clone https://github.com/Vchitect/VBench.git
cd VBench
pip install -e .
```

2. **Download Pretrained Models**:
```bash
# Download required models (adjust paths as needed)
mkdir -p pretrained/{aesthetic_model,clip_model,pyiqa_model,raft_model}

# Follow VBench documentation to download:
# - MUSIQ model for imaging quality
# - CLIP model for aesthetic quality
# - RAFT model for motion analysis
# - Other required models
```

3. **Configure VBench**:
Ensure `vbench/VBench_full_info.json` is properly configured with model paths.

## 📖 Usage

### Single Process Evaluation

For evaluating videos in a single process:

```python
from single_process_evaluator import SingleProcessEvaluator

# Configuration
config = {
    'process_id': 0,
    'gpu_id': 0,
    'video_files': ['/path/to/video1.mp4', '/path/to/video2.mp4'],
    'output_file': 'results.csv',
    'progress_file': 'progress.json',
    'log_file': 'evaluation.log'
}

# Create and run evaluator
evaluator = SingleProcessEvaluator(config)
evaluator.run_evaluation()
evaluator.cleanup()
```

### Parallel Evaluation

For large-scale parallel evaluation across multiple GPUs:

```python
from parallel_vbench_evaluator import ParallelVBenchEvaluator

# Create parallel evaluator
evaluator = ParallelVBenchEvaluator(
    video_dir='/path/to/videos/',
    output_dir='evaluation_results/'
)

# Run parallel evaluation
summary = evaluator.run_parallel_evaluation()

# Print results
evaluator.print_final_summary(summary)
```

### Command Line Usage

**Single Process**:
```bash
python single_process_evaluator.py config.json
```

**Parallel Evaluation**:
```bash
python parallel_vbench_evaluator.py /path/to/videos/ results/
```

## 📋 Configuration

### Single Process Configuration

```json
{
    "process_id": 0,
    "gpu_id": 0,
    "video_files": ["video1.mp4", "video2.mp4"],
    "output_file": "results.csv",
    "progress_file": "progress.json",
    "log_file": "evaluation.log",
    "dimensions": ["imaging_quality", "motion_smoothness", "dynamic_degree", "aesthetic_quality"],
    "weights": {
        "imaging_quality": 0.25,
        "motion_smoothness": 0.25,
        "dynamic_degree": 0.25,
        "aesthetic_quality": 0.25
    }
}
```

### GPU Resource Configuration

The system automatically detects and allocates GPU resources based on:
- Available GPU memory
- Current GPU utilization
- Memory requirements per process
- Safety margins to prevent overflow

## 📊 Evaluation Metrics

### 1. Imaging Quality
- **Technology**: MUSIQ (Multi-Scale Image Quality) architecture
- **Score Range**: 0-100 (raw) → 0-1 (normalized)
- **Purpose**: Evaluates technical image quality including sharpness, noise, compression artifacts
- **Normalization**: Linear scaling by dividing by 100

### 2. Motion Smoothness
- **Technology**: Optical flow analysis for motion continuity
- **Score Range**: 0-1 (already normalized)
- **Purpose**: Assesses smoothness of motion, detects jitter and discontinuities
- **Normalization**: Already in 0-1 range

### 3. Dynamic Degree
- **Technology**: Motion magnitude and variation analysis
- **Score Range**: 0-1 (already normalized)
- **Purpose**: Measures dynamic content richness and motion diversity
- **Normalization**: Already in 0-1 range

### 4. Aesthetic Quality
- **Technology**: LAION aesthetic predictor with CLIP features
- **Score Range**: 0-10 (raw) → 0-1 (normalized)
- **Purpose**: Evaluates visual appeal and aesthetic quality
- **Normalization**: Linear scaling by dividing by 10

### Composite Scoring

The final VBench score is calculated as a weighted average:

```python
vbench_total_score = (
    imaging_quality_normalized * 0.25 +
    motion_smoothness_normalized * 0.25 +
    dynamic_degree_normalized * 0.25 +
    aesthetic_quality_normalized * 0.25
)
```

## 📁 Output Format

### CSV Results Structure

```csv
video_name,imaging_quality,imaging_quality_normalized,motion_smoothness,motion_smoothness_normalized,dynamic_degree,dynamic_degree_normalized,aesthetic_quality,aesthetic_quality_normalized,vbench_total_score
video1,65.2,0.652,0.943,0.943,0.8,0.8,5.4,0.54,0.734
video2,72.1,0.721,0.891,0.891,1.0,1.0,6.2,0.62,0.808
```

### Output Files

- **Results CSV**: Main evaluation results with all metrics
- **Progress JSON**: Checkpoint data for resume capability
- **Log Files**: Detailed execution logs per process
- **Summary JSON**: Comprehensive evaluation statistics

## 🚀 Performance Optimization

### GPU Memory Management

- **Automatic Detection**: Scans available GPU memory and utilization
- **Smart Allocation**: Distributes processes based on memory requirements
- **Safety Margins**: Prevents memory overflow with configurable margins
- **Cache Management**: Automatic GPU cache clearing between evaluations

### Parallel Processing

- **Multi-GPU Support**: Utilizes all available GPUs efficiently
- **Load Balancing**: Even distribution of videos across processes
- **Process Isolation**: Independent processes prevent interference
- **Resource Monitoring**: Real-time resource usage tracking

### Resume Capability

- **Progress Tracking**: Saves progress after every 10 evaluations
- **Automatic Resume**: Detects and skips completed videos
- **Crash Recovery**: Robust recovery from unexpected interruptions
- **Incremental Results**: Periodic saving prevents data loss

## 🔧 Troubleshooting

### Common Issues

**1. VBench Import Error**
```bash
❌ VBench module import failed
```
**Solution**: Ensure VBench is properly installed and in Python path

**2. GPU Memory Error**
```bash
❌ CUDA out of memory
```
**Solution**: Reduce `memory_per_process` or increase `safety_margin`

**3. Model Not Found**
```bash
❌ Pretrained model not found
```
**Solution**: Verify model paths in `VBench_full_info.json`

**4. No Available GPUs**
```bash
❌ No available GPU resources found
```
**Solution**: Check GPU status with `nvidia-smi` and free up memory

## 🔗 Integration with AirScape.code

### Project Structure

```
AirScape.code/phase2/best_selection/video_benchmark_metrics/
├── README.md                      # This documentation
├── single_process_evaluator.py    # Single-process evaluation
├── parallel_vbench_evaluator.py   # Parallel evaluation system
├── gpu_resource_manager.py        # GPU resource management
└── video_task_distributor.py      # Task distribution logic
```

## 📚 API Reference

### SingleProcessEvaluator

```python
class SingleProcessEvaluator:
    def __init__(self, process_config: Dict)
    def evaluate_single_video(self, video_path: str) -> Optional[Dict]
    def run_evaluation(self) -> None
    def normalize_score(self, score, dimension: str) -> float
    def cleanup(self) -> None
```

### ParallelVBenchEvaluator

```python
class ParallelVBenchEvaluator:
    def __init__(self, video_dir: str, output_dir: str)
    def run_parallel_evaluation(self, max_workers: Optional[int]) -> Dict
    def setup_parallel_evaluation(self) -> Tuple[List[Dict], str, str]
    def print_final_summary(self, summary: Dict) -> None
```

### GPUResourceManager

```python
class GPUResourceManager:
    def __init__(self, memory_per_process: float, safety_margin: float)
    def get_gpu_status(self) -> List[Dict]
    def generate_process_allocation(self) -> Tuple[List[Dict], int]
    def print_allocation_summary(self, gpu_allocation: List[Dict], total_processes: int)
```

### VideoTaskDistributor

```python
class VideoTaskDistributor:
    def __init__(self, video_dir: str, output_dir: str)
    def scan_video_files(self) -> List[str]
    def generate_process_configs(self, gpu_allocation: List[Dict]) -> List[Dict]
    def merge_results(self, process_configs: List[Dict], output_filename: str) -> Optional[str]
```

## 🙏 Acknowledgments

- **VBench Team**: For the comprehensive video evaluation framework
- **Open Source Community**: For the underlying libraries and tools

## 📞 Support

For issues, questions, or contributions:

1. **GitHub Issues**: Report bugs and feature requests
2. **Documentation**: Refer to VBench official documentation

---

**Note**: This module requires proper VBench installation and pretrained models. Ensure all dependencies are correctly configured before use.