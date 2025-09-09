# Instruction on trajectory similarity

## Dependencies

First, you can go through the detailed documents to download [COLMAP](https://github.com/colmap/colmap) and [VGGT](https://github.com/facebookresearch/vggt), the 3D reconstruction utils to realize Strunture from Motion(SfM).

Clone VGGT folder under `traj_similarity_metric/`, and then replace the original `demo_colmap.py` with our `traj_similarity_metric/traj_colmap.py`

Second, set up the environment with necessary libraries.
```
conda create -n traj
conda activate traj
cd best_selection\traj_similarity_metric
pip install -r requirements.txt
```

## Usage

Just specify the exact directory or paths in the files and run `bash traj_rate_pipline.sh`