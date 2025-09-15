#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VBench Single Process Evaluator for AirScape.code

This module provides a single-process video evaluation system using VBench metrics.
It evaluates videos on four key dimensions: imaging quality, motion smoothness, 
dynamic degree, and aesthetic quality.
"""

import os
import sys
import json
import time
import tempfile
import shutil
import traceback
from typing import List, Dict, Optional
import pandas as pd
import torch
from tqdm import tqdm

# Prevent tokenizer parallelism conflicts in multi-process environments
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Import VBench modules
try:
    from vbench import VBench
    VBENCH_AVAILABLE = True
except ImportError as e:
    print(f"❌ VBench module import failed: {e}")
    print("Please ensure VBench is properly installed and accessible.")
    VBENCH_AVAILABLE = False


class SingleProcessEvaluator:
    """
    Single-process VBench evaluator for video quality assessment.
    
    This class handles the evaluation of videos using VBench metrics in a 
    single-process environment, suitable for integration into larger systems.
    """
    
    # VBench evaluation dimensions
    SUPPORTED_DIMENSIONS = [
        'imaging_quality',      # Technical image quality (MUSIQ-based)
        'motion_smoothness',    # Motion continuity assessment
        'dynamic_degree',       # Dynamic content analysis
        'aesthetic_quality'     # Aesthetic appeal evaluation
    ]
    
    # Default weights for composite scoring
    DEFAULT_WEIGHTS = {
        'imaging_quality': 0.25,
        'motion_smoothness': 0.25,
        'dynamic_degree': 0.25,
        'aesthetic_quality': 0.25
    }

    def __init__(self, process_config: Dict):
        """
        Initialize the single-process evaluator.
        
        Args:
            process_config (Dict): Configuration dictionary containing:
                - process_id: Process identifier
                - gpu_id: GPU device ID to use
                - video_files: List of video file paths to evaluate
                - output_file: Path for results CSV file
                - progress_file: Path for progress tracking file
                - log_file: Path for log file
                - dimensions: Optional list of dimensions to evaluate
                - weights: Optional custom weights for composite scoring
        """
        self.config = process_config
        self.process_id = process_config['process_id']
        self.gpu_id = process_config['gpu_id']
        self.video_files = process_config['video_files']
        self.output_file = process_config['output_file']
        self.progress_file = process_config['progress_file']
        self.log_file = process_config['log_file']
        
        # Use custom dimensions or default ones
        self.dimensions = process_config.get('dimensions', self.SUPPORTED_DIMENSIONS)
        self.weights = process_config.get('weights', self.DEFAULT_WEIGHTS)
        
        # Setup compute device
        self.device = self._setup_device()
        
        # VBench configuration
        self.output_dir = f"vbench_process_{self.process_id}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        self.log(f"🚀 Process {self.process_id} initialized")
        self.log(f"📱 Using device: {self.device}")
        self.log(f"📊 Videos to evaluate: {len(self.video_files)}")
        self.log(f"🎯 Evaluation dimensions: {', '.join(self.dimensions)}")

    def _setup_device(self) -> torch.device:
        """
        Setup compute device for evaluation.
        
        Returns:
            torch.device: Configured device (CUDA or CPU)
        """
        if torch.cuda.is_available():
            # Set CUDA device
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
            device = torch.device('cuda:0')  # Use cuda:0 in single-GPU environment
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            return device
        else:
            self.log("⚠️  CUDA not available, using CPU")
            return torch.device('cpu')

    def _setup_logging(self):
        """Setup logging system."""
        self.log_handle = open(self.log_file, 'w', encoding='utf-8')

    def log(self, message: str):
        """
        Log a message with timestamp.
        
        Args:
            message (str): Message to log
        """
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [P{self.process_id}] {message}"
        print(log_message)
        
        if hasattr(self, 'log_handle'):
            self.log_handle.write(log_message + '\n')
            self.log_handle.flush()

    def normalize_score(self, score, dimension: str) -> float:
        """
        Normalize score to 0-1 range according to VBench specifications.
        
        This method implements the normalization procedures as specified in the
        VBench paper for each evaluation dimension.
        
        Args:
            score: Raw score from VBench evaluation
            dimension (str): Evaluation dimension name
            
        Returns:
            float: Normalized score in [0, 1] range
        """
        if isinstance(score, bool):
            return 1.0 if score else 0.0
        elif score is None:
            return 0.0
        elif not isinstance(score, (int, float)):
            return 0.0

        # Apply dimension-specific normalization according to VBench paper
        if dimension == 'motion_smoothness':
            # Motion Smoothness: Already normalized by VBench module
            return max(0.0, min(1.0, score))

        elif dimension == 'aesthetic_quality':
            # Aesthetic Quality: Linear normalization from 0-10 to 0-1
            if score > 1.0:
                # Raw 0-10 LAION score, needs normalization
                return max(0.0, min(1.0, score / 10.0))
            else:
                # Already normalized 0-1 score
                return max(0.0, min(1.0, score))

        elif dimension == 'imaging_quality':
            # Imaging Quality: Linear normalization from 0-100 to 0-1
            # MUSIQ score range 0-100, normalize to 0-1
            return max(0.0, min(1.0, score / 100.0))

        elif dimension == 'dynamic_degree':
            # Dynamic Degree: Already in 0-1 range
            return max(0.0, min(1.0, score))

        else:
            # Unknown dimension, return 0
            self.log(f"⚠️  Unknown dimension for normalization: {dimension}")
            return 0.0

    def evaluate_single_video(self, video_path: str) -> Optional[Dict]:
        """
        Evaluate a single video using VBench metrics.

        Args:
            video_path (str): Path to the video file

        Returns:
            Optional[Dict]: Evaluation results dictionary or None if failed
        """
        if not os.path.exists(video_path):
            self.log(f"❌ Video file not found: {video_path}")
            return None

        video_name = os.path.basename(video_path).replace('.mp4', '')

        # Create temporary directory for VBench processing
        temp_dir = tempfile.mkdtemp()
        try:
            # Copy video to temporary directory
            temp_video_path = os.path.join(temp_dir, f"{video_name}.mp4")
            shutil.copy2(video_path, temp_video_path)

            # Initialize VBench
            if not VBENCH_AVAILABLE:
                self.log(f"❌ VBench not available, skipping {video_name}")
                return None

            vbench = VBench(
                device=self.device,
                full_info_dir="vbench/VBench_full_info.json",
                output_path=self.output_dir
            )

            # Run evaluation
            results = vbench.evaluate(
                videos_path=temp_dir,
                name=f"eval_{video_name}",
                dimension_list=self.dimensions,
                mode='custom_input',
                local=True
            )

            if results is None:
                # Try to read results from file
                result_file = os.path.join(self.output_dir, f"eval_{video_name}_eval_results.json")
                if os.path.exists(result_file):
                    with open(result_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                else:
                    self.log(f"❌ Unable to get evaluation results for {video_name}")
                    return None

            # Parse results
            video_result = {'video_name': video_name}

            for dimension in self.dimensions:
                if results and dimension in results:
                    dimension_results = results[dimension][1]  # [avg_score, video_results]
                    if dimension_results and len(dimension_results) > 0:
                        score = dimension_results[0]['video_results']

                        # Handle non-numeric scores
                        if isinstance(score, bool):
                            score = 1.0 if score else 0.0
                        elif score is None:
                            score = 0.0
                        elif not isinstance(score, (int, float)):
                            score = 0.0

                        # Store raw and normalized scores
                        video_result[dimension] = score
                        video_result[f'{dimension}_normalized'] = self.normalize_score(score, dimension)
                    else:
                        video_result[dimension] = 0.0
                        video_result[f'{dimension}_normalized'] = 0.0
                else:
                    video_result[dimension] = 0.0
                    video_result[f'{dimension}_normalized'] = 0.0

            # Calculate composite score
            total_score = sum(
                video_result[f'{dim}_normalized'] * self.weights[dim]
                for dim in self.dimensions if dim in self.weights
            )
            video_result['vbench_total_score'] = total_score

            return video_result

        except Exception as e:
            self.log(f"❌ Error evaluating {video_name}: {e}")
            self.log(f"Detailed error: {traceback.format_exc()}")
            return None
        finally:
            # Cleanup temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_progress(self) -> Dict:
        """
        Load progress information from file.

        Returns:
            Dict: Progress information
        """
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.log(f"⚠️  Failed to load progress file: {e}")

        return {'completed_videos': [], 'last_index': 0}

    def save_progress(self, progress: Dict):
        """
        Save progress information to file.

        Args:
            progress (Dict): Progress information to save
        """
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"⚠️  Failed to save progress: {e}")

    def run_evaluation(self):
        """
        Run the complete evaluation process for all assigned videos.

        This method handles progress tracking, resume capability, and
        periodic result saving.
        """
        self.log(f"🚀 Starting evaluation task")

        # Load progress
        progress = self.load_progress()
        completed_videos = set(progress.get('completed_videos', []))
        start_index = progress.get('last_index', 0)

        results = []

        # Load existing results if available
        if os.path.exists(self.output_file):
            try:
                existing_df = pd.read_csv(self.output_file)
                results = existing_df.to_dict('records')
                self.log(f"📂 Loaded existing results: {len(results)} entries")
            except Exception as e:
                self.log(f"⚠️  Failed to load existing results: {e}")

        # Filter remaining videos
        remaining_videos = [v for v in self.video_files
                          if os.path.basename(v) not in completed_videos]

        if not remaining_videos:
            self.log(f"✅ All videos already evaluated")
            return

        self.log(f"📊 Remaining videos to evaluate: {len(remaining_videos)}")

        # Process videos with progress bar
        with tqdm(remaining_videos, desc=f"P{self.process_id}") as pbar:
            for i, video_path in enumerate(pbar):
                try:
                    result = self.evaluate_single_video(video_path)

                    if result:
                        results.append(result)
                        completed_videos.add(os.path.basename(video_path))

                        # Periodic save (every 10 videos)
                        if len(results) % 10 == 0:
                            self._save_intermediate_results(results)

                            # Update progress
                            progress['completed_videos'] = list(completed_videos)
                            progress['last_index'] = start_index + i + 1
                            self.save_progress(progress)

                    pbar.set_postfix({
                        'completed': len(results),
                        'gpu': self.gpu_id
                    })

                except Exception as e:
                    self.log(f"❌ Error processing video: {e}")
                    continue

        # Save final results
        self._save_final_results(results)
        self.log(f"🎉 Evaluation completed, processed {len(results)} videos")

    def _save_intermediate_results(self, results: List[Dict]):
        """
        Save intermediate results to CSV file.

        Args:
            results (List[Dict]): List of evaluation results
        """
        if results:
            df = pd.DataFrame(results)
            df.to_csv(self.output_file, index=False, encoding='utf-8')

    def _save_final_results(self, results: List[Dict]):
        """
        Save final results to CSV file.

        Args:
            results (List[Dict]): List of evaluation results
        """
        if results:
            df = pd.DataFrame(results)
            df.to_csv(self.output_file, index=False, encoding='utf-8')
            self.log(f"💾 Results saved: {self.output_file}")
        else:
            self.log(f"⚠️  No valid results to save")

    def cleanup(self):
        """Clean up resources and temporary files."""
        if hasattr(self, 'log_handle'):
            self.log_handle.close()

        # Clean up temporary directory
        if os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
            except Exception as e:
                self.log(f"⚠️  Failed to clean up temporary directory: {e}")

    @classmethod
    def create_from_config_file(cls, config_file: str) -> 'SingleProcessEvaluator':
        """
        Create evaluator instance from configuration file.

        Args:
            config_file (str): Path to JSON configuration file

        Returns:
            SingleProcessEvaluator: Configured evaluator instance
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, 'r', encoding='utf-8') as f:
            process_config = json.load(f)

        return cls(process_config)


def main():
    """
    Main function for standalone execution.

    Usage: python single_process_evaluator.py <config_file>
    """
    if len(sys.argv) != 2:
        print("Usage: python single_process_evaluator.py <config_file>")
        print("\nConfig file should contain:")
        print("  - process_id: Process identifier")
        print("  - gpu_id: GPU device ID")
        print("  - video_files: List of video file paths")
        print("  - output_file: Results CSV file path")
        print("  - progress_file: Progress tracking file path")
        print("  - log_file: Log file path")
        sys.exit(1)

    config_file = sys.argv[1]

    # Check VBench availability
    if not VBENCH_AVAILABLE:
        print("❌ VBench module not available")
        print("Please ensure VBench is properly installed and accessible")
        sys.exit(1)

    try:
        # Create and run evaluator
        evaluator = SingleProcessEvaluator.create_from_config_file(config_file)
        evaluator.run_evaluation()
    except KeyboardInterrupt:
        print("⚠️  Received interrupt signal, cleaning up...")
        if 'evaluator' in locals():
            evaluator.cleanup()
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print(f"Detailed error: {traceback.format_exc()}")
        if 'evaluator' in locals():
            evaluator.cleanup()
        sys.exit(1)
    finally:
        if 'evaluator' in locals():
            evaluator.cleanup()


if __name__ == "__main__":
    main()
