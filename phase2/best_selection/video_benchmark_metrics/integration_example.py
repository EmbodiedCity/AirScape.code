#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Example for AirScape.code Video Benchmark Metrics

This script demonstrates how to integrate the video_benchmark_metrics module
into the AirScape.code pipeline, particularly for the discriminator model workflow.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from parallel_vbench_evaluator import ParallelVBenchEvaluator
from single_process_evaluator import SingleProcessEvaluator


class AirScapeVideoEvaluator:
    """
    Integration wrapper for AirScape.code video evaluation pipeline.
    
    This class provides a simplified interface for evaluating videos
    in the AirScape.code project workflow.
    """
    
    def __init__(self, vbench_config_path: Optional[str] = None):
        """
        Initialize the AirScape video evaluator.
        
        Args:
            vbench_config_path (Optional[str]): Path to VBench configuration file
        """
        self.vbench_config_path = vbench_config_path or "vbench/VBench_full_info.json"
        
        # Verify VBench configuration exists
        if not os.path.exists(self.vbench_config_path):
            print(f"⚠️  VBench config not found: {self.vbench_config_path}")
            print("Please ensure VBench is properly installed and configured")

    def evaluate_video_directory(self, video_dir: str, output_dir: str, 
                                use_parallel: bool = True) -> Optional[str]:
        """
        Evaluate all videos in a directory using VBench metrics.
        
        Args:
            video_dir (str): Directory containing video files
            output_dir (str): Directory for output results
            use_parallel (bool): Whether to use parallel evaluation
            
        Returns:
            Optional[str]: Path to results CSV file or None if failed
        """
        if not os.path.exists(video_dir):
            print(f"❌ Video directory not found: {video_dir}")
            return None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        if use_parallel:
            return self._run_parallel_evaluation(video_dir, output_dir)
        else:
            return self._run_single_evaluation(video_dir, output_dir)

    def _run_parallel_evaluation(self, video_dir: str, output_dir: str) -> Optional[str]:
        """Run parallel evaluation across multiple GPUs."""
        try:
            evaluator = ParallelVBenchEvaluator(video_dir, output_dir)
            summary = evaluator.run_parallel_evaluation()
            
            if 'final_results' in summary and summary['final_results']['output_file']:
                results_file = summary['final_results']['output_file']
                print(f"✅ Parallel evaluation completed: {results_file}")
                return results_file
            else:
                print("❌ Parallel evaluation failed")
                return None
                
        except Exception as e:
            print(f"❌ Error in parallel evaluation: {e}")
            return None

    def _run_single_evaluation(self, video_dir: str, output_dir: str) -> Optional[str]:
        """Run single-process evaluation."""
        try:
            # Scan video files
            video_files = []
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                video_files.extend(Path(video_dir).glob(f"*{ext}"))
            
            if not video_files:
                print(f"❌ No video files found in {video_dir}")
                return None
            
            # Configure single process
            config = {
                'process_id': 0,
                'gpu_id': 0,
                'video_files': [str(f) for f in video_files],
                'output_file': os.path.join(output_dir, 'vbench_results.csv'),
                'progress_file': os.path.join(output_dir, 'progress.json'),
                'log_file': os.path.join(output_dir, 'evaluation.log')
            }
            
            # Run evaluation
            evaluator = SingleProcessEvaluator(config)
            evaluator.run_evaluation()
            evaluator.cleanup()
            
            if os.path.exists(config['output_file']):
                print(f"✅ Single evaluation completed: {config['output_file']}")
                return config['output_file']
            else:
                print("❌ Single evaluation failed")
                return None
                
        except Exception as e:
            print(f"❌ Error in single evaluation: {e}")
            return None

    def prepare_discriminator_data(self, vbench_csv: str, traj_csv: str, 
                                 output_csv: str) -> bool:
        """
        Prepare combined data for discriminator model training.
        
        This method combines VBench metrics with trajectory similarity scores
        to create the input data for the discriminator model.
        
        Args:
            vbench_csv (str): Path to VBench evaluation results CSV
            traj_csv (str): Path to trajectory similarity CSV
            output_csv (str): Path for combined output CSV
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load VBench results
            if not os.path.exists(vbench_csv):
                print(f"❌ VBench CSV not found: {vbench_csv}")
                return False
            
            vbench_df = pd.read_csv(vbench_csv)
            print(f"📊 Loaded VBench results: {len(vbench_df)} videos")
            
            # Load trajectory similarity results
            traj_dict = {}
            if os.path.exists(traj_csv):
                traj_df = pd.read_csv(traj_csv)
                traj_dict = dict(zip(traj_df.iloc[:, 0], traj_df.iloc[:, 1]))
                print(f"📊 Loaded trajectory results: {len(traj_dict)} videos")
            else:
                print(f"⚠️  Trajectory CSV not found: {traj_csv}, using default values")
            
            # Merge data
            vbench_df['similarity'] = vbench_df['video_name'].map(traj_dict).fillna(0.0)
            
            # Select required columns for discriminator
            required_columns = [
                'video_name',
                'imaging_quality', 
                'motion_smoothness', 
                'dynamic_degree', 
                'aesthetic_quality',
                'similarity'
            ]
            
            # Check if normalized columns exist and use them instead
            normalized_columns = []
            for col in required_columns[1:-1]:  # Skip video_name and similarity
                norm_col = f"{col}_normalized"
                if norm_col in vbench_df.columns:
                    normalized_columns.append(norm_col)
                else:
                    normalized_columns.append(col)
            
            final_columns = ['video_name'] + normalized_columns + ['similarity']
            output_df = vbench_df[final_columns].copy()
            
            # Rename normalized columns back to original names for compatibility
            rename_dict = {}
            for i, col in enumerate(normalized_columns):
                if col.endswith('_normalized'):
                    original_name = required_columns[i + 1]
                    rename_dict[col] = original_name
            
            if rename_dict:
                output_df = output_df.rename(columns=rename_dict)
            
            # Save combined data
            output_df.to_csv(output_csv, index=False)
            print(f"✅ Combined data saved: {output_csv}")
            print(f"📊 Final dataset: {len(output_df)} videos with 5 metrics")
            
            return True
            
        except Exception as e:
            print(f"❌ Error preparing discriminator data: {e}")
            return False

    def validate_results(self, results_csv: str) -> Dict:
        """
        Validate evaluation results and provide statistics.
        
        Args:
            results_csv (str): Path to results CSV file
            
        Returns:
            Dict: Validation statistics
        """
        try:
            df = pd.read_csv(results_csv)
            
            stats = {
                'total_videos': len(df),
                'columns': list(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'score_ranges': {}
            }
            
            # Check score ranges for each metric
            metric_columns = ['imaging_quality', 'motion_smoothness', 
                            'dynamic_degree', 'aesthetic_quality']
            
            for col in metric_columns:
                if col in df.columns:
                    stats['score_ranges'][col] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std())
                    }
                
                # Check normalized version
                norm_col = f"{col}_normalized"
                if norm_col in df.columns:
                    stats['score_ranges'][norm_col] = {
                        'min': float(df[norm_col].min()),
                        'max': float(df[norm_col].max()),
                        'mean': float(df[norm_col].mean()),
                        'std': float(df[norm_col].std())
                    }
            
            return stats
            
        except Exception as e:
            print(f"❌ Error validating results: {e}")
            return {}


def main():
    """
    Example usage of the AirScape video evaluator.
    """
    print("🚀 AirScape Video Benchmark Metrics - Integration Example")
    
    # Example configuration
    video_dir = "/path/to/your/videos"
    output_dir = "evaluation_results"
    traj_csv = "/path/to/trajectory_similarity.csv"
    
    # Initialize evaluator
    evaluator = AirScapeVideoEvaluator()
    
    # Run evaluation
    print(f"\n📊 Evaluating videos in: {video_dir}")
    results_csv = evaluator.evaluate_video_directory(
        video_dir=video_dir,
        output_dir=output_dir,
        use_parallel=True  # Set to False for single-process evaluation
    )
    
    if results_csv:
        # Validate results
        print(f"\n🔍 Validating results...")
        stats = evaluator.validate_results(results_csv)
        print(f"✅ Validation complete: {stats['total_videos']} videos processed")
        
        # Prepare data for discriminator model
        print(f"\n🔧 Preparing discriminator data...")
        discriminator_csv = os.path.join(output_dir, "discriminator_input.csv")
        success = evaluator.prepare_discriminator_data(
            vbench_csv=results_csv,
            traj_csv=traj_csv,
            output_csv=discriminator_csv
        )
        
        if success:
            print(f"✅ Ready for discriminator training: {discriminator_csv}")
        else:
            print(f"❌ Failed to prepare discriminator data")
    else:
        print(f"❌ Evaluation failed")


if __name__ == "__main__":
    main()
