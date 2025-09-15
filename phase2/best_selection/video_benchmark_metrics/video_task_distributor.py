#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Task Distributor for AirScape.code

This module handles the distribution of video evaluation tasks across multiple processes,
with support for resume capability and progress tracking.
"""

import os
import json
import hashlib
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd


class VideoTaskDistributor:
    """
    Video task distributor for parallel processing.
    
    This class handles the distribution of video files across multiple evaluation
    processes, providing load balancing and resume capabilities.
    """

    def __init__(self, video_dir: str, output_dir: str = "parallel_results"):
        """
        Initialize video task distributor.
        
        Args:
            video_dir (str): Directory containing video files
            output_dir (str): Directory for output results
        """
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.video_files = self.scan_video_files()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📁 Video directory: {video_dir}")
        print(f"📊 Found video files: {len(self.video_files)}")
        print(f"💾 Output directory: {output_dir}")

    def scan_video_files(self) -> List[str]:
        """
        Scan for video files in the specified directory.
        
        Returns:
            List[str]: List of video file paths
        """
        video_files = []
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        
        for root, dirs, files in os.walk(self.video_dir):
            for file in files:
                if Path(file).suffix.lower() in video_extensions:
                    video_files.append(os.path.join(root, file))
        
        # Sort for consistent ordering
        video_files.sort()
        return video_files

    def distribute_videos(self, num_processes: int) -> List[List[str]]:
        """
        Distribute video files evenly across processes.
        
        Args:
            num_processes (int): Number of processes to distribute across
            
        Returns:
            List[List[str]]: List of video file lists for each process
        """
        if num_processes <= 0:
            raise ValueError("Number of processes must be positive")
        
        if not self.video_files:
            return [[] for _ in range(num_processes)]
        
        # Calculate videos per process
        videos_per_process = len(self.video_files) // num_processes
        remainder = len(self.video_files) % num_processes
        
        distributed_videos = []
        start_idx = 0
        
        for i in range(num_processes):
            # Add one extra video to first 'remainder' processes
            current_count = videos_per_process + (1 if i < remainder else 0)
            end_idx = start_idx + current_count
            
            process_videos = self.video_files[start_idx:end_idx]
            distributed_videos.append(process_videos)
            
            start_idx = end_idx
        
        return distributed_videos

    def generate_process_configs(self, gpu_allocation: List[Dict]) -> List[Dict]:
        """
        Generate process configurations based on GPU allocation.
        
        Args:
            gpu_allocation (List[Dict]): GPU allocation information
            
        Returns:
            List[Dict]: List of process configuration dictionaries
        """
        process_configs = []
        process_id = 0
        
        # Calculate total processes
        total_processes = sum(alloc['processes_count'] for alloc in gpu_allocation)
        
        if total_processes == 0:
            print("❌ No processes to configure")
            return []
        
        # Distribute videos across all processes
        distributed_videos = self.distribute_videos(total_processes)
        
        # Generate configurations for each GPU
        for gpu_alloc in gpu_allocation:
            gpu_id = gpu_alloc['gpu_id']
            processes_count = gpu_alloc['processes_count']
            
            for proc_idx in range(processes_count):
                if process_id < len(distributed_videos):
                    config = {
                        'process_id': process_id,
                        'gpu_id': gpu_id,
                        'video_files': distributed_videos[process_id],
                        'output_file': os.path.join(
                            self.output_dir, 
                            f"process_{process_id}_gpu_{gpu_id}_results.csv"
                        ),
                        'progress_file': os.path.join(
                            self.output_dir, 
                            f"process_{process_id}_progress.json"
                        ),
                        'log_file': os.path.join(
                            self.output_dir, 
                            f"process_{process_id}_gpu_{gpu_id}.log"
                        )
                    }
                    process_configs.append(config)
                    process_id += 1
        
        return process_configs

    def print_distribution_summary(self, process_configs: List[Dict]):
        """
        Print summary of video distribution across processes.
        
        Args:
            process_configs (List[Dict]): Process configuration list
        """
        if not process_configs:
            print("❌ No process configurations to summarize")
            return
        
        total_videos = sum(len(config['video_files']) for config in process_configs)
        
        print(f"\n📦 Video Distribution Summary:")
        print(f"   Total processes: {len(process_configs)}")
        print(f"   Total videos: {total_videos}")
        print(f"   Average videos per process: {total_videos / len(process_configs):.1f}")
        
        # Group by GPU
        gpu_summary = {}
        for config in process_configs:
            gpu_id = config['gpu_id']
            if gpu_id not in gpu_summary:
                gpu_summary[gpu_id] = {'processes': 0, 'videos': 0}
            gpu_summary[gpu_id]['processes'] += 1
            gpu_summary[gpu_id]['videos'] += len(config['video_files'])
        
        print(f"\n🎯 Per-GPU Distribution:")
        for gpu_id, summary in gpu_summary.items():
            print(f"   GPU {gpu_id}: {summary['processes']} processes, "
                  f"{summary['videos']} videos")

    def save_task_distribution(self, process_configs: List[Dict]) -> str:
        """
        Save task distribution configuration to file.
        
        Args:
            process_configs (List[Dict]): Process configuration list
            
        Returns:
            str: Path to saved configuration file
        """
        config_path = os.path.join(self.output_dir, "task_distribution.json")
        
        distribution_config = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'video_dir': self.video_dir,
            'total_videos': len(self.video_files),
            'total_processes': len(process_configs),
            'process_configs': process_configs
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(distribution_config, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Task distribution saved: {config_path}")
        return config_path

    def check_completed_tasks(self, process_configs: List[Dict]) -> Tuple[List[int], List[int]]:
        """
        Check which processes have completed their tasks.
        
        Args:
            process_configs (List[Dict]): Process configuration list
            
        Returns:
            Tuple[List[int], List[int]]: (completed_process_ids, pending_process_ids)
        """
        completed = []
        pending = []
        
        for config in process_configs:
            process_id = config['process_id']
            output_file = config['output_file']
            
            if os.path.exists(output_file):
                try:
                    # Check if results file has content
                    df = pd.read_csv(output_file)
                    expected_videos = len(config['video_files'])
                    
                    if len(df) >= expected_videos:
                        completed.append(process_id)
                    else:
                        pending.append(process_id)
                        
                except Exception as e:
                    print(f"⚠️  Error checking results for process {process_id}: {e}")
                    pending.append(process_id)
            else:
                pending.append(process_id)
        
        return completed, pending

    def merge_results(self, process_configs: List[Dict], 
                     output_filename: str = "merged_results.csv") -> Optional[str]:
        """
        Merge results from all processes into a single file.
        
        Args:
            process_configs (List[Dict]): Process configuration list
            output_filename (str): Name of the merged output file
            
        Returns:
            Optional[str]: Path to merged results file or None if failed
        """
        merged_results = []
        
        for config in process_configs:
            output_file = config['output_file']
            
            if os.path.exists(output_file):
                try:
                    df = pd.read_csv(output_file)
                    if not df.empty:
                        merged_results.append(df)
                        print(f"✅ Loaded {len(df)} results from process {config['process_id']}")
                    else:
                        print(f"⚠️  Empty results file: {output_file}")
                except Exception as e:
                    print(f"❌ Error loading results from {output_file}: {e}")
            else:
                print(f"⚠️  Results file not found: {output_file}")
        
        if not merged_results:
            print("❌ No results to merge")
            return None
        
        # Merge all results
        try:
            final_df = pd.concat(merged_results, ignore_index=True)
            
            # Remove duplicates based on video_name
            initial_count = len(final_df)
            final_df = final_df.drop_duplicates(subset=['video_name'], keep='first')
            final_count = len(final_df)
            
            if initial_count != final_count:
                print(f"⚠️  Removed {initial_count - final_count} duplicate entries")
            
            # Sort by video_name for consistency
            final_df = final_df.sort_values('video_name').reset_index(drop=True)
            
            # Save merged results
            output_path = os.path.join(self.output_dir, output_filename)
            final_df.to_csv(output_path, index=False, encoding='utf-8')
            
            print(f"✅ Merged results saved: {output_path}")
            print(f"📊 Total unique videos: {len(final_df)}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error merging results: {e}")
            return None

    @classmethod
    def load_task_distribution(cls, config_path: str) -> Optional['VideoTaskDistributor']:
        """
        Load task distributor from saved configuration.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            Optional[VideoTaskDistributor]: Loaded distributor or None if failed
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            video_dir = config['video_dir']
            output_dir = os.path.dirname(config_path)
            
            return cls(video_dir, output_dir)
            
        except Exception as e:
            print(f"❌ Failed to load task distribution: {e}")
            return None


def main():
    """
    Main function for standalone testing.
    """
    print("🔍 Video Task Distributor - Standalone Test")
    
    # Test with current directory
    distributor = VideoTaskDistributor(".", "test_output")
    
    if distributor.video_files:
        # Mock GPU allocation
        gpu_allocation = [
            {'gpu_id': 0, 'processes_count': 2},
            {'gpu_id': 1, 'processes_count': 1}
        ]
        
        # Generate configurations
        configs = distributor.generate_process_configs(gpu_allocation)
        distributor.print_distribution_summary(configs)
        
        # Save configuration
        config_path = distributor.save_task_distribution(configs)
        
        # Test loading
        loaded_distributor = VideoTaskDistributor.load_task_distribution(config_path)
        if loaded_distributor:
            print("✅ Configuration loaded successfully")
        
        # Cleanup
        import shutil
        if os.path.exists("test_output"):
            shutil.rmtree("test_output")
    else:
        print("❌ No video files found for testing")


if __name__ == "__main__":
    main()
