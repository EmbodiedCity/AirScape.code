#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VBench Parallel Evaluator for AirScape.code

This module provides a parallel video evaluation system using VBench metrics.
It manages multiple processes for efficient GPU utilization and includes
intelligent resource management and resume capabilities.
"""

import os
import sys
import json
import time
import signal
import subprocess
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from pathlib import Path

# Import resource management modules (these should be available in your environment)
try:
    from .gpu_resource_manager import GPUResourceManager
    from .video_task_distributor import VideoTaskDistributor
except ImportError:
    # Fallback for standalone execution
    try:
        from gpu_resource_manager import GPUResourceManager
        from video_task_distributor import VideoTaskDistributor
    except ImportError:
        print("❌ Required modules not found: gpu_resource_manager, video_task_distributor")
        print("Please ensure these modules are available in your environment")
        sys.exit(1)


class ParallelVBenchEvaluator:
    """
    Parallel VBench evaluator for efficient video quality assessment.
    
    This class manages multiple evaluation processes across available GPUs,
    providing intelligent resource allocation, progress tracking, and 
    resume capabilities for large-scale video evaluation tasks.
    """

    def __init__(self, video_dir: str, output_dir: str = "parallel_vbench_results"):
        """
        Initialize the parallel VBench evaluator.
        
        Args:
            video_dir (str): Directory containing video files to evaluate
            output_dir (str): Directory for output results and logs
        """
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.running_processes = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🚀 VBench Parallel Evaluator initialized")
        print(f"📁 Video directory: {video_dir}")
        print(f"💾 Output directory: {output_dir}")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """
        Handle interrupt signals for graceful shutdown.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        print(f"\n⚠️  Received signal {signum}, cleaning up processes...")
        self._cleanup_processes()
        sys.exit(0)

    def _cleanup_processes(self):
        """Clean up all running processes."""
        for process in self.running_processes:
            try:
                if process.poll() is None:  # Process still running
                    process.terminate()
                    process.wait(timeout=5)
            except Exception as e:
                print(f"⚠️  Error cleaning up process: {e}")

    def setup_parallel_evaluation(self) -> Tuple[List[Dict], str, str]:
        """
        Setup the parallel evaluation environment.
        
        This method analyzes available GPU resources, distributes video tasks
        across processes, and prepares configuration files.
        
        Returns:
            Tuple[List[Dict], str, str]: (process_configs, gpu_config_path, task_config_path)
        """
        print("\n" + "="*60)
        print("🔧 Setting up parallel evaluation environment")
        print("="*60)
        
        # 1. Analyze GPU resources
        print("📊 Analyzing GPU resources...")
        gpu_manager = GPUResourceManager(memory_per_process=2.0, safety_margin=0.8)
        gpu_allocation, total_processes = gpu_manager.generate_process_allocation()
        
        if not gpu_allocation:
            raise RuntimeError("❌ No available GPU resources found")
        
        gpu_manager.print_allocation_summary(gpu_allocation, total_processes)
        gpu_config_path = os.path.join(self.output_dir, "gpu_allocation.json")
        gpu_manager.save_allocation_config(gpu_allocation, gpu_config_path)
        
        # 2. Distribute video tasks
        print("\n📦 Distributing video tasks...")
        task_distributor = VideoTaskDistributor(self.video_dir, self.output_dir)
        process_configs = task_distributor.generate_process_configs(gpu_allocation)
        
        task_distributor.print_distribution_summary(process_configs)
        task_config_path = task_distributor.save_task_distribution(process_configs)
        
        return process_configs, gpu_config_path, task_config_path

    def run_single_process(self, process_config: Dict) -> Dict:
        """
        Execute a single evaluation process.
        
        Args:
            process_config (Dict): Process configuration dictionary
            
        Returns:
            Dict: Process execution results
        """
        process_id = process_config['process_id']
        gpu_id = process_config['gpu_id']
        
        # Save process configuration to temporary file
        config_file = os.path.join(self.output_dir, f"process_{process_id}_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(process_config, f, indent=2, ensure_ascii=False)
        
        try:
            # Setup environment variables
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            env['TOKENIZERS_PARALLELISM'] = 'false'

            # Command to run single process evaluator
            # Note: Adjust the path to single_process_evaluator.py as needed
            cmd = [
                sys.executable, 
                os.path.join(os.path.dirname(__file__), 'single_process_evaluator.py'),
                config_file
            ]
            
            start_time = time.time()
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.running_processes.append(process)
            
            # Wait for process completion
            stdout, stderr = process.communicate()
            end_time = time.time()
            
            # Remove completed process from tracking
            if process in self.running_processes:
                self.running_processes.remove(process)
            
            # Clean up configuration file
            if os.path.exists(config_file):
                os.remove(config_file)
            
            return {
                'process_id': process_id,
                'gpu_id': gpu_id,
                'return_code': process.returncode,
                'execution_time': end_time - start_time,
                'stdout': stdout,
                'stderr': stderr,
                'output_file': process_config['output_file'],
                'video_count': len(process_config['video_files'])
            }
            
        except Exception as e:
            return {
                'process_id': process_id,
                'gpu_id': gpu_id,
                'return_code': -1,
                'execution_time': 0,
                'error': str(e),
                'output_file': process_config['output_file'],
                'video_count': len(process_config['video_files'])
            }

    def check_resume_status(self, process_configs: List[Dict]) -> Tuple[List[int], List[int]]:
        """
        Check which processes have completed and which need to run.

        Args:
            process_configs (List[Dict]): List of process configurations

        Returns:
            Tuple[List[int], List[int]]: (completed_process_ids, pending_process_ids)
        """
        distributor = VideoTaskDistributor(self.video_dir, self.output_dir)
        return distributor.check_completed_tasks(process_configs)

    def run_parallel_evaluation(self, max_workers: Optional[int] = None) -> Dict:
        """
        Execute parallel evaluation across multiple processes.

        Args:
            max_workers (Optional[int]): Maximum number of parallel processes.
                                       None means use all available processes.

        Returns:
            Dict: Comprehensive evaluation results and statistics
        """
        # Setup parallel environment
        process_configs, gpu_config_path, task_config_path = self.setup_parallel_evaluation()

        if not process_configs:
            raise RuntimeError("❌ No executable process configurations found")

        # Check for resume capability
        completed, pending = self.check_resume_status(process_configs)

        if completed:
            print(f"\n✅ Found {len(completed)} completed processes")
            print(f"⏳ Need to execute {len(pending)} processes")

            # Filter to only pending processes
            process_configs = [config for config in process_configs
                             if config['process_id'] in pending]

        if not process_configs:
            print("🎉 All processes completed, finalizing results...")
            return self._finalize_results(process_configs)

        # Limit parallel workers
        if max_workers is None:
            max_workers = len(process_configs)
        else:
            max_workers = min(max_workers, len(process_configs))

        print(f"\n🚀 Starting parallel evaluation")
        print(f"📊 Parallel processes: {max_workers}")
        print(f"⏱️  Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Execute parallel evaluation
        results = []
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_config = {
                executor.submit(self.run_single_process, config): config
                for config in process_configs
            }

            # Collect results
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Print progress
                    if result['return_code'] == 0:
                        print(f"✅ Process {result['process_id']} (GPU {result['gpu_id']}) "
                              f"completed in {result['execution_time']:.1f}s")
                    else:
                        print(f"❌ Process {result['process_id']} (GPU {result['gpu_id']}) "
                              f"failed with return code {result['return_code']}")
                        if 'error' in result:
                            print(f"   Error: {result['error']}")
                        if result.get('stderr'):
                            print(f"   stderr: {result['stderr'][:200]}...")

                except Exception as e:
                    print(f"❌ Error getting process result: {e}")
                    results.append({
                        'process_id': config['process_id'],
                        'gpu_id': config['gpu_id'],
                        'return_code': -1,
                        'error': str(e)
                    })

        end_time = time.time()
        total_time = end_time - start_time

        # Generate execution summary
        summary = self._generate_execution_summary(results, total_time)

        # Finalize results
        final_results = self._finalize_results(process_configs)
        summary.update(final_results)

        return summary

    def _generate_execution_summary(self, results: List[Dict], total_time: float) -> Dict:
        """
        Generate execution summary statistics.

        Args:
            results (List[Dict]): Process execution results
            total_time (float): Total execution time in seconds

        Returns:
            Dict: Execution summary statistics
        """
        successful = [r for r in results if r['return_code'] == 0]
        failed = [r for r in results if r['return_code'] != 0]

        summary = {
            'execution_summary': {
                'total_processes': len(results),
                'successful_processes': len(successful),
                'failed_processes': len(failed),
                'success_rate': len(successful) / len(results) * 100 if results else 0,
                'total_execution_time': total_time,
                'average_process_time': (sum(r.get('execution_time', 0) for r in successful) /
                                       len(successful) if successful else 0),
                'total_videos_processed': sum(r.get('video_count', 0) for r in successful),
                'processing_speed': (sum(r.get('video_count', 0) for r in successful) /
                                   total_time if total_time > 0 else 0)
            },
            'failed_processes': failed
        }

        return summary

    def _finalize_results(self, process_configs: List[Dict]) -> Dict:
        """
        Merge and finalize evaluation results from all processes.

        Args:
            process_configs (List[Dict]): Process configuration list

        Returns:
            Dict: Final results summary
        """
        print("\n📊 Merging evaluation results...")

        distributor = VideoTaskDistributor(self.video_dir, self.output_dir)

        # Load all process configurations (including completed ones)
        task_config_path = os.path.join(self.output_dir, "task_distribution.json")
        if os.path.exists(task_config_path):
            with open(task_config_path, 'r', encoding='utf-8') as f:
                task_config = json.load(f)
                all_process_configs = task_config['process_configs']
        else:
            all_process_configs = process_configs

        # Merge results
        final_csv_path = distributor.merge_results(
            all_process_configs,
            "vbench_parallel_final_results.csv"
        )

        if final_csv_path and os.path.exists(final_csv_path):
            # Read final results for statistics
            df = pd.read_csv(final_csv_path)

            return {
                'final_results': {
                    'output_file': final_csv_path,
                    'total_videos_evaluated': len(df),
                    'output_format': 'CSV with VBench 4 metrics + normalized scores',
                    'columns': list(df.columns),
                    'sample_data': df.head(3).to_dict('records') if len(df) > 0 else []
                }
            }
        else:
            return {
                'final_results': {
                    'error': 'Failed to merge results',
                    'output_file': None
                }
            }

    def print_final_summary(self, summary: Dict):
        """
        Print comprehensive final summary of the evaluation.

        Args:
            summary (Dict): Complete evaluation summary
        """
        print("\n" + "="*80)
        print("🎉 VBench Parallel Evaluation Complete")
        print("="*80)

        if 'execution_summary' in summary:
            exec_summary = summary['execution_summary']
            print(f"📊 Execution Statistics:")
            print(f"   Total processes: {exec_summary['total_processes']}")
            print(f"   Successful processes: {exec_summary['successful_processes']}")
            print(f"   Failed processes: {exec_summary['failed_processes']}")
            print(f"   Success rate: {exec_summary['success_rate']:.1f}%")
            print(f"   Total execution time: {exec_summary['total_execution_time']:.1f}s")
            print(f"   Average process time: {exec_summary['average_process_time']:.1f}s")
            print(f"   Total videos processed: {exec_summary['total_videos_processed']}")
            print(f"   Processing speed: {exec_summary['processing_speed']:.2f} videos/sec")

        if 'final_results' in summary:
            final_results = summary['final_results']
            if 'output_file' in final_results and final_results['output_file']:
                print(f"\n📁 Final results file: {final_results['output_file']}")
                print(f"📊 Total videos evaluated: {final_results['total_videos_evaluated']}")
                print(f"📋 Output format: {final_results['output_format']}")
            else:
                print(f"\n❌ Results merge failed")

        print("="*80)


def main():
    """
    Main function for standalone execution.

    Usage: python parallel_vbench_evaluator.py <video_directory> [output_directory]
    """
    if len(sys.argv) < 2:
        print("Usage: python parallel_vbench_evaluator.py <video_directory> [output_directory]")
        print("\nExample:")
        print("  python parallel_vbench_evaluator.py /path/to/videos/ results/")
        print("\nThis will:")
        print("  - Analyze available GPU resources")
        print("  - Distribute videos across multiple processes")
        print("  - Run VBench evaluation in parallel")
        print("  - Merge results into a single CSV file")
        sys.exit(1)

    video_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "parallel_vbench_results"

    if not os.path.exists(video_dir):
        print(f"❌ Video directory not found: {video_dir}")
        sys.exit(1)

    # Create parallel evaluator
    evaluator = ParallelVBenchEvaluator(video_dir, output_dir)

    try:
        # Run parallel evaluation
        summary = evaluator.run_parallel_evaluation()

        # Print final summary
        evaluator.print_final_summary(summary)

        # Save summary to file
        summary_file = os.path.join(output_dir, "evaluation_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"📄 Evaluation summary saved: {summary_file}")

    except KeyboardInterrupt:
        print("\n⚠️  User interrupted evaluation")
        evaluator._cleanup_processes()
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        evaluator._cleanup_processes()
        sys.exit(1)


if __name__ == "__main__":
    main()
