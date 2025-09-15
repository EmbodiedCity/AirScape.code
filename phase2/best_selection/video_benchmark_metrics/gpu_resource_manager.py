#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Resource Manager for AirScape.code

This module provides intelligent GPU resource allocation for parallel video evaluation.
It analyzes available GPU resources and distributes processes efficiently.
"""

import subprocess
import time
import os
from typing import List, Dict, Tuple, Optional
import json


class GPUResourceManager:
    """
    Intelligent GPU resource manager for parallel processing.
    
    This class analyzes available GPU resources, monitors memory usage,
    and provides optimal process allocation strategies.
    """

    def __init__(self, memory_per_process: float = 2.0, safety_margin: float = 0.8):
        """
        Initialize GPU resource manager.
        
        Args:
            memory_per_process (float): Expected memory usage per process in GB
            safety_margin (float): Safety margin factor to avoid memory overflow
        """
        self.memory_per_process = memory_per_process
        self.safety_margin = safety_margin
        self.gpu_info = self.get_gpu_status()

    def get_gpu_status(self) -> List[Dict]:
        """
        Get status information for all available GPUs.
        
        Returns:
            List[Dict]: List of GPU status dictionaries
        """
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                print(f"❌ nvidia-smi command failed: {result.stderr}")
                return []
                
            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        try:
                            gpu_info.append({
                                'index': int(parts[0]),
                                'name': parts[1],
                                'memory_used': int(parts[2]),  # MB
                                'memory_total': int(parts[3]),  # MB
                                'utilization': int(parts[4])  # %
                            })
                        except ValueError as e:
                            print(f"⚠️  Error parsing GPU info: {e}")
                            continue
            
            return gpu_info
            
        except subprocess.TimeoutExpired:
            print("❌ nvidia-smi command timed out")
            return []
        except FileNotFoundError:
            print("❌ nvidia-smi not found. Please ensure NVIDIA drivers are installed.")
            return []
        except Exception as e:
            print(f"❌ Error getting GPU status: {e}")
            return []

    def get_available_gpus(self, min_free_memory_gb: float = 1.0, 
                          max_utilization: int = 20) -> List[Dict]:
        """
        Get list of available GPUs based on memory and utilization criteria.
        
        Args:
            min_free_memory_gb (float): Minimum free memory required in GB
            max_utilization (int): Maximum GPU utilization percentage
            
        Returns:
            List[Dict]: List of available GPU information
        """
        available_gpus = []
        
        for gpu in self.gpu_info:
            free_memory_mb = gpu['memory_total'] - gpu['memory_used']
            free_memory_gb = free_memory_mb / 1024.0
            
            if (free_memory_gb >= min_free_memory_gb and 
                gpu['utilization'] <= max_utilization):
                
                gpu['free_memory_gb'] = free_memory_gb
                available_gpus.append(gpu)
        
        return available_gpus

    def calculate_processes_per_gpu(self, gpu: Dict) -> int:
        """
        Calculate optimal number of processes for a GPU.
        
        Args:
            gpu (Dict): GPU information dictionary
            
        Returns:
            int: Number of processes that can run on this GPU
        """
        free_memory_gb = gpu.get('free_memory_gb', 0)
        usable_memory = free_memory_gb * self.safety_margin
        
        max_processes = int(usable_memory / self.memory_per_process)
        return max(0, max_processes)

    def generate_process_allocation(self) -> Tuple[List[Dict], int]:
        """
        Generate optimal process allocation across available GPUs.
        
        Returns:
            Tuple[List[Dict], int]: (gpu_allocation_list, total_processes)
        """
        available_gpus = self.get_available_gpus()
        
        if not available_gpus:
            print("❌ No available GPUs found")
            return [], 0
        
        gpu_allocation = []
        total_processes = 0
        
        for gpu in available_gpus:
            processes_count = self.calculate_processes_per_gpu(gpu)
            
            if processes_count > 0:
                allocation = {
                    'gpu_id': gpu['index'],
                    'gpu_name': gpu['name'],
                    'processes_count': processes_count,
                    'free_memory_gb': gpu['free_memory_gb'],
                    'utilization': gpu['utilization']
                }
                gpu_allocation.append(allocation)
                total_processes += processes_count
        
        return gpu_allocation, total_processes

    def print_allocation_summary(self, gpu_allocation: List[Dict], total_processes: int):
        """
        Print a summary of GPU allocation.
        
        Args:
            gpu_allocation (List[Dict]): GPU allocation list
            total_processes (int): Total number of processes
        """
        print(f"\n📊 GPU Resource Allocation Summary:")
        print(f"   Total available GPUs: {len(gpu_allocation)}")
        print(f"   Total processes: {total_processes}")
        print(f"   Memory per process: {self.memory_per_process:.1f} GB")
        print(f"   Safety margin: {self.safety_margin:.1f}")
        
        print(f"\n🎯 GPU Allocation Details:")
        for allocation in gpu_allocation:
            print(f"   GPU {allocation['gpu_id']} ({allocation['gpu_name']}):")
            print(f"     - Processes: {allocation['processes_count']}")
            print(f"     - Free memory: {allocation['free_memory_gb']:.1f} GB")
            print(f"     - Utilization: {allocation['utilization']}%")

    def save_allocation_config(self, gpu_allocation: List[Dict], config_path: str):
        """
        Save GPU allocation configuration to file.
        
        Args:
            gpu_allocation (List[Dict]): GPU allocation list
            config_path (str): Path to save configuration file
        """
        config = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'memory_per_process': self.memory_per_process,
            'safety_margin': self.safety_margin,
            'gpu_allocation': gpu_allocation,
            'total_processes': sum(alloc['processes_count'] for alloc in gpu_allocation)
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"💾 GPU allocation config saved: {config_path}")

    @classmethod
    def load_allocation_config(cls, config_path: str) -> Optional[Dict]:
        """
        Load GPU allocation configuration from file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            Optional[Dict]: Configuration dictionary or None if failed
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load GPU allocation config: {e}")
            return None

    def print_gpu_status(self):
        """Print current status of all GPUs."""
        print("\n🖥️  Current GPU Status:")
        
        if not self.gpu_info:
            print("   No GPUs found or nvidia-smi not available")
            return
        
        for gpu in self.gpu_info:
            free_memory = gpu['memory_total'] - gpu['memory_used']
            print(f"   GPU {gpu['index']} ({gpu['name']}):")
            print(f"     - Memory: {gpu['memory_used']}/{gpu['memory_total']} MB "
                  f"({free_memory} MB free)")
            print(f"     - Utilization: {gpu['utilization']}%")


def main():
    """
    Main function for standalone testing.
    """
    print("🔍 GPU Resource Manager - Standalone Test")
    
    manager = GPUResourceManager()
    
    # Print current GPU status
    manager.print_gpu_status()
    
    # Generate allocation
    gpu_allocation, total_processes = manager.generate_process_allocation()
    
    if gpu_allocation:
        manager.print_allocation_summary(gpu_allocation, total_processes)
        
        # Save configuration
        config_path = "gpu_allocation_test.json"
        manager.save_allocation_config(gpu_allocation, config_path)
        
        # Test loading
        loaded_config = GPUResourceManager.load_allocation_config(config_path)
        if loaded_config:
            print(f"✅ Configuration loaded successfully")
        
        # Cleanup test file
        if os.path.exists(config_path):
            os.remove(config_path)
    else:
        print("❌ No GPU allocation possible")


if __name__ == "__main__":
    main()
