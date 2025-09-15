#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Benchmark Metrics Package for AirScape.code

This package provides comprehensive video quality evaluation using VBench metrics.
It includes both single-process and parallel evaluation capabilities with intelligent
resource management and integration support for the AirScape.code project.
"""

# Import main classes for easy access
try:
    from .single_process_evaluator import SingleProcessEvaluator
    from .parallel_vbench_evaluator import ParallelVBenchEvaluator
    from .gpu_resource_manager import GPUResourceManager
    from .video_task_distributor import VideoTaskDistributor
    from .integration_example import AirScapeVideoEvaluator
    
    __all__ = [
        'SingleProcessEvaluator',
        'ParallelVBenchEvaluator', 
        'GPUResourceManager',
        'VideoTaskDistributor',
        'AirScapeVideoEvaluator'
    ]
    
except ImportError as e:
    # Handle import errors gracefully
    print(f"Warning: Some modules could not be imported: {e}")
    __all__ = []

# Package metadata
SUPPORTED_DIMENSIONS = [
    'imaging_quality',      # Technical image quality (MUSIQ-based)
    'motion_smoothness',    # Motion continuity assessment
    'dynamic_degree',       # Dynamic content analysis
    'aesthetic_quality'     # Aesthetic appeal evaluation
]

DEFAULT_WEIGHTS = {
    'imaging_quality': 0.25,
    'motion_smoothness': 0.25,
    'dynamic_degree': 0.25,
    'aesthetic_quality': 0.25
}

# Utility functions
def get_version():
    """Get package version."""
    return __version__

def get_supported_dimensions():
    """Get list of supported VBench evaluation dimensions."""
    return SUPPORTED_DIMENSIONS.copy()

def get_default_weights():
    """Get default weights for composite scoring."""
    return DEFAULT_WEIGHTS.copy()

def check_vbench_availability():
    """Check if VBench is available and properly configured."""
    try:
        from vbench import VBench
        return True
    except ImportError:
        return False

# Package information
def print_package_info():
    """Print package information."""
    print(f"Video Benchmark Metrics v{__version__}")
    print(f"Supported dimensions: {', '.join(SUPPORTED_DIMENSIONS)}")
    print(f"VBench available: {check_vbench_availability()}")
