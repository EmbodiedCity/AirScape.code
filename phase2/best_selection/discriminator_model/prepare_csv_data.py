#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare CSV Data for Discriminator Model

This script combines VBench video quality metrics with trajectory similarity scores
to create the input data for the discriminator model training.

Updated for AirScape.code integration with video_benchmark_metrics module.
"""

import csv
import os
import sys
import pandas as pd
from pathlib import Path

# Add video_benchmark_metrics to path if available
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'video_benchmark_metrics'))

def prepare_discriminator_data(metrics_csv_path: str, traj_csv_path: str,
                             output_csv_path: str, use_normalized: bool = True) -> bool:
    """
    Combine VBench metrics with trajectory similarity for discriminator training.

    Args:
        metrics_csv_path (str): Path to VBench evaluation results CSV
        traj_csv_path (str): Path to trajectory similarity CSV
        output_csv_path (str): Path for combined output CSV
        use_normalized (bool): Whether to use normalized scores from VBench

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Validate input files
        if not os.path.exists(metrics_csv_path):
            print(f"❌ VBench metrics CSV not found: {metrics_csv_path}")
            return False

        # Load VBench metrics
        print(f"📊 Loading VBench metrics from: {metrics_csv_path}")
        metrics_df = pd.read_csv(metrics_csv_path)
        print(f"   Found {len(metrics_df)} videos with metrics")

        # Load trajectory similarity (optional)
        similarity_dict = {}
        if os.path.exists(traj_csv_path):
            print(f"📊 Loading trajectory similarity from: {traj_csv_path}")
            traj_df = pd.read_csv(traj_csv_path)
            similarity_dict = dict(zip(traj_df.iloc[:, 0], traj_df.iloc[:, 1]))
            print(f"   Found {len(similarity_dict)} videos with trajectory scores")
        else:
            print(f"⚠️  Trajectory CSV not found: {traj_csv_path}")
            print("   Using default similarity score of 0.0")

        # Select appropriate metric columns
        base_metrics = ['imaging_quality', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality']

        if use_normalized:
            # Try to use normalized columns first
            metric_columns = []
            for metric in base_metrics:
                norm_col = f"{metric}_normalized"
                if norm_col in metrics_df.columns:
                    metric_columns.append(norm_col)
                    print(f"✅ Using normalized column: {norm_col}")
                elif metric in metrics_df.columns:
                    metric_columns.append(metric)
                    print(f"⚠️  Using raw column (normalized not found): {metric}")
                else:
                    print(f"❌ Column not found: {metric}")
                    return False
        else:
            # Use raw metric columns
            metric_columns = []
            for metric in base_metrics:
                if metric in metrics_df.columns:
                    metric_columns.append(metric)
                else:
                    print(f"❌ Column not found: {metric}")
                    return False

        # Create output dataframe
        output_columns = ['video_name'] + base_metrics + ['similarity']
        output_data = []

        for _, row in metrics_df.iterrows():
            video_name = row['video_name']

            # Get metric values
            metric_values = []
            for i, col in enumerate(metric_columns):
                value = row[col]
                metric_values.append(value)

            # Get similarity score
            similarity = similarity_dict.get(video_name, 0.0)

            # Create output row
            output_row = [video_name] + metric_values + [similarity]
            output_data.append(output_row)

        # Save combined data
        output_df = pd.DataFrame(output_data, columns=output_columns)
        output_df.to_csv(output_csv_path, index=False)

        print(f"✅ Combined data saved: {output_csv_path}")
        print(f"📊 Final dataset: {len(output_df)} videos with 5 metrics")

        # Print summary statistics
        print(f"\n📈 Data Summary:")
        for metric in base_metrics:
            if metric in output_df.columns:
                values = output_df[metric]
                print(f"   {metric}: mean={values.mean():.3f}, std={values.std():.3f}, range=[{values.min():.3f}, {values.max():.3f}]")

        similarity_values = output_df['similarity']
        print(f"   similarity: mean={similarity_values.mean():.3f}, std={similarity_values.std():.3f}, range=[{similarity_values.min():.3f}, {similarity_values.max():.3f}]")

        return True

    except Exception as e:
        print(f"❌ Error preparing discriminator data: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function with example usage.

    Update the paths below to match your data locations.
    """
    print("🔧 Preparing CSV Data for Discriminator Model")

    # Configuration - UPDATE THESE PATHS
    metrics_csv_path = "path/to/vbench_results.csv"  # VBench evaluation results
    traj_csv_path = "path/to/trajectory_similarity.csv"  # Trajectory similarity scores
    output_csv_path = "path/to/discriminator_input.csv"  # Combined output for discriminator

    # Check if paths are configured
    if any(path.startswith("path/to/") for path in [metrics_csv_path, traj_csv_path, output_csv_path]):
        print("❌ Please update the file paths in the script before running")
        print("\nRequired files:")
        print(f"  - VBench metrics CSV: {metrics_csv_path}")
        print(f"  - Trajectory similarity CSV: {traj_csv_path}")
        print(f"  - Output CSV: {output_csv_path}")
        return

    # Prepare discriminator data
    success = prepare_discriminator_data(
        metrics_csv_path=metrics_csv_path,
        traj_csv_path=traj_csv_path,
        output_csv_path=output_csv_path,
        use_normalized=True  # Use normalized VBench scores
    )

    if success:
        print("🎉 Data preparation completed successfully!")
        print(f"📁 Output file ready for discriminator training: {output_csv_path}")
    else:
        print("❌ Data preparation failed")

if __name__ == "__main__":
    main()
