#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
import pandas as pd
import argparse
from pathlib import Path

def extract_id_from_filename(filename: str):
    """
    Extract numeric ID from the beginning of video filename
    Example: 00942_NAT2021_test_N01001_1.mp4 -> 00942
            00944_NAT2021_test_N01002_0.mp4 -> 00944
    """
    m = re.search(r'^(\d+)', filename)
    return m.group(1) if m else None

def main():
    parser = argparse.ArgumentParser(description='Generate video filename and prompt list')
    parser.add_argument('--csv_path', type=str, default='your_path/.csv', help='CSV file absolute path')
    parser.add_argument('--video_dir', type=str, default='your_path', help='Original video folder absolute path')
    parser.add_argument('--output_dir', type=str, default='your_path', help='Output train-dataset directory absolute path')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    videos_dir = output_dir / 'videos'
    videos_txt = output_dir / 'videos.txt'
    prompts_txt = output_dir / 'prompts.txt'
    os.makedirs(videos_dir, exist_ok=True)

    # Read CSV
    try:
        csv_data = pd.read_csv(args.csv_path, dtype={'extracted_id': str})
        id2prompt = {}
        for _, row in csv_data.iterrows():
            extracted_id = str(row['extracted_id']).zfill(5)  # Ensure zero-padding alignment
            prompt = str(row['description']) if isinstance(row['description'], str) else None
            if prompt and prompt.strip():
                prompt = prompt.replace('\r', '').strip()
                prompt = re.sub(r'\s+', ' ', prompt)
                id2prompt[extracted_id] = prompt
        print(f"CSV contains {len(id2prompt)} valid ID-prompt pairs")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    video_files = [f for f in os.listdir(args.video_dir) if f.endswith('.mp4')]
    video_files.sort()

    valid_videos, valid_prompts = [], []
    matched_pairs, unmatched_videos, missing_csv_ids = [], [], []

    print(f"\nStarting matching... Total videos: {len(video_files)}")
    for video_file in video_files:
        vid_id = extract_id_from_filename(video_file)
        if vid_id and vid_id in id2prompt:
            prompt = id2prompt[vid_id]
            shutil.copy2(os.path.join(args.video_dir, video_file), videos_dir / video_file)
            valid_videos.append(video_file)
            valid_prompts.append(prompt)
            matched_pairs.append((video_file, vid_id, prompt))
            print(f"✓ Match successful: {video_file} -> {vid_id}")
        else:
            unmatched_videos.append(video_file)
            print(f"✗ No match: {video_file}")

    matched_ids = set(pair[1] for pair in matched_pairs)
    for extracted_id in id2prompt.keys():
        if extracted_id not in matched_ids:
            missing_csv_ids.append(extracted_id)

    with open(videos_txt, 'w', encoding='utf-8') as f:
        for v in valid_videos:
            f.write('videos/' + v + '\n')

    with open(prompts_txt, 'w', encoding='utf-8') as f:
        for p in valid_prompts:
            f.write(p + '\n')

    print(f"\n=== Complete ===")
    print(f"Successfully matched: {len(matched_pairs)}")
    print(f"Unmatched videos: {len(unmatched_videos)}")
    print(f"Missing videos in CSV: {len(missing_csv_ids)}")

if __name__ == "__main__":
    main()
