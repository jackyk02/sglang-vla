#!/usr/bin/env python3
"""
Bridge Dataset Downloader

This script automatically downloads language instructions (first line of lang.txt)
and first images (im_0.jpg) from Bridge dataset trajectories.

Usage:
    python bridge_dataset_downloader.py --base-url <base_url> --output-dir <output_dir>
    
Example:
    python bridge_dataset_downloader.py \
        --base-url "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/" \
        --output-dir "./bridge_data"
"""

import os
import sys
import argparse
import requests
import json
import csv
from urllib.parse import urljoin, urlparse
from pathlib import Path
import time
from typing import List, Tuple, Optional
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BridgeDatasetDownloader:
    def __init__(self, base_url: str, output_dir: str, max_workers: int = 10):
        self.base_url = base_url.rstrip('/')
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.session = requests.Session()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        # File to store language instructions
        self.lang_file = self.output_dir / "language_instructions.csv"
        
        # Statistics
        self.downloaded_trajectories = 0
        self.failed_downloads = 0
        
    def discover_trajectories(self, dataset_path: str = "") -> List[str]:
        """
        Discover all trajectory paths by exploring the dataset structure.
        This is a simple implementation that tries common patterns.
        """
        trajectories = []
        
        # Try some common dataset patterns based on the example
        # You might need to adjust these patterns based on the actual dataset structure
        common_patterns = [
            "datacol2_toykitchen7/drawer_pnp/01/*/raw/traj_group*/traj*",
            "datacol2_toykitchen7/*/01/*/raw/traj_group*/traj*",
            "*/drawer_pnp/01/*/raw/traj_group*/traj*",
        ]
        
        logger.info("Discovering trajectories... This might take a while.")
        
        # For now, we'll use a simpler approach - try the example pattern first
        example_traj = "datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj6"
        trajectories.append(example_traj)
        
        # Try to discover more by varying the numbers
        base_pattern = "datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0"
        for traj_num in range(20):  # Try traj0 to traj19
            traj_path = f"{base_pattern}/traj{traj_num}"
            if self.check_trajectory_exists(traj_path):
                if traj_path not in trajectories:
                    trajectories.append(traj_path)
        
        logger.info(f"Found {len(trajectories)} trajectories to process.")
        return trajectories
    
    def check_trajectory_exists(self, traj_path: str) -> bool:
        """Check if a trajectory path exists by trying to access lang.txt."""
        lang_url = f"{self.base_url}/{traj_path}/lang.txt"
        try:
            response = self.session.head(lang_url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def download_file(self, url: str, local_path: Path, timeout: int = 30) -> bool:
        """Download a file from URL to local path."""
        try:
            response = self.session.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def get_language_instruction(self, traj_path: str) -> Optional[str]:
        """Get the first line of lang.txt for a trajectory."""
        lang_url = f"{self.base_url}/{traj_path}/lang.txt"
        try:
            response = self.session.get(lang_url, timeout=10)
            response.raise_for_status()
            
            # Get the first line (the instruction)
            lines = response.text.strip().split('\n')
            if lines:
                return lines[0].strip()
            return None
        except Exception as e:
            logger.error(f"Failed to get language instruction from {lang_url}: {e}")
            return None
    
    def download_trajectory_data(self, traj_path: str) -> Tuple[bool, str]:
        """Download both language instruction and first image for a trajectory."""
        try:
            # Get language instruction
            instruction = self.get_language_instruction(traj_path)
            if not instruction:
                return False, f"Failed to get language instruction for {traj_path}"
            
            # Download first image
            image_url = f"{self.base_url}/{traj_path}/images0/im_0.jpg"
            
            # Create a safe filename from trajectory path
            safe_traj_name = traj_path.replace('/', '_').replace('\\', '_')
            image_local_path = self.images_dir / f"{safe_traj_name}_im_0.jpg"
            
            image_success = self.download_file(image_url, image_local_path)
            if not image_success:
                return False, f"Failed to download image for {traj_path}"
            
            # Save language instruction to CSV
            self.save_language_instruction(traj_path, instruction, image_local_path.name)
            
            logger.info(f"Successfully processed trajectory: {traj_path}")
            return True, "Success"
            
        except Exception as e:
            error_msg = f"Error processing trajectory {traj_path}: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def save_language_instruction(self, traj_path: str, instruction: str, image_filename: str):
        """Save language instruction to CSV file."""
        file_exists = self.lang_file.exists()
        
        with open(self.lang_file, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['trajectory_path', 'language_instruction', 'image_filename']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'trajectory_path': traj_path,
                'language_instruction': instruction,
                'image_filename': image_filename
            })
    
    def download_all_trajectories(self, trajectories: List[str]):
        """Download data for all trajectories using thread pool."""
        logger.info(f"Starting download of {len(trajectories)} trajectories...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_traj = {
                executor.submit(self.download_trajectory_data, traj): traj 
                for traj in trajectories
            }
            
            # Process completed tasks
            for future in as_completed(future_to_traj):
                traj = future_to_traj[future]
                try:
                    success, message = future.result()
                    if success:
                        self.downloaded_trajectories += 1
                    else:
                        self.failed_downloads += 1
                        logger.warning(f"Failed: {traj} - {message}")
                except Exception as e:
                    self.failed_downloads += 1
                    logger.error(f"Exception for trajectory {traj}: {e}")
                
                # Progress update
                total_processed = self.downloaded_trajectories + self.failed_downloads
                if total_processed % 10 == 0:
                    logger.info(f"Progress: {total_processed}/{len(trajectories)} trajectories processed")
    
    def generate_summary(self):
        """Generate a summary of the download process."""
        summary = {
            "total_trajectories_processed": self.downloaded_trajectories + self.failed_downloads,
            "successful_downloads": self.downloaded_trajectories,
            "failed_downloads": self.failed_downloads,
            "output_directory": str(self.output_dir),
            "language_instructions_file": str(self.lang_file),
            "images_directory": str(self.images_dir)
        }
        
        summary_file = self.output_dir / "download_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Download Summary:")
        logger.info(f"  Total processed: {summary['total_trajectories_processed']}")
        logger.info(f"  Successful: {summary['successful_downloads']}")
        logger.info(f"  Failed: {summary['failed_downloads']}")
        logger.info(f"  Output directory: {summary['output_directory']}")
        logger.info(f"  Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Download Bridge dataset trajectories')
    parser.add_argument(
        '--base-url', 
        default='https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2',
        help='Base URL of the Bridge dataset'
    )
    parser.add_argument(
        '--output-dir', 
        default='./bridge_data',
        help='Output directory for downloaded data'
    )
    parser.add_argument(
        '--max-workers', 
        type=int, 
        default=5,
        help='Maximum number of concurrent downloads'
    )
    parser.add_argument(
        '--trajectories', 
        nargs='*',
        help='Specific trajectory paths to download (if not provided, will auto-discover)'
    )
    parser.add_argument(
        '--single-trajectory',
        help='Download a single trajectory (example format: datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj6)'
    )
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = BridgeDatasetDownloader(
        base_url=args.base_url,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    # Determine which trajectories to download
    if args.single_trajectory:
        trajectories = [args.single_trajectory]
    elif args.trajectories:
        trajectories = args.trajectories
    else:
        trajectories = downloader.discover_trajectories()
    
    if not trajectories:
        logger.error("No trajectories found to download!")
        sys.exit(1)
    
    # Download trajectories
    start_time = time.time()
    downloader.download_all_trajectories(trajectories)
    end_time = time.time()
    
    logger.info(f"Download completed in {end_time - start_time:.2f} seconds")
    downloader.generate_summary()

if __name__ == "__main__":
    main() 