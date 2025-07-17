#!/usr/bin/env python3
"""
Extract specific Bridge dataset trajectories with custom organization

This script downloads the 10 specific trajectories provided by the user,
organizes images as 1.jpg, 2.jpg, etc., and puts instructions in a CSV file.
"""

import os
import csv
import requests
import logging
from pathlib import Path
from typing import List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpecificTrajectoryExtractor:
    def __init__(self, base_url: str, output_dir: str):
        self.base_url = base_url.rstrip('/')
        self.output_dir = Path(output_dir)
        self.session = requests.Session()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        # File to store language instructions
        self.csv_file = self.output_dir / "instructions.csv"
        
        # Statistics
        self.successful_extractions = 0
        self.failed_extractions = 0

    def extract_trajectory_path(self, full_url: str) -> str:
        """Extract trajectory path from full URL."""
        # Remove the base URL to get just the trajectory path
        base_url_part = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/"
        if full_url.startswith(base_url_part):
            trajectory_path = full_url[len(base_url_part):].rstrip('/')
            return trajectory_path
        else:
            raise ValueError(f"URL doesn't start with expected base: {full_url}")

    def get_language_instruction(self, traj_path: str) -> Optional[str]:
        """Get the first line of lang.txt for a trajectory."""
        lang_url = f"{self.base_url}/{traj_path}/lang.txt"
        try:
            logger.info(f"Fetching instruction from: {lang_url}")
            response = self.session.get(lang_url, timeout=15)
            response.raise_for_status()
            
            # Get the first line (the instruction)
            lines = response.text.strip().split('\n')
            if lines:
                return lines[0].strip()
            return None
        except Exception as e:
            logger.error(f"Failed to get language instruction from {lang_url}: {e}")
            return None

    def download_image(self, traj_path: str, image_number: int) -> bool:
        """Download first image for a trajectory with numbered filename."""
        image_url = f"{self.base_url}/{traj_path}/images0/im_0.jpg"
        image_local_path = self.images_dir / f"{image_number}.jpg"
        
        try:
            logger.info(f"Downloading image from: {image_url}")
            response = self.session.get(image_url, timeout=30, stream=True)
            response.raise_for_status()
            
            with open(image_local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Saved image as: {image_local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download image from {image_url}: {e}")
            return False

    def extract_trajectory_data(self, full_url: str, image_number: int) -> Tuple[bool, str, Optional[str]]:
        """Extract both language instruction and image for a trajectory."""
        try:
            # Extract trajectory path from full URL
            traj_path = self.extract_trajectory_path(full_url)
            logger.info(f"Processing trajectory {image_number}: {traj_path}")
            
            # Get language instruction
            instruction = self.get_language_instruction(traj_path)
            if not instruction:
                return False, f"Failed to get language instruction for {traj_path}", None
            
            # Download image with numbered filename
            image_success = self.download_image(traj_path, image_number)
            if not image_success:
                return False, f"Failed to download image for {traj_path}", instruction
            
            logger.info(f"Successfully processed trajectory {image_number}: {traj_path}")
            return True, "Success", instruction
            
        except Exception as e:
            error_msg = f"Error processing trajectory {image_number}: {e}"
            logger.error(error_msg)
            return False, error_msg, None

    def save_instructions_to_csv(self, instructions_data: List[Tuple[int, str, str]]):
        """Save all language instructions to CSV file."""
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['image_number', 'trajectory_path', 'instruction']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for image_num, traj_path, instruction in instructions_data:
                writer.writerow({
                    'image_number': image_num,
                    'trajectory_path': traj_path,
                    'instruction': instruction
                })
        
        logger.info(f"Instructions saved to: {self.csv_file}")

    def extract_all_trajectories(self, trajectory_urls: List[str]):
        """Extract data for all trajectories."""
        logger.info(f"Starting extraction of {len(trajectory_urls)} trajectories...")
        
        instructions_data = []
        
        for i, url in enumerate(trajectory_urls, 1):
            success, message, instruction = self.extract_trajectory_data(url, i)
            
            if success and instruction:
                try:
                    traj_path = self.extract_trajectory_path(url)
                    instructions_data.append((i, traj_path, instruction))
                    self.successful_extractions += 1
                except:
                    self.failed_extractions += 1
            else:
                self.failed_extractions += 1
                logger.warning(f"Failed: trajectory {i} - {message}")
        
        # Save all instructions to CSV
        if instructions_data:
            self.save_instructions_to_csv(instructions_data)
        
        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print extraction summary."""
        total = self.successful_extractions + self.failed_extractions
        logger.info(f"\n=== Extraction Summary ===")
        logger.info(f"Total trajectories processed: {total}")
        logger.info(f"Successful extractions: {self.successful_extractions}")
        logger.info(f"Failed extractions: {self.failed_extractions}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Images saved to: {self.images_dir}")
        logger.info(f"Instructions saved to: {self.csv_file}")

def main():
    # The 10 trajectory URLs provided by the user
    trajectory_urls = [
        "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/02/2023-03-15_14-02-55/raw/traj_group0/traj4/",
        "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol1_toykitchen6/pnp_sweep/04/2023-01-25_18-48-39/raw/traj_group0/traj6/",
        "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_folding_table/stack_blocks/01/2023-05-29_10-36-05/raw/traj_group0/traj0/",
        "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_laundry_machine/pnp_sweep/00/2023-02-10_11-41-14/raw/traj_group0/traj0/",
        "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/many_skills/06/2023-04-13_15-27-06/raw/traj_group0/traj9/",
            "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen2/many_skills/19/2023-03-08_15-25-38/raw/traj_group0/traj10/",
        "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/deepthought_folding_table/stack_blocks/11/2023-05-01_15-35-39/raw/traj_group0/traj5/",
        "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/deepthought_robot_desk/drawer_pnp/18/2023-04-13_10-29-21/raw/traj_group0/traj13/",
        "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/deepthought_toykitchen1/many_skills/05/2023-04-04_13-11-59/raw/traj_group0/traj13/",
                    "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen2/many_skills/09/2023-03-08_14-08-59/raw/traj_group0/traj7/",
    ]
    
    # Initialize extractor
    extractor = SpecificTrajectoryExtractor(
        base_url="https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2",
        output_dir="./extracted_trajectories"
    )
    
    # Extract all trajectories
    extractor.extract_all_trajectories(trajectory_urls)
    
    print("\n✅ Extraction completed!")
    print(f"📁 Images saved as: 1.jpg, 2.jpg, ..., 10.jpg in ./extracted_trajectories/images/")
    print(f"📄 Instructions saved to: ./extracted_trajectories/instructions.csv")

if __name__ == "__main__":
    main() 