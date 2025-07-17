#!/usr/bin/env python3
"""
Example usage of the Bridge Dataset Downloader

This script demonstrates how to download data from the Bridge dataset
using the provided downloader.
"""

from bridge_dataset_downloader import BridgeDatasetDownloader
import logging

# Set up logging to see progress
logging.basicConfig(level=logging.INFO)

def download_example_trajectory():
    """Download the example trajectory provided by the user."""
    
    # Initialize the downloader
    downloader = BridgeDatasetDownloader(
        base_url="https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2",
        output_dir="./bridge_data_example",
        max_workers=1  # Use 1 worker for this example
    )
    
    # The specific trajectory from the user's example
    example_trajectory = "datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj6"
    
    print(f"Downloading example trajectory: {example_trajectory}")
    
    # Download the trajectory data
    success, message = downloader.download_trajectory_data(example_trajectory)
    
    if success:
        print("✅ Successfully downloaded:")
        print(f"   Language instruction saved to: {downloader.lang_file}")
        print(f"   Image saved to: {downloader.images_dir}")
        print("\nFiles created:")
        print(f"   - {downloader.lang_file}")
        print(f"   - {downloader.images_dir}/datacol2_toykitchen7_drawer_pnp_01_2023-04-19_09-18-15_raw_traj_group0_traj6_im_0.jpg")
    else:
        print(f"❌ Failed to download: {message}")
    
    # Generate summary
    downloader.generate_summary()

def download_multiple_trajectories():
    """Example of downloading multiple trajectories automatically."""
    
    # Initialize the downloader
    downloader = BridgeDatasetDownloader(
        base_url="https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2",
        output_dir="./bridge_data_multiple",
        max_workers=3  # Use 3 concurrent downloads
    )
    
    print("Discovering and downloading multiple trajectories...")
    
    # Auto-discover trajectories (this will try to find more trajectories)
    trajectories = downloader.discover_trajectories()
    
    if trajectories:
        print(f"Found {len(trajectories)} trajectories to download")
        downloader.download_all_trajectories(trajectories)
        downloader.generate_summary()
    else:
        print("No trajectories found for automatic discovery")

if __name__ == "__main__":
    print("Bridge Dataset Downloader - Example Usage")
    print("=" * 50)
    
    print("\n1. Downloading the specific example trajectory...")
    download_example_trajectory()
    
    print("\n2. Attempting to discover and download multiple trajectories...")
    download_multiple_trajectories()
    
    print("\nExample completed!")
    print("\nTo use the command-line interface directly:")
    print("python bridge_dataset_downloader.py --single-trajectory 'datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj6'") 