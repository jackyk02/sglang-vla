import numpy as np
import sglang as sgl
from token2action import TokenToAction, image_qa
import time
import pandas as pd
import json
import os
from typing import List, Dict
import matplotlib.pyplot as plt
import random

def generate_random_instruction() -> str:
    """Generate a random robot instruction."""
    actions = [
        "grab the red block",
        "pick up the blue cube",
        "move the green object",
        "push the yellow block",
        "lift the white cube",
        "slide the orange block",
        "stack the purple cube",
        "place the block on top",
        "rotate the cube",
        "align the blocks",
        "group similar blocks",
        "separate the blocks",
        "arrange blocks in order",
        "remove the top block",
        "combine the blocks"
    ]
    return random.choice(actions)

def perform_warm_up(num_warm_up: int = 3, batch_size: int = 1) -> None:
    """
    Perform warm-up runs to stabilize the system.
    
    Args:
        num_warm_up: Number of warm-up iterations
        batch_size: Batch size for warm-up runs
    """
    print(f"Performing {num_warm_up} warm-up runs...")
    converter = TokenToAction()
    
    for i in range(num_warm_up):
        arguments = [
            {
                "image_path": "images/robot.jpg",
                "question": f"In: What action should the robot take to {generate_random_instruction()}?\nOut:",
            }
        ] * batch_size
        
        states = image_qa.run_batch(
            arguments,
            max_new_tokens=7,
            temperature=0.0
        )
        _ = [converter.convert(s.get_meta_info("action")["output_ids"]).tolist() for s in states]
    
    print("Warm-up completed")

def measure_batch_latency(batch_sizes: List[int], temperatures: List[float], num_trials: int = 3) -> pd.DataFrame:
    """
    Measure latency for different batch sizes and temperatures.
    
    Args:
        batch_sizes: List of batch sizes to test
        temperatures: List of temperatures to test
        num_trials: Number of trials for each configuration
    
    Returns:
        DataFrame with latency measurements
    """
    results = []
    converter = TokenToAction()
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        for temp in temperatures:
            print(f"Temperature: {temp}")
            for trial in range(num_trials):
                print(f"Trial {trial + 1}/{num_trials}", end='\r')
                
                # Generate random instructions for each item in the batch
                arguments = [
                    {
                        "image_path": "images/robot.jpg",
                        "question": f"In: What action should the robot take to {generate_random_instruction()}?\nOut:",
                    }
                    for _ in range(batch_size)
                ]
                
                # Measure time
                start_time = time.perf_counter()
                states = image_qa.run_batch(
                    arguments,
                    max_new_tokens=7,
                    temperature=temp
                )
                actions = [converter.convert(s.get_meta_info("action")["output_ids"]).tolist() for s in states]
                end_time = time.perf_counter()
                
                latency = end_time - start_time
                throughput = batch_size / latency  # requests per second
                
                results.append({
                    'batch_size': batch_size,
                    'temperature': temp,
                    'trial': trial,
                    'latency': latency,
                    'throughput': throughput
                })
    
    return pd.DataFrame(results)

def plot_results(df: pd.DataFrame) -> None:
    """Plot latency and throughput results."""
    # Calculate mean and std for each batch size
    summary = df.groupby('batch_size').agg({
        'latency': ['mean', 'std'],
        'throughput': ['mean', 'std']
    }).reset_index()
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot latency
    ax1.errorbar(summary['batch_size'], 
                summary['latency']['mean'], 
                yerr=summary['latency']['std'],
                marker='o')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Latency (seconds)')
    ax1.set_title('Latency vs Batch Size')
    ax1.grid(True)
    
    # Plot throughput
    ax2.errorbar(summary['batch_size'], 
                summary['throughput']['mean'], 
                yerr=summary['throughput']['std'],
                marker='o')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Throughput (requests/second)')
    ax2.set_title('Throughput vs Batch Size')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Set random seed for reproducibility
    random.seed(42)
    
    # Connect to the server
    sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
    
    # Define test parameters
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    temperatures = [1.0]
    num_trials = 10
    num_warm_up = 10  # Number of warm-up runs
    
    # Perform warm-up runs
    perform_warm_up(num_warm_up, batch_size=1)
    
    # Run measurements
    results_df = measure_batch_latency(batch_sizes, temperatures, num_trials)
    
    # Save results
    results_df.to_csv('batch_latency_results.csv', index=False)
    
    # Plot results
    plot_results(results_df)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(results_df.groupby('batch_size').agg({
        'latency': ['mean', 'std'],
        'throughput': ['mean', 'std']
    }))