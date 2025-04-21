import requests
import numpy as np
import json_numpy as json
import time
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from typing import List, Dict

def generate_random_instruction() -> str:
    """Generate a random robot instruction."""
    actions = [
        "move the yellow knife to the right of the pan",
        "place the red cup on the table",
        "grab the blue fork",
        "push the green plate forward",
        "lift the white bowl",
        "slide the orange spatula",
        "stack the purple containers",
        "arrange the utensils in order",
        "rotate the cutting board",
        "align the silverware",
        "group similar kitchen items",
        "separate the cooking tools",
        "remove the top dish",
        "combine the cooking ingredients"
    ]
    return random.choice(actions)

def generate_random_action(batch_size: int = 8) -> np.ndarray:
    """
    Generate a random action matrix of shape (batch_size, 7) similar to the example.
    
    Args:
        batch_size: Number of rows in the action matrix
        
    Returns:
        Numpy array of shape (batch_size, 7)
    """
    # Create a batch_size x 7 matrix with values in the range of the example
    action = np.random.randint(31800, 31960, size=(batch_size, 7))
    # Set the last column to 31744 to match the pattern in the example
    action[:, -1] = 31744
    return action

def perform_warm_up(url: str, num_warm_up: int = 3) -> None:
    """
    Perform warm-up runs to stabilize the system.
    
    Args:
        url: API endpoint URL
        num_warm_up: Number of warm-up iterations
    """
    print(f"Performing {num_warm_up} warm-up runs...")
    
    for i in range(num_warm_up):
        instruction = generate_random_instruction()
        action = generate_random_action()
        
        payload = {
            "instruction": instruction,
            "image_path": "/root/V-GPS/scripts/79000.jpg",
            "action": action
        }
        
        try:
            response = requests.post(url, data=json.dumps(payload))
            response = json.loads(response.text)
            _ = response["rewards"]
        except Exception as e:
            print(f"Warm-up error: {e}")
    
    print("Warm-up completed")

def measure_batch_latency(url: str, batch_sizes: List[int], num_trials: int = 3) -> pd.DataFrame:
    """
    Measure latency for different batch sizes.
    
    Args:
        url: API endpoint URL
        batch_sizes: List of batch sizes to test
        num_trials: Number of trials for each configuration
    
    Returns:
        DataFrame with latency measurements
    """
    results = []
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        for trial in range(num_trials):
            print(f"Trial {trial + 1}/{num_trials}", end='\r')
            
            # Generate action matrix for the batch
            action = generate_random_action(batch_size)
            instruction = generate_random_instruction()
            
            # Create payload
            payload = {
                "instruction": instruction,
                "image_path": "/root/V-GPS/scripts/79000.jpg",
                "action": action
            }
            
            # Measure time
            start_time = time.perf_counter()
            
            try:
                response = requests.post(url, data=json.dumps(payload))
                response_data = json.loads(response.text)
                rewards = response_data["rewards"]
            except Exception as e:
                print(f"Error in batch processing: {e}")
                continue
            
            end_time = time.perf_counter()
            
            latency = end_time - start_time
            throughput = batch_size / latency  # actions per second
            
            results.append({
                'batch_size': batch_size,
                'trial': trial,
                'latency': latency,
                'throughput': throughput,
                'avg_latency_per_action': latency / batch_size
            })
    
    return pd.DataFrame(results)

def plot_results(df: pd.DataFrame, output_dir: str = './') -> None:
    """
    Plot latency and throughput results.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save the plots
    """
    # Calculate mean and std for each batch size
    summary = df.groupby('batch_size').agg({
        'latency': ['mean', 'std'],
        'throughput': ['mean', 'std'],
        'avg_latency_per_action': ['mean', 'std']
    }).reset_index()
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot total latency
    ax1.errorbar(summary['batch_size'], 
                summary['latency']['mean'], 
                yerr=summary['latency']['std'],
                marker='o')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Total Latency (seconds)')
    ax1.set_title('Reward Model Total Latency vs Batch Size')
    ax1.grid(True)
    
    # Plot throughput
    ax2.errorbar(summary['batch_size'], 
                summary['throughput']['mean'], 
                yerr=summary['throughput']['std'],
                marker='o')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Throughput (actions/second)')
    ax2.set_title('Reward Model Throughput vs Batch Size')
    ax2.grid(True)
    
    # Plot average latency per action
    ax3.errorbar(summary['batch_size'], 
                summary['avg_latency_per_action']['mean'], 
                yerr=summary['avg_latency_per_action']['std'],
                marker='o')
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Avg Latency per Action (seconds)')
    ax3.set_title('Avg Latency per Action vs Batch Size')
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'reward_model_performance.png'))
    plt.show()

if __name__ == '__main__':
    # Set random seed for reproducibility
    random.seed(42)
    
    # API endpoint
    url = "http://127.0.0.1:3100/process"
    
    # Define test parameters
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    num_trials = 10
    num_warm_up = 10  # Number of warm-up runs
    output_dir = './reward_model_results'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform warm-up runs
    perform_warm_up(url, num_warm_up)
    
    # Run batch measurements
    print("\n===== Testing Batch Processing =====")
    results_df = measure_batch_latency(url, batch_sizes, num_trials)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'reward_model_batch_results.csv'), index=False)
    
    # Plot results
    plot_results(results_df, output_dir)
    
    # Print summary statistics
    print("\nBatch Processing Summary:")
    print(results_df.groupby('batch_size').agg({
        'latency': ['mean', 'std'],
        'throughput': ['mean', 'std'],
        'avg_latency_per_action': ['mean', 'std']
    }))