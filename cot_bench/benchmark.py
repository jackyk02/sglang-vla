import numpy as np
import sglang as sgl
from token2action import TokenToAction, image_qa
import pandas as pd
import json
import os
import time
from typing import List, Dict
import statistics

converter = TokenToAction()

def batch(batch_size: int, temp: float) -> List[Dict]:
    INSTRUCTION = "place the watermelon on the towel"
    SYSTEM_PROMPT = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    
    def get_openvla_prompt(instruction: str) -> str:
        return f"USER: What action should the robot take to {instruction.lower()}? ASSISTANT: TASK:"
    
    prompt = get_openvla_prompt(INSTRUCTION)
    arguments = [
        {
            "image_path": "images/test_obs.jpg",
            "question": prompt,
        }
    ] * batch_size
    
    states = image_qa.run_batch(
        arguments,
        max_new_tokens=1024,
        temperature=temp
    )
    return [{'cot': s.text(), 'action': s.get_meta_info("action")["output_ids"][-8:-1]} for s in states]

def run_benchmark(batch_sizes: List[int], n_runs: int = 5, n_warmup: int = 2, temperature: float = 0.2) -> pd.DataFrame:
    """
    Run benchmark across different batch sizes with warm-up runs.
    
    Args:
        batch_sizes: List of batch sizes to test
        n_runs: Number of runs for each batch size
        n_warmup: Number of warm-up runs
        temperature: Temperature for generation
        
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch size: {batch_size}")
        
        # Warm-up runs
        print(f"Performing {n_warmup} warm-up runs...")
        for _ in range(n_warmup):
            _ = batch(batch_size, temperature)
        
        # Actual benchmark runs
        times = []
        for run in range(n_runs):
            print(f"Run {run + 1}/{n_runs}")
            start_time = time.time()
            _ = batch(batch_size, temperature)
            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)
        
        # Calculate statistics
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        throughput = batch_size / mean_time  # samples per second
        
        results.append({
            'batch_size': batch_size,
            'mean_time': mean_time,
            'std_time': std_time,
            'throughput': throughput,
            'samples_per_second': throughput
        })
    
    return pd.DataFrame(results)

if __name__ == '__main__':
    # Set up SGLang backend
    sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
    
    # Define benchmark parameters
    batch_sizes = [1, 4, 8, 16, 32, 64, 128]
    n_runs = 5
    n_warmup = 2
    temperature = 0.2
    
    # Run benchmark
    print("Starting benchmark...")
    results_df = run_benchmark(
        batch_sizes=batch_sizes,
        n_runs=n_runs,
        n_warmup=n_warmup,
        temperature=temperature
    )
    
    # Save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_df.to_csv(f'benchmark_results_{timestamp}.csv', index=False)
    
    # Print results
    print("\nBenchmark Results:")
    print(results_df.to_string(float_format=lambda x: f"{x:.3f}"))
    
    # Optional: Create a simple visualization
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        # Throughput plot
        plt.subplot(1, 2, 1)
        plt.plot(results_df['batch_size'], results_df['samples_per_second'], 'o-')
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (samples/second)')
        plt.title('Throughput vs Batch Size')
        plt.grid(True)
        
        # Latency plot
        plt.subplot(1, 2, 2)
        plt.errorbar(results_df['batch_size'], 
                    results_df['mean_time'], 
                    yerr=results_df['std_time'], 
                    fmt='o-',
                    capsize=5)
        plt.xlabel('Batch Size')
        plt.ylabel('Latency (seconds)')
        plt.title('Latency vs Batch Size')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'benchmark_plot_{timestamp}.png')
        
    except ImportError:
        print("matplotlib not installed - skipping visualization")