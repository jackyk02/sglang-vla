import requests
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime
from token2action import TokenToAction

# Configuration
META_ACTIONS = [
    "move down",
    "move left", 
    "move right",
    "move up",
    "move forward",
    "move backward",
    "move up, open gripper",
    "move forward right",
    "move up, close gripper",
    "move backward left"
]

SAMPLES_PER_META_ACTION = 10
TEMPERATURE = 1.0
IMAGE_PATH = "/root/sglang-vla/traces/processed_images/processed_image.jpg"
OUTPUT_DIR = "./meta_action_analysis/"

# Action dimensions for labeling
ACTION_LABELS = ['delta_x', 'delta_y', 'delta_z', 'delta_roll', 'delta_pitch', 'delta_yaw', 'gripper']

def create_meta_action_prompt(meta_action):
    """Create a prompt with the specified meta action."""
    return ("A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            f"USER: What action should the robot take? ASSISTANT: TASK: Robot manipulation task. "
            f"PLAN: Execute the specified movement. SUBTASK REASONING: Need to perform the movement. "
            f"SUBTASK: {meta_action.capitalize()}. MOVE REASONING: Executing {meta_action}. "
            f"MOVE: {meta_action.capitalize()}.")

def make_inference_request(prompt, image_path, temperature, num_samples):
    """Make inference request for multiple samples."""
    prompt_batch = [prompt] * num_samples
    image_batch = [os.path.abspath(image_path)] * num_samples
    
    response = requests.post(
        "http://localhost:30000/generate",
        json={
            "text": prompt_batch,
            "image_data": image_batch,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": 2048,
            },
        },
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: HTTP {response.status_code}")
        print(response.text)
        return None

def extract_unnormalized_actions(results):
    """Extract unnormalized actions from inference results."""
    token_to_action = TokenToAction()
    unnormalized_actions = []
    
    if not results:
        return unnormalized_actions
    
    results_list = results if isinstance(results, list) else [results]
    
    for i, result in enumerate(results_list):
        if 'meta_info' in result and 'output_ids' in result['meta_info']:
            output_ids = result['meta_info']['output_ids']
            if len(output_ids) >= 8:  # Ensure we have at least 8 tokens (7 action + 1 final)
                action_tokens = output_ids[-8:-1]  # Get 7 tokens before the last one
                try:
                    action_result = token_to_action.convert(action_tokens)
                    unnormalized_actions.append(action_result['unnormalized_actions'])
                    print(f"    Sample {i+1}: Successfully extracted unnormalized actions")
                except Exception as e:
                    print(f"    Warning: Failed to convert action tokens for sample {i+1}: {str(e)}")
            else:
                print(f"    Warning: Not enough tokens for action extraction in sample {i+1}")
        else:
            print(f"    Warning: No output_ids found in sample {i+1}")
    
    return unnormalized_actions

def collect_all_meta_action_data():
    """Collect unnormalized actions for all meta actions."""
    all_data = {}
    
    print(f"Collecting data for {len(META_ACTIONS)} meta actions...")
    print(f"Sampling {SAMPLES_PER_META_ACTION} actions per meta action with temperature {TEMPERATURE}")
    
    for meta_action in META_ACTIONS:
        print(f"\nProcessing meta action: '{meta_action}'")
        
        # Create prompt
        prompt = create_meta_action_prompt(meta_action)
        print(f"  Prompt: {prompt}")
        
        # Make inference request
        results = make_inference_request(
            prompt=prompt,
            image_path=IMAGE_PATH,
            temperature=TEMPERATURE,
            num_samples=SAMPLES_PER_META_ACTION
        )
        
        # Extract unnormalized actions
        unnormalized_actions = extract_unnormalized_actions(results)
        
        # Store data
        all_data[meta_action] = {
            'prompt': prompt,
            'unnormalized_actions': unnormalized_actions,
            'num_samples': len(unnormalized_actions)
        }
        
        print(f"  Collected {len(unnormalized_actions)} valid action samples")
    
    return all_data

def prepare_data_for_analysis(all_data):
    """Prepare data for clustering and visualization."""
    actions_list = []
    meta_action_labels = []
    
    for meta_action, data in all_data.items():
        for action in data['unnormalized_actions']:
            actions_list.append(action)
            meta_action_labels.append(meta_action)
    
    # Convert to numpy array
    actions_array = np.array(actions_list)
    
    # Apply threshold to gripper dimension (index 6) to make it binary
    if len(actions_array) > 0 and actions_array.shape[1] > 6:
        gripper_values = actions_array[:, 6]
        print(f"\nApplying gripper threshold (0.5):")
        print(f"  Original gripper range: [{np.min(gripper_values):.3f}, {np.max(gripper_values):.3f}]")
        
        # Apply threshold: values >= 0.5 become 1, values < 0.5 become 0
        actions_array[:, 6] = (gripper_values >= 0.5).astype(float)
        
        print(f"  Thresholded gripper values: {np.unique(actions_array[:, 6])}")
        print(f"  Gripper=1 count: {np.sum(actions_array[:, 6] == 1)}")
        print(f"  Gripper=0 count: {np.sum(actions_array[:, 6] == 0)}")
    
    print(f"\nPrepared data for analysis:")
    print(f"  Total samples: {len(actions_list)}")
    print(f"  Action dimensions: {actions_array.shape[1] if len(actions_array) > 0 else 0}")
    print(f"  Meta actions: {len(set(meta_action_labels))}")
    
    return actions_array, meta_action_labels

def perform_clustering(actions_array, n_clusters=5):
    """Perform K-means clustering on the action data."""
    if len(actions_array) == 0:
        print("Warning: No action data available for clustering")
        return None, None
    
    # Standardize the data
    scaler = StandardScaler()
    actions_scaled = scaler.fit_transform(actions_array)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(actions_scaled)
    
    print(f"\nClustering results:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    
    # Print cluster centers in original scale
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    print(f"  Cluster centers (unnormalized):")
    for i, center in enumerate(cluster_centers):
        print(f"    Cluster {i}: {center}")
    
    return cluster_labels, scaler

def create_visualizations(actions_array, meta_action_labels, cluster_labels, output_dir):
    """Create comprehensive visualizations of the analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    if len(actions_array) == 0:
        print("Warning: No action data available for visualization")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Action distributions by meta action
    plt.figure(figsize=(20, 12))
    
    for i, action_label in enumerate(ACTION_LABELS):
        plt.subplot(3, 3, i+1)
        
        # Create data for violin plot
        plot_data = []
        plot_labels = []
        
        for meta_action in META_ACTIONS:
            meta_indices = [j for j, label in enumerate(meta_action_labels) if label == meta_action]
            if meta_indices:
                meta_actions_data = actions_array[meta_indices, i]
                plot_data.extend(meta_actions_data)
                plot_labels.extend([meta_action] * len(meta_actions_data))
        
        if plot_data:
            df = pd.DataFrame({'action_value': plot_data, 'meta_action': plot_labels})
            sns.violinplot(data=df, x='meta_action', y='action_value')
            plt.xticks(rotation=45, ha='right')
            plt.title(f'{action_label} Distribution by Meta Action')
            plt.ylabel(f'{action_label}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_distributions_by_meta_action.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. PCA visualization
    if len(actions_array) > 2:
        plt.figure(figsize=(15, 5))
        
        # Standardize for PCA
        scaler = StandardScaler()
        actions_scaled = scaler.fit_transform(actions_array)
        
        # 2D PCA
        pca_2d = PCA(n_components=2)
        actions_pca_2d = pca_2d.fit_transform(actions_scaled)
        
        plt.subplot(1, 3, 1)
        colors = plt.cm.tab10(np.linspace(0, 1, len(META_ACTIONS)))
        for i, meta_action in enumerate(META_ACTIONS):
            indices = [j for j, label in enumerate(meta_action_labels) if label == meta_action]
            if indices:
                plt.scatter(actions_pca_2d[indices, 0], actions_pca_2d[indices, 1], 
                          c=[colors[i]], label=meta_action, alpha=0.7)
        
        plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA: Actions by Meta Action')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2D PCA with clusters
        plt.subplot(1, 3, 2)
        if cluster_labels is not None:
            scatter = plt.scatter(actions_pca_2d[:, 0], actions_pca_2d[:, 1], 
                                c=cluster_labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter)
        plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA: Actions by Cluster')
        
        # Explained variance
        plt.subplot(1, 3, 3)
        pca_full = PCA()
        pca_full.fit(actions_scaled)
        plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
                np.cumsum(pca_full.explained_variance_ratio_), 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pca_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Correlation heatmap between action dimensions
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(actions_array.T)
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                xticklabels=ACTION_LABELS, 
                yticklabels=ACTION_LABELS,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True)
    plt.title('Correlation Matrix of Action Dimensions')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_correlation_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Meta action vs cluster assignment
    if cluster_labels is not None:
        plt.figure(figsize=(12, 8))
        
        # Create contingency table
        meta_action_list = list(META_ACTIONS)
        cluster_list = list(range(max(cluster_labels) + 1))
        
        contingency = np.zeros((len(meta_action_list), len(cluster_list)))
        
        for i, meta_action in enumerate(meta_action_list):
            indices = [j for j, label in enumerate(meta_action_labels) if label == meta_action]
            if indices:
                for cluster in cluster_list:
                    count = sum(1 for idx in indices if cluster_labels[idx] == cluster)
                    contingency[i, cluster] = count
        
        # Create heatmap
        sns.heatmap(contingency, 
                    xticklabels=[f'Cluster {i}' for i in cluster_list],
                    yticklabels=meta_action_list,
                    annot=True, 
                    fmt='g',
                    cmap='Blues')
        plt.title('Meta Action vs Cluster Assignment')
        plt.xlabel('Cluster')
        plt.ylabel('Meta Action')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'meta_action_cluster_heatmap.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nVisualization plots saved to: {output_dir}")

def save_analysis_results(all_data, actions_array, meta_action_labels, cluster_labels, output_dir):
    """Save detailed analysis results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw data
    results = {
        'timestamp': datetime.now().isoformat(),
        'meta_actions': META_ACTIONS,
        'samples_per_meta_action': SAMPLES_PER_META_ACTION,
        'temperature': TEMPERATURE,
        'action_labels': ACTION_LABELS,
        'data': all_data
    }
    
    with open(os.path.join(output_dir, 'raw_data.json'), 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    # Save processed data for analysis
    if len(actions_array) > 0:
        processed_data = {
            'actions_array': actions_array.tolist(),
            'meta_action_labels': meta_action_labels,
            'cluster_labels': cluster_labels.tolist() if cluster_labels is not None else None,
            'action_statistics': {}
        }
        
        # Calculate statistics for each meta action
        for meta_action in META_ACTIONS:
            indices = [i for i, label in enumerate(meta_action_labels) if label == meta_action]
            if indices:
                meta_actions_data = actions_array[indices]
                processed_data['action_statistics'][meta_action] = {
                    'mean': np.mean(meta_actions_data, axis=0).tolist(),
                    'std': np.std(meta_actions_data, axis=0).tolist(),
                    'min': np.min(meta_actions_data, axis=0).tolist(),
                    'max': np.max(meta_actions_data, axis=0).tolist(),
                    'sample_count': len(indices)
                }
        
        with open(os.path.join(output_dir, 'processed_data.json'), 'w') as f:
            json.dump(processed_data, f, indent=2)
    
    # Save summary statistics
    summary = {
        'total_samples_collected': len(actions_array),
        'meta_actions_tested': len(META_ACTIONS),
        'successful_meta_actions': len(set(meta_action_labels)),
        'action_dimensions': len(ACTION_LABELS)
    }
    
    if len(actions_array) > 0:
        summary.update({
            'overall_action_stats': {
                'mean': np.mean(actions_array, axis=0).tolist(),
                'std': np.std(actions_array, axis=0).tolist()
            }
        })
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Analysis results saved to: {output_dir}")

def main():
    """Main analysis pipeline."""
    print("="*60)
    print("META ACTION vs UNNORMALIZED ACTION CORRELATION ANALYSIS")
    print("="*60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Collect data for all meta actions
    print("\nStep 1: Collecting data for all meta actions...")
    all_data = collect_all_meta_action_data()
    
    # Step 2: Prepare data for analysis
    print("\nStep 2: Preparing data for analysis...")
    actions_array, meta_action_labels = prepare_data_for_analysis(all_data)
    
    if len(actions_array) == 0:
        print("Error: No valid action data collected. Please check the inference server and try again.")
        return
    
    # Step 3: Perform clustering
    print("\nStep 3: Performing clustering analysis...")
    cluster_labels, scaler = perform_clustering(actions_array, n_clusters=min(5, len(META_ACTIONS)))
    
    # Step 4: Create visualizations
    print("\nStep 4: Creating visualizations...")
    create_visualizations(actions_array, meta_action_labels, cluster_labels, OUTPUT_DIR)
    
    # Step 5: Save results
    print("\nStep 5: Saving analysis results...")
    save_analysis_results(all_data, actions_array, meta_action_labels, cluster_labels, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Total samples analyzed: {len(actions_array)}")
    print(f"Meta actions tested: {len(set(meta_action_labels))}")
    print("Check the generated plots and JSON files for detailed results.")

if __name__ == "__main__":
    main() 