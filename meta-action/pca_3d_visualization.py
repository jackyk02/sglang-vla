import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import os

# Configuration
DATA_DIR = "./meta_action_analysis/"
OUTPUT_DIR = "./meta_action_analysis/3d_pca/"
ACTION_LABELS = ['delta_x', 'delta_y', 'delta_z', 'delta_roll', 'delta_pitch', 'delta_yaw', 'gripper']

def load_processed_data():
    """Load processed data from the main analysis."""
    processed_data_path = os.path.join(DATA_DIR, "processed_data.json")
    
    if not os.path.exists(processed_data_path):
        print(f"Error: Processed data file not found at {processed_data_path}")
        print("Please run the main meta_action_analysis.py script first.")
        return None, None, None
    
    with open(processed_data_path, 'r') as f:
        data = json.load(f)
    
    actions_array = np.array(data['actions_array'])
    meta_action_labels = data['meta_action_labels']
    cluster_labels = np.array(data['cluster_labels']) if data['cluster_labels'] else None
    
    print(f"Loaded data:")
    print(f"  Actions shape: {actions_array.shape}")
    print(f"  Number of meta action labels: {len(meta_action_labels)}")
    print(f"  Unique meta actions: {len(set(meta_action_labels))}")
    
    return actions_array, meta_action_labels, cluster_labels

def perform_3d_pca(actions_array):
    """Perform 3D PCA on the action data."""
    if len(actions_array) < 3:
        print("Error: Need at least 3 samples for 3D PCA")
        return None, None
    
    # Standardize the data
    scaler = StandardScaler()
    actions_scaled = scaler.fit_transform(actions_array)
    
    # Perform 3D PCA
    pca_3d = PCA(n_components=3)
    actions_pca_3d = pca_3d.fit_transform(actions_scaled)
    
    print(f"\n3D PCA Results:")
    print(f"  PC1 explains {pca_3d.explained_variance_ratio_[0]:.2%} of variance")
    print(f"  PC2 explains {pca_3d.explained_variance_ratio_[1]:.2%} of variance")
    print(f"  PC3 explains {pca_3d.explained_variance_ratio_[2]:.2%} of variance")
    print(f"  Total variance explained: {sum(pca_3d.explained_variance_ratio_):.2%}")
    
    # Print component loadings
    print(f"\nPCA Component Loadings:")
    for i, component in enumerate(pca_3d.components_):
        print(f"  PC{i+1}:")
        for j, loading in enumerate(component):
            print(f"    {ACTION_LABELS[j]}: {loading:.3f}")
    
    return actions_pca_3d, pca_3d

def create_matplotlib_3d_plots(actions_pca_3d, meta_action_labels, cluster_labels, pca_3d, output_dir):
    """Create 3D PCA plots using matplotlib."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique meta actions and assign colors
    unique_meta_actions = list(set(meta_action_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_meta_actions)))
    
    # 3D PCA colored by meta action
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, meta_action in enumerate(unique_meta_actions):
        indices = [j for j, label in enumerate(meta_action_labels) if label == meta_action]
        if indices:
            ax.scatter(actions_pca_3d[indices, 0], 
                      actions_pca_3d[indices, 1], 
                      actions_pca_3d[indices, 2],
                      c=[colors[i]], 
                      label=meta_action, 
                      alpha=0.7,
                      s=50)
    
    ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})')
    ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})')
    ax.set_title('3D PCA: Meta Actions')
    
    # Position legend to avoid blocking PC3 label
    ax.legend(bbox_to_anchor=(1.15, 0.8), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3d_pca_matplotlib.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_interactive_3d_plots(actions_pca_3d, meta_action_labels, cluster_labels, pca_3d, output_dir):
    """Create interactive 3D PCA plots using plotly."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotly
    df_data = {
        'PC1': actions_pca_3d[:, 0],
        'PC2': actions_pca_3d[:, 1],
        'PC3': actions_pca_3d[:, 2],
        'Meta_Action': meta_action_labels,
        'Sample_Index': range(len(meta_action_labels))
    }
    
    if cluster_labels is not None:
        df_data['Cluster'] = cluster_labels
    
    # 1. Interactive plot colored by meta action
    fig1 = px.scatter_3d(df_data, 
                        x='PC1', y='PC2', z='PC3',
                        color='Meta_Action',
                        title='Interactive 3D PCA: Meta Actions',
                        hover_data=['Sample_Index'],
                        labels={
                            'PC1': f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})',
                            'PC2': f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})',
                            'PC3': f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})'
                        })
    
    fig1.update_traces(marker=dict(size=5))
    fig1.update_layout(height=700)
    
    # Save as HTML
    pyo.plot(fig1, filename=os.path.join(output_dir, '3d_pca_meta_actions.html'), auto_open=False)
    
    # 2. Interactive plot colored by cluster (if available)
    if cluster_labels is not None:
        fig2 = px.scatter_3d(df_data, 
                            x='PC1', y='PC2', z='PC3',
                            color='Cluster',
                            title='Interactive 3D PCA: Clusters',
                            hover_data=['Sample_Index', 'Meta_Action'],
                            labels={
                                'PC1': f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})',
                                'PC2': f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})',
                                'PC3': f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})'
                            })
        
        fig2.update_traces(marker=dict(size=5))
        fig2.update_layout(height=700)
        
        # Save as HTML
        pyo.plot(fig2, filename=os.path.join(output_dir, '3d_pca_clusters.html'), auto_open=False)
    
    # 3. Component loadings visualization
    fig3 = go.Figure()
    
    # Add bars for each component
    for i, component_name in enumerate(['PC1', 'PC2', 'PC3']):
        fig3.add_trace(go.Bar(
            name=component_name,
            x=ACTION_LABELS,
            y=pca_3d.components_[i],
            opacity=0.8
        ))
    
    fig3.update_layout(
        title='PCA Component Loadings',
        xaxis_title='Action Dimensions',
        yaxis_title='Loading Value',
        barmode='group',
        height=500
    )
    
    # Save as HTML
    pyo.plot(fig3, filename=os.path.join(output_dir, '3d_pca_loadings.html'), auto_open=False)

def create_projection_analysis(actions_pca_3d, meta_action_labels, pca_3d, output_dir):
    """Create 2D projections of the 3D PCA for detailed analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    unique_meta_actions = list(set(meta_action_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_meta_actions)))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    projections = [
        (0, 1, 'PC1', 'PC2'),
        (0, 2, 'PC1', 'PC3'),
        (1, 2, 'PC2', 'PC3')
    ]
    
    for idx, (x_idx, y_idx, x_label, y_label) in enumerate(projections):
        ax = axes[idx]
        
        for i, meta_action in enumerate(unique_meta_actions):
            indices = [j for j, label in enumerate(meta_action_labels) if label == meta_action]
            if indices:
                ax.scatter(actions_pca_3d[indices, x_idx], 
                          actions_pca_3d[indices, y_idx],
                          c=[colors[i]], 
                          label=meta_action, 
                          alpha=0.7,
                          s=50)
        
        ax.set_xlabel(f'{x_label} ({pca_3d.explained_variance_ratio_[x_idx]:.1%})')
        ax.set_ylabel(f'{y_label} ({pca_3d.explained_variance_ratio_[y_idx]:.1%})')
        ax.set_title(f'{x_label} vs {y_label}')
        ax.grid(True, alpha=0.3)
        
        if idx == 0:  # Only show legend on first plot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3d_pca_projections.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_3d_pca_results(actions_pca_3d, pca_3d, output_dir):
    """Save 3D PCA results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save PCA results
    pca_results = {
        'pca_3d_coordinates': actions_pca_3d.tolist(),
        'explained_variance_ratio': pca_3d.explained_variance_ratio_.tolist(),
        'components': pca_3d.components_.tolist(),
        'component_labels': ['PC1', 'PC2', 'PC3'],
        'feature_labels': ACTION_LABELS,
        'total_variance_explained': sum(pca_3d.explained_variance_ratio_)
    }
    
    with open(os.path.join(output_dir, '3d_pca_results.json'), 'w') as f:
        json.dump(pca_results, f, indent=2)
    
    # Save component interpretation
    interpretation = {
        'component_interpretation': {}
    }
    
    for i, component in enumerate(pca_3d.components_):
        # Find the features with highest absolute loadings
        abs_loadings = np.abs(component)
        top_features_idx = np.argsort(abs_loadings)[::-1][:3]  # Top 3 features
        
        interpretation['component_interpretation'][f'PC{i+1}'] = {
            'variance_explained': f"{pca_3d.explained_variance_ratio_[i]:.2%}",
            'top_features': [
                {
                    'feature': ACTION_LABELS[idx],
                    'loading': float(component[idx]),
                    'abs_loading': float(abs_loadings[idx])
                }
                for idx in top_features_idx
            ]
        }
    
    with open(os.path.join(output_dir, '3d_pca_interpretation.json'), 'w') as f:
        json.dump(interpretation, f, indent=2)

def main():
    """Main 3D PCA analysis pipeline."""
    print("="*60)
    print("3D PCA VISUALIZATION FOR META ACTION ANALYSIS")
    print("="*60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Load processed data
    print("\nStep 1: Loading processed data...")
    actions_array, meta_action_labels, cluster_labels = load_processed_data()
    
    if actions_array is None:
        return
    
    # Step 2: Perform 3D PCA
    print("\nStep 2: Performing 3D PCA...")
    actions_pca_3d, pca_3d = perform_3d_pca(actions_array)
    
    if actions_pca_3d is None:
        return
    
    # Step 3: Create matplotlib 3D plots
    print("\nStep 3: Creating matplotlib 3D visualizations...")
    create_matplotlib_3d_plots(actions_pca_3d, meta_action_labels, cluster_labels, pca_3d, OUTPUT_DIR)
    
    # Step 4: Create interactive plotly visualizations
    print("\nStep 4: Creating interactive 3D visualizations...")
    try:
        create_interactive_3d_plots(actions_pca_3d, meta_action_labels, cluster_labels, pca_3d, OUTPUT_DIR)
        print("  Interactive plots created successfully!")
    except Exception as e:
        print(f"  Warning: Could not create interactive plots: {e}")
        print("  Make sure plotly is installed: pip install plotly")
    
    # Step 5: Create 2D projections for detailed analysis
    print("\nStep 5: Creating 2D projection analysis...")
    create_projection_analysis(actions_pca_3d, meta_action_labels, pca_3d, OUTPUT_DIR)
    
    # Step 6: Save results
    print("\nStep 6: Saving 3D PCA results...")
    save_3d_pca_results(actions_pca_3d, pca_3d, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("3D PCA ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - 3d_pca_matplotlib.png: Static 3D plots")
    print("  - 3d_pca_meta_actions.html: Interactive 3D plot by meta action")
    print("  - 3d_pca_clusters.html: Interactive 3D plot by cluster")
    print("  - 3d_pca_loadings.html: Interactive component loadings")
    print("  - 3d_pca_projections.png: 2D projections")
    print("  - 3d_pca_results.json: Numerical results")
    print("  - 3d_pca_interpretation.json: Component interpretation")

if __name__ == "__main__":
    main() 