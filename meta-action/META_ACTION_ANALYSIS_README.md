# Meta Action Analysis: Correlation Between Meta Actions and Unnormalized Action Tokens

This analysis investigates the correlation between generated meta actions (e.g., "MOVE: Move right.") and the downstream UNNORMALIZED action tokens in a 7D robotic action space.

## Overview

The 7D action space represents:
- **delta_x**: Change in X position
- **delta_y**: Change in Y position  
- **delta_z**: Change in Z position
- **delta_roll**: Change in roll rotation
- **delta_pitch**: Change in pitch rotation
- **delta_yaw**: Change in yaw rotation
- **gripper**: Gripper state (open/close)

## Meta Actions Analyzed

The script analyzes the following meta actions:
1. "move down"
2. "move left"
3. "move right"
4. "move up"
5. "move forward"
6. "move backward"
7. "move up, open gripper"
8. "move forward right"
9. "move up, close gripper"
10. "move backward left"

## Methodology

1. **Data Collection**: For each meta action, 10 samples are generated using temperature 1.0
2. **Action Extraction**: Unnormalized actions are extracted from the output tokens
3. **Statistical Analysis**: Mean, standard deviation, min/max calculated for each meta action
4. **Clustering**: K-means clustering applied to identify action patterns
5. **Visualization**: Multiple plots generated to visualize correlations and clusters

## Usage

### Prerequisites
- SGLang inference server running on localhost:30000
- OpenVLA model loaded
- Required Python packages (see requirements_analysis.txt)

### Running the Analysis

```bash
# Method 1: Use the runner script (recommended)
python run_meta_action_analysis.py

# Method 2: Run directly
python meta_action_analysis.py
```

### Installing Dependencies Only
```bash
pip install -r requirements_analysis.txt
```

## Output Files

The analysis generates the following files in `./meta_action_analysis/`:

### Visualizations
- `action_distributions_by_meta_action.png`: Violin plots showing action value distributions for each meta action
- `pca_analysis.png`: PCA visualization with actions colored by meta action and clusters
- `action_correlation_heatmap.png`: Correlation matrix between action dimensions
- `meta_action_cluster_heatmap.png`: Heatmap showing meta action vs cluster assignments

### Data Files
- `raw_data.json`: Complete raw data including prompts and all generated actions
- `processed_data.json`: Processed data with statistics and cluster assignments
- `summary.json`: Summary statistics and overall analysis results

## Interpretation

### Action Distributions
The violin plots show how each action dimension varies for different meta actions. Look for:
- **Consistent patterns**: Meta actions with similar meanings should show similar distributions
- **Distinctive patterns**: Different meta actions should have distinguishable action profiles
- **Expected correlations**: "move right" should show positive delta_x, "move up" should show positive delta_z, etc.

### PCA Analysis
The PCA plots reveal:
- **Clustering patterns**: How meta actions group together in the reduced action space
- **Variance explained**: How much of the action variation is captured by the first few components
- **Separation quality**: How well different meta actions separate in the action space

### Correlation Heatmap
Shows correlations between different action dimensions:
- **Strong correlations**: May indicate coupled movements
- **Weak correlations**: Independent action dimensions
- **Unexpected correlations**: May reveal model biases or patterns

### Cluster Analysis
K-means clustering reveals:
- **Natural groupings**: Meta actions that result in similar action patterns
- **Cluster centers**: Representative actions for each cluster
- **Assignment patterns**: Which meta actions belong to which clusters

## Expected Results

For a well-trained model, you should observe:

1. **Directional Consistency**: 
   - "move right" → positive delta_x
   - "move left" → negative delta_x
   - "move up" → positive delta_z
   - "move down" → negative delta_z
   - "move forward" → positive delta_y
   - "move backward" → negative delta_y

2. **Gripper Consistency**:
   - "open gripper" actions → gripper values indicating open state
   - "close gripper" actions → gripper values indicating closed state

3. **Compound Actions**:
   - "move forward right" → positive delta_x AND positive delta_y
   - "move up, open gripper" → positive delta_z AND gripper open state

4. **Clustering**:
   - Similar directional actions should cluster together
   - Gripper actions should form distinct clusters
   - Compound actions may form intermediate clusters

## Troubleshooting

### Common Issues

1. **Inference Server Not Running**:
   ```bash
   python -m sglang.launch_server --model-path openvla/openvla-7b --port 30000
   ```

2. **Missing Dependencies**:
   ```bash
   pip install -r requirements_analysis.txt
   ```

3. **No Action Data Collected**:
   - Check if the inference server is responding
   - Verify the image path exists
   - Check the token extraction logic

4. **Visualization Errors**:
   - Ensure matplotlib backend is properly configured
   - Check if output directory is writable
   - Verify all required packages are installed

### Performance Notes

- The analysis generates 100 inference requests (10 per meta action)
- With temperature 1.0, expect some variability in results
- Runtime depends on model inference speed (typically 5-15 minutes)

## Extending the Analysis

You can modify the script to:
- Add more meta actions to `META_ACTIONS` list
- Change the number of samples per meta action (`SAMPLES_PER_META_ACTION`)
- Adjust temperature settings
- Add different clustering algorithms
- Include additional visualization methods
- Analyze normalized vs unnormalized actions
- Add statistical significance testing 