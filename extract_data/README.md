# Bridge Dataset Downloader

This Python script automatically downloads language instructions (first line of `lang.txt`) and first images (`im_0.jpg`) from Bridge dataset trajectories.

## Features

- 🚀 **Automatic Discovery**: Finds trajectories in the dataset structure
- 📥 **Batch Download**: Downloads multiple trajectories concurrently  
- 📊 **CSV Output**: Saves language instructions in organized CSV format
- 🖼️ **Image Management**: Downloads and organizes first images from each trajectory
- 📈 **Progress Tracking**: Real-time logging and summary generation
- 🔧 **Flexible Usage**: Command-line interface with multiple options

## Installation

1. Install the required dependency:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

#### Download a single trajectory:
```bash
python bridge_dataset_downloader.py --single-trajectory "datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj6"
```

#### Auto-discover and download multiple trajectories:
```bash
python bridge_dataset_downloader.py --output-dir "./my_bridge_data"
```

#### Download specific trajectories:
```bash
python bridge_dataset_downloader.py --trajectories \
  "datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj6" \
  "datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj7"
```

#### Custom configuration:
```bash
python bridge_dataset_downloader.py \
  --base-url "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2" \
  --output-dir "./bridge_data" \
  --max-workers 10
```

### Programmatic Usage

```python
from bridge_dataset_downloader import BridgeDatasetDownloader

# Initialize downloader
downloader = BridgeDatasetDownloader(
    base_url="https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2",
    output_dir="./bridge_data",
    max_workers=5
)

# Download a single trajectory
success, message = downloader.download_trajectory_data(
    "datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj6"
)

# Auto-discover and download multiple trajectories
trajectories = downloader.discover_trajectories()
downloader.download_all_trajectories(trajectories)
downloader.generate_summary()
```

## Output Structure

The script creates the following output structure:

```
output_dir/
├── language_instructions.csv    # CSV file with all language instructions
├── images/                      # Directory containing downloaded images
│   ├── datacol2_toykitchen7_drawer_pnp_01_2023-04-19_09-18-15_raw_traj_group0_traj6_im_0.jpg
│   └── ...
└── download_summary.json        # Summary of the download process
```

### CSV Format

The `language_instructions.csv` file contains:
- `trajectory_path`: Full path to the trajectory
- `language_instruction`: First line from lang.txt (the instruction)  
- `image_filename`: Corresponding image filename

Example:
```csv
trajectory_path,language_instruction,image_filename
datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj6,close the drawer,datacol2_toykitchen7_drawer_pnp_01_2023-04-19_09-18-15_raw_traj_group0_traj6_im_0.jpg
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--base-url` | Base URL of the Bridge dataset | `https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2` |
| `--output-dir` | Output directory for downloaded data | `./bridge_data` |
| `--max-workers` | Maximum number of concurrent downloads | `5` |
| `--trajectories` | Specific trajectory paths to download | Auto-discover |
| `--single-trajectory` | Download a single trajectory | None |

## Example Run

Try the example script to see it in action:

```bash
python example_usage.py
```

This will:
1. Download the example trajectory you provided
2. Attempt to discover and download additional trajectories
3. Show you the output structure and files created

## URL Structure

The script expects the Bridge dataset to follow this URL structure:
```
{base_url}/{trajectory_path}/lang.txt
{base_url}/{trajectory_path}/images0/im_0.jpg
```

Example:
- Language: `https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj6/lang.txt`
- Image: `https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj6/images0/im_0.jpg`

## Troubleshooting

- **Connection issues**: The script uses timeouts and retry logic. Check your internet connection if downloads fail.
- **File permissions**: Ensure you have write permissions in the output directory.
- **Missing trajectories**: The auto-discovery feature is limited. For comprehensive downloading, manually specify trajectory paths.

## Features in Detail

- **Concurrent Downloads**: Uses ThreadPoolExecutor for fast parallel downloads
- **Error Handling**: Robust error handling with detailed logging  
- **Resume Capability**: Can resume interrupted downloads (CSV append mode)
- **Safe Filenames**: Converts trajectory paths to safe filenames for cross-platform compatibility 