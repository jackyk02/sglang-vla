<div align="center" id="sglangtop">
<img src="assets/logo.png" alt="SGLang VLA Logo" width="400" margin="10px"></img>
</div>

---

# SGLang VLA: Optimized Serving Engine for OpenVLA CoT

This repository provides a **high-performance serving engine for OpenVLA** and other Prismatic-VLM models.

## 🛠️ Prerequisites

- **Python**: 3.9 or higher
- **CUDA**: Compatible GPU with CUDA 12.1 support
- **System**: Linux-based environment recommended

## 📦 Environment Setup

### 1. Create and Activate Conda Environment

```bash
conda create -n sglang-vla python=3.9 -y
conda activate sglang-vla
```

### 2. Install Dependencies

Install the required packages in the following order:

```bash
# Core SGLang package
pip install -e "python[all]"

# Additional utilities
pip install json_numpy

# GPU acceleration (CUDA 12.1 + PyTorch 2.4)
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Vision and ML dependencies
pip install timm==0.9.10 openpyxl==3.1.5

# Web server components
pip install fastapi uvicorn

# Framework dependencies
pip install tensorflow==2.19.0
pip install transformers==4.51.3
```

## 🚀 Quick Start

### 1. Launch the Inference Server

Start the OpenVLA inference server:

```bash
conda activate sglang-vla
python -m sglang.launch_server \
    --model-path Embodied-CoT/ecot-openvla-7b-bridge \
    --trust-remote-code \
    --port 30000 \
    --disable-radix-cache
```

The server will be available at `http://localhost:30000` once started.

### 2. Run Basic Inference Example

Test the setup with a simple inference example:

```bash
python traces/example.py
```

## 📊 Working with Trajectories

### Extract Trajectory Data

Download and extract trajectory URLs from the dataset:

```bash
cd traces
python extract_trajectories.py
```

This script will create an `extracted_trajectories/` folder containing 10 trajectory URLs for processing.

### Generate Reasoning Traces

Process the extracted trajectories to generate reasoning traces:

```bash
cd traces
python batch_inference.py
```

**Output**: The reasoning traces and logs will be saved in the `traces/logs/` directory.

## 📁 Repository Structure

```
traces/
├── example.py              # Basic inference example
├── extract_trajectories.py # Trajectory extraction script
├── batch_inference.py      # Batch processing for reasoning traces
├── image_transform.py      # Image preprocessing utilities
├── token2action.py         # Token-to-action conversion
├── extracted_trajectories/ # Downloaded trajectory URLs
├── processed_images/       # Processed image data
└── logs/                   # Generated reasoning traces and logs
``


