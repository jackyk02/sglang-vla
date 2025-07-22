import requests
import os
import csv
import json
import shutil
from datetime import datetime
from image_transform import process_image
from token2action import TokenToAction

# Configuration
EXTRACTED_TRAJECTORIES_DIR = "./extracted_trajectories/"
IMAGES_DIR = os.path.join(EXTRACTED_TRAJECTORIES_DIR, "images")
INSTRUCTIONS_CSV = os.path.join(EXTRACTED_TRAJECTORIES_DIR, "instructions.csv")
OUTPUT_DIR = "./processed_images/"
TARGET_SIZE = (224, 224)
CROP_SCALE = 0.9

# Temperature settings and sample counts
TEMPERATURE_CONFIGS = [
    {"temperature": 0.0, "samples": 1, "folder": "temp_0.0"},
    {"temperature": 0.5, "samples": 10, "folder": "temp_0.5"},
    {"temperature": 1.0, "samples": 10, "folder": "temp_1.0"}
]

def create_prompt(instruction):
    """Create the chat prompt for the assistant."""
    return ("A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            f"USER: What action should the robot take to {instruction.lower()}? ASSISTANT: TASK:")

def load_datapoints():
    """Load all 10 datapoints from the CSV file."""
    datapoints = []
    with open(INSTRUCTIONS_CSV, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            datapoints.append({
                'image_number': int(row['image_number']),
                'trajectory_path': row['trajectory_path'],
                'instruction': row['instruction']
            })
    return datapoints

def setup_log_directory():
    """Create the base logs directory."""
    log_base_dir = "./logs"
    os.makedirs(log_base_dir, exist_ok=True)
    return log_base_dir

def make_inference_request(prompt, image_path, temperature, max_samples):
    """Make inference request with specified temperature."""
    # Prepare batch data
    prompt_batch = [prompt] * max_samples
    image_batch = [os.path.abspath(image_path)] * max_samples
    
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

def clean_instruction_for_folder(instruction):
    """Clean instruction text to make it a valid folder name."""
    # Replace problematic characters with underscores
    cleaned = instruction.replace('/', '_').replace('\\', '_').replace(':', '_')
    cleaned = cleaned.replace('?', '').replace('*', '').replace('<', '').replace('>', '')
    cleaned = cleaned.replace('|', '_').replace('"', '').replace("'", '')
    # Remove extra spaces and replace with underscores
    cleaned = '_'.join(cleaned.split())
    # Limit length to avoid filesystem issues
    if len(cleaned) > 100:
        cleaned = cleaned[:100]
    return cleaned

def save_images_and_metadata(log_base_dir, datapoint, original_image_path, processed_image_path):
    """Save images and metadata once per instruction (not per temperature)."""
    instruction_folder = clean_instruction_for_folder(datapoint['instruction'])
    instruction_dir = os.path.join(log_base_dir, instruction_folder)
    os.makedirs(instruction_dir, exist_ok=True)
    
    # Save instruction at instruction level
    instruction_file = os.path.join(instruction_dir, "instruction.txt")
    if not os.path.exists(instruction_file):  # Only save once
        with open(instruction_file, 'w') as f:
            f.write(datapoint['instruction'])
    
    # Save trajectory path info at instruction level
    trajectory_file = os.path.join(instruction_dir, "trajectory_info.txt")
    if not os.path.exists(trajectory_file):  # Only save once
        with open(trajectory_file, 'w') as f:
            f.write(f"Trajectory Path: {datapoint['trajectory_path']}\n")
            f.write(f"Image Number: {datapoint['image_number']}\n")
    
    # Copy original image
    original_dest = os.path.join(instruction_dir, f"original_image_{datapoint['image_number']}.jpg")
    if not os.path.exists(original_dest):  # Only copy once
        shutil.copy2(original_image_path, original_dest)
    
    # Copy processed image
    processed_dest = os.path.join(instruction_dir, f"processed_image_{datapoint['image_number']}.jpg")
    if not os.path.exists(processed_dest):  # Only copy once
        shutil.copy2(processed_image_path, processed_dest)

def save_results(log_base_dir, datapoint, temp_config, results):
    """Save results for a specific temperature configuration."""
    # Create instruction-based folder structure
    instruction_folder = clean_instruction_for_folder(datapoint['instruction'])
    instruction_dir = os.path.join(log_base_dir, instruction_folder)
    temp_dir = os.path.join(instruction_dir, temp_config["folder"])
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize TokenToAction converter
    token_to_action = TokenToAction()
    
    # Process results to extract continuous actions
    processed_results = []
    if results:
        # Handle case where results is a list of result objects (API response format)
        results_list = results if isinstance(results, list) else [results]
        
        for i, result in enumerate(results_list):
            # Copy the original result
            processed_result = result.copy()
            
            if 'meta_info' in result and 'output_ids' in result['meta_info']:
                output_ids = result['meta_info']['output_ids']
                if len(output_ids) >= 8:  # Ensure we have at least 8 tokens (7 action + 1 final)
                    # Extract the 7 action tokens (before the final token which should be 2)
                    action_tokens = output_ids[-8:-1]  # Get 7 tokens before the last one
                    try:
                        # Convert action tokens to continuous actions
                        continuous_actions = token_to_action.convert(action_tokens)
                        # Add continuous actions to the result
                        processed_result['continuous_actions'] = continuous_actions.tolist()
                        print(f"    Converted action tokens to continuous actions for sample {i+1}")
                    except Exception as e:
                        print(f"Warning: Failed to convert action tokens for result {i}: {str(e)}")
                        processed_result['continuous_actions'] = None
                else:
                    processed_result['continuous_actions'] = None
                    print(f"Warning: Not enough tokens for action extraction in result {i}")
            else:
                processed_result['continuous_actions'] = None
                print(f"Warning: No output_ids found in result {i}")
            
            processed_results.append(processed_result)
    
    # Save inference results with continuous actions
    results_file = os.path.join(temp_dir, "inference_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "temperature": temp_config["temperature"],
            "samples_requested": temp_config["samples"],
            "results": processed_results
        }, f, indent=2)
    
    # Save individual outputs
    if processed_results:
        for i, result in enumerate(processed_results):
            if 'text' in result:
                output_file = os.path.join(temp_dir, f"output_{i+1}.txt")
                with open(output_file, 'w') as f:
                    f.write(result['text'])
    
    # Save continuous actions separately for easy access
    if processed_results:
        actions_file = os.path.join(temp_dir, "continuous_actions.json")
        actions_data = {
            "actions": [result.get('continuous_actions') for result in processed_results],
            "temperature": temp_config["temperature"],
            "instruction": datapoint['instruction'],
            "num_samples": len(processed_results)
        }
        
        with open(actions_file, 'w') as f:
            json.dump(actions_data, f, indent=2)

def process_datapoint(datapoint, log_base_dir):
    """Process a single datapoint with all temperature configurations."""
    print(f"\nProcessing datapoint {datapoint['image_number']}: {datapoint['instruction']}")
    
    # Process the image
    original_image_path = os.path.join(IMAGES_DIR, f"{datapoint['image_number']}.jpg")
    if not os.path.exists(original_image_path):
        print(f"Warning: Image not found at {original_image_path}")
        return
    
    processed_image_path = process_image(
        image_path=original_image_path,
        output_dir=OUTPUT_DIR,
        crop_scale=CROP_SCALE,
        target_size=TARGET_SIZE,
        batch_size=1
    )
    
    # Save images and metadata once per instruction
    save_images_and_metadata(log_base_dir, datapoint, original_image_path, processed_image_path)
    print(f"  Saved images and metadata for instruction")
    
    # Create prompt
    prompt = create_prompt(datapoint['instruction'])
    
    # Process each temperature configuration
    for temp_config in TEMPERATURE_CONFIGS:
        print(f"  Temperature {temp_config['temperature']}: generating {temp_config['samples']} samples...")
        
        results = make_inference_request(
            prompt=prompt,
            image_path=processed_image_path,
            temperature=temp_config['temperature'],
            max_samples=temp_config['samples']
        )
        
        if results:
            save_results(log_base_dir, datapoint, temp_config, results)
            print(f"    Saved results to {temp_config['folder']}")
        else:
            print(f"    Failed to get results for temperature {temp_config['temperature']}")

def main():
    """Main batch inference pipeline."""
    print("Starting batch inference pipeline...")
    
    # Load datapoints
    datapoints = load_datapoints()
    print(f"Loaded {len(datapoints)} datapoints")
    
    # Setup log directory
    log_base_dir = setup_log_directory()
    print(f"Created log base directory: {log_base_dir}")
    
    # Process each datapoint
    for datapoint in datapoints:
        try:
            process_datapoint(datapoint, log_base_dir)
        except Exception as e:
            print(f"Error processing datapoint {datapoint['image_number']}: {str(e)}")
            continue
    
    print(f"\nBatch inference completed! Results saved in: {log_base_dir}")
    
    # Save summary
    summary_file = os.path.join(log_base_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_datapoints": len(datapoints),
            "temperature_configs": TEMPERATURE_CONFIGS,
            "datapoints": datapoints
        }, f, indent=2)

if __name__ == "__main__":
    main() 