import requests
import os
from image_transform import process_image

# Configuration
INSTRUCTION = "place the watermelon on the towel"
IMAGE_PATH = "/root/sglang-vla/traces/test_obs.jpg"
OUTPUT_DIR = "./processed_images/"
TARGET_SIZE = (224, 224)  # Default size for vision models
CROP_SCALE = 0.9

def create_prompt(instruction):
    """Create the chat prompt for the assistant."""
    return ("A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            f"USER: What action should the robot take to {instruction.lower()}? ASSISTANT: TASK:")

def repeat_string(s, batch_size):
    """Repeat a string for batch processing."""
    return [s] * batch_size

def main():
    """Main inference pipeline."""
    # Step 1: Process the image using image_transform
    print("Processing image...")
    processed_image_path = process_image(
        image_path=IMAGE_PATH,
        output_dir=OUTPUT_DIR,
        crop_scale=CROP_SCALE,
        target_size=TARGET_SIZE,
        batch_size=1
    )
    print(f"Image processed and saved to: {processed_image_path}")
    
    # Step 2: Create the prompt
    prompt = create_prompt(INSTRUCTION)
    print(f"Prompt: {prompt}")
    
    # Step 3: Prepare batch data
    batch_size = 5
    prompt_batch = repeat_string(prompt, batch_size)
    # Convert to absolute path for the request
    absolute_image_path = os.path.abspath(processed_image_path)
    image_batch = repeat_string(absolute_image_path, batch_size)
    
    # Step 4: Make inference request
    print("Making inference request...")
    response = requests.post(
        "http://localhost:30000/generate",
        json={
            "text": prompt_batch,
            "image_data": image_batch,
            "sampling_params": {
                "temperature": 0.5,
                "max_new_tokens": 2048,
            },
        },
    )
    
    # Step 5: Process and display results
    if response.status_code == 200:
        result = response.json()
        print("\n" + "="*50)
        print("INFERENCE RESULT:")
        print("="*50)
        print(result)
        print("="*50)
    else:
        print(f"Error: HTTP {response.status_code}")
        print(response.text)
            

if __name__ == "__main__":
    main() 