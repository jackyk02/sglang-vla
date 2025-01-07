import requests
import json_numpy as json
from PIL import Image
import numpy as np
import os

def get_batch_actions(instruction: str, image_path: str, batch_size: int = 4, temperature: float = 1.0):
    """
    Get batch predictions from the batch processing server.
    
    Args:
        instruction (str): The instruction for the robot
        image_path (str): Path to the input image
        batch_size (int, optional): Size of the batch. Defaults to 4.
        temperature (float, optional): Sampling temperature. Defaults to 1.0.
    
    Returns:
        numpy.ndarray: Array of predicted actions
    """
    # Verify image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Prepare the payload
    payload = {
        "instruction": instruction,
        "image_path": image_path,
        "batch_size": batch_size,
        "temperature": temperature
    }
    
    # Send request to server
    response = requests.post(
        "http://127.0.0.1:3200/batch",
        data=json.dumps(payload),
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code != 200:
        raise Exception(f"Error from server: {response.text}")
    
    response_data = json.loads(response.text)
    return np.array(response_data["actions"])

# Example usage
if __name__ == "__main__":
    # Example instruction and image path
    instruction = "move the yellow knife to the right of the pan"
    image_path = "/root/sglang-vla/serve_vla/images/robot.jpg"
    
    try:
        # Get batch predictions
        actions = get_batch_actions(
            instruction=instruction,
            image_path=image_path,
            batch_size=1,
            temperature=0
        )
        print("Predicted actions:")
        print(actions)
        print(f"Shape of actions array: {actions.shape}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")