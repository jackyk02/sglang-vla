import tensorflow as tf
import numpy as np
from PIL import Image
import os

def process_image(image_path, output_dir="./transfer_images/", crop_scale=0.9, target_size=(224, 224), batch_size=1):
    """
    Process an image by center-cropping and resizing using TensorFlow.
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save the processed image
        crop_scale (float): The area of the center crop with respect to the original image
        target_size (tuple): Target size for the processed image (height, width)
        batch_size (int): Batch size for processing
        
    Returns:
        str: Path to the processed image
    """
    def crop_and_resize(image, crop_scale, batch_size, target_size):
        """
        Center-crops an image and resizes it back to target size.
        
        Args:
            image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C)
            crop_scale: The area of the center crop with respect to the original image
            batch_size: Batch size
            target_size: Tuple of (height, width) for the output image
        """
        # Handle input dimensions
        if image.shape.ndims == 3:
            image = tf.expand_dims(image, axis=0)
            expanded_dims = True
        else:
            expanded_dims = False

        # Calculate crop dimensions
        new_scale = tf.reshape(
            tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), 
            shape=(batch_size,)
        )
        
        # Calculate bounding box
        offsets = (1 - new_scale) / 2
        bounding_boxes = tf.stack(
            [
                offsets,          # height offset
                offsets,          # width offset
                offsets + new_scale,  # height + offset
                offsets + new_scale   # width + offset
            ],
            axis=1
        )

        # Perform crop and resize
        image = tf.image.crop_and_resize(
            image, 
            bounding_boxes, 
            tf.range(batch_size), 
            target_size
        )

        # Remove batch dimension if input was 3D
        if expanded_dims:
            image = image[0]

        return image

    try:
        # Load and convert image to tensor
        image = Image.open(image_path)
        image = image.convert("RGB")

        current_size = image.size  # Returns (width, height)
        
        # Check if current size matches target size (accounting for PIL's width,height order)
        if current_size == (target_size[1], target_size[0]):
            return image_path
            
        image = tf.convert_to_tensor(np.array(image))
        
        # Store original dtype
        original_dtype = image.dtype

        # Convert to float32 [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Apply transformations
        image = crop_and_resize(image, crop_scale, batch_size, target_size)

        # Convert back to original dtype
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, original_dtype, saturate=True)

        # Convert to PIL Image and save
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

        # Create output directory and save
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "processed_image.jpg")
        image.save(output_path)

        return output_path

    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")