import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import cv2
import os
from pathlib import Path

def add_chart_noise(img):
    """
    Add realistic chart-like noise to the image.
    
    Gaussian noise: Random noise that follows a normal (Gaussian) distribution.
    This simulates small random variations in pixel values, similar to what you might
    see in real charts due to market volatility or data collection noise.
    
    Grid lines: Simulates the background grid of a trading chart.
    """
    img = img.astype(np.float32)
    
    # Add Gaussian noise (random variations in pixel values)
    noise_level = np.random.randint(5, 20)
    noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
    noisy = cv2.add(img, noise)
    
    h, w = img.shape[:2]
    grid_color = (200, 200, 200)
    
    # Add horizontal grid lines
    for j in range(0, h, 50):
        cv2.line(noisy, (0, j), (w, j), grid_color, 1)
    
    # Add vertical grid lines
    for i in range(0, w, 50):
        cv2.line(noisy, (i, 0), (i, h), grid_color, 1)
    
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def custom_preprocess(img):
    """Custom preprocessing function for chart patterns"""
    # Add chart-specific noise
    img = add_chart_noise(img)
    # Normalize pixel values
    return img / 255.0
    

def generate_augmented_images(base_dir, num_augmentations):
    """
    Generate and save augmented images directly to the original directories.
    
    Parameters:
    - base_dir: Path to the directory containing pattern folders
    - num_augmentations: Number of augmented images to generate per original image
    """
    # Create generators
    datagen = ImageDataGenerator(
        rotation_range=15,  # Rotate images by up to 15 degrees
        width_shift_range=0.1,  # Shift width by up to 10%
        height_shift_range=0.1,  # Shift height by up to 10%
        zoom_range=0.15,  # Zoom in/out by up to 15%
        shear_range=10,  # Shear transformation up to 10 degrees
        brightness_range=(0.8, 1.2),  # Adjust brightness by ±20%
        preprocessing_function=custom_preprocess,
        fill_mode='nearest'
    )
    
    base_path = Path(base_dir)
    
    # Iterate over each class folder
    for class_folder in base_path.iterdir():
        if not class_folder.is_dir():
            continue
        
        image_paths = list(class_folder.glob("*.jpg")) 
        print(f"Augmenting class: {class_folder.name} ({len(image_paths)} images)")

        for img_path in image_paths:
            img = load_img(img_path, target_size=(224, 224))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            
            gen = datagen.flow(x, batch_size=1)
            
            # Generate augmented images
            for i in range(num_augmentations):
                aug_img = next(gen)[0]
                aug_img = np.clip(aug_img * 255, 0, 255).astype(np.uint8)
                save_path = class_folder / f"{img_path.stem}_aug_{i}.png"
                cv2.imwrite(str(save_path), aug_img)
                print(f"Saved: {save_path}")

def main():
    # Path to your pattern directories
    base_dir = Path("/mnt/c/Users/legen/Stock Analyzer/stockanalyzer/data/.cache/kagglehub/datasets/mustaphaelbakai/stock-chart-patterns/versions/5/Patterns")
    
    # Generate 10 augmented images for each pattern
    generate_augmented_images(base_dir, num_augmentations=5)

if __name__ == "__main__":
    main() 