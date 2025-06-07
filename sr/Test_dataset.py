import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

# Paths
hr_path = 'C:/D/Project_Files/Samsung/MuLUT_Balaji/data/Test/HR/*.png'
lr_base_path = 'C:/D/Project_Files/Samsung/MuLUT_Balaji/data/Test/LR'

# Create directories if not exist
scales = [2, 3, 4]
for scale in scales:
    os.makedirs(os.path.join(lr_base_path, f'X{scale}'), exist_ok=True)

# Function to process a single image
def process_image(img_path):
    img = Image.open(img_path)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    
    for scale in scales:
        new_size = (img.width // scale, img.height // scale)
        img_resized = img.resize(new_size, resample=Image.BICUBIC)
        output_path = os.path.join(lr_base_path, f'X{scale}', f'{img_name}x{scale}.png')
        img_resized.save(output_path)

# Get list of all images
img_paths = glob.glob(hr_path)
total_images = len(img_paths)

# Use OpenMP for parallel processing with progress bar
num_cores = multiprocessing.cpu_count()
print(f"Processing {total_images} images using {num_cores} CPU cores...")

Parallel(n_jobs=num_cores, backend="threading")(
    delayed(process_image)(img_path) 
    for img_path in tqdm(img_paths, desc="Processing images", unit="img")
)

print("All images processed and saved.")
