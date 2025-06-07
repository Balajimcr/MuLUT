#!/usr/bin/env python3
"""
MuLUT Testing Script for 2x Super-Resolution
Tests pretrained Look-Up Tables (LUTs) for efficient image super-resolution.

Usage:
    python 4_test_lut.py --stages 2 --modes sdy -e ../models/sr_x2sdy

Based on the MuLUT paper: "Cooperating Multiple Look-Up Tables for Efficient Image Super-Resolution"
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import json
import math

# Global variable for hardcoded test directory
USER_TEST_DIR = None

# Image quality metrics
def calculate_psnr(img1, img2, max_val=255.0):
    """Calculate Peak Signal-to-Noise Ratio (PSNR)"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))

def calculate_ssim(img1, img2, max_val=255.0):
    """Calculate Structural Similarity Index (SSIM)"""
    # Convert to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Constants
    k1, k2 = 0.01, 0.03
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    
    # Calculate means
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    # Calculate SSIM
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    
    ssim_map = numerator / denominator
    return np.mean(ssim_map)

def verify_color_image(image, name="image"):
    """Helper function to verify color image statistics"""
    if len(image.shape) == 3:
        # RGB image
        r_stats = f"R: min={np.min(image[:,:,0])}, max={np.max(image[:,:,0])}, mean={np.mean(image[:,:,0]):.1f}"
        g_stats = f"G: min={np.min(image[:,:,1])}, max={np.max(image[:,:,1])}, mean={np.mean(image[:,:,1]):.1f}"
        b_stats = f"B: min={np.min(image[:,:,2])}, max={np.max(image[:,:,2])}, mean={np.mean(image[:,:,2]):.1f}"
        print(f"    {name} RGB stats: {r_stats}, {g_stats}, {b_stats}")
        
        # YUV stats
        yuv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2YUV)
        y_stats = f"Y: min={np.min(yuv[:,:,0])}, max={np.max(yuv[:,:,0])}, mean={np.mean(yuv[:,:,0]):.1f}"
        u_stats = f"U: min={np.min(yuv[:,:,1])}, max={np.max(yuv[:,:,1])}, mean={np.mean(yuv[:,:,1]):.1f}"
        v_stats = f"V: min={np.min(yuv[:,:,2])}, max={np.max(yuv[:,:,2])}, mean={np.mean(yuv[:,:,2]):.1f}"
        print(f"    {name} YUV stats: {y_stats}, {u_stats}, {v_stats}")
    else:
        # Grayscale image
        gray_stats = f"min={np.min(image)}, max={np.max(image)}, mean={np.mean(image):.1f}"
        print(f"    {name} Gray stats: {gray_stats}")

class MuLUTInference:
    """MuLUT inference engine using pretrained Look-Up Tables"""
    
    def __init__(self, lut_dir, stages=2, modes='sdy', scale=2):
        self.lut_dir = Path(lut_dir)
        self.stages = stages
        self.modes = modes
        self.scale = scale
        self.luts = []
        
        # Load all LUT files
        self.load_luts()
        
    def load_luts(self):
        """Load all LUT .npy files from the specified directory"""
        print(f"Loading LUTs from: {self.lut_dir}")
        
        # Define expected LUT filenames based on stages and modes
        lut_patterns = []
        for stage in range(self.stages):
            for mode in self.modes:
                lut_patterns.extend([
                    f"lut_s{stage}_{mode}_x.npy",
                    f"lut_s{stage}_{mode}_y.npy",
                    f"lut_stage{stage}_{mode}.npy",
                    f"mulut_s{stage}_{mode}.npy"
                ])
        
        # Try to find and load LUT files
        for pattern in lut_patterns:
            lut_path = self.lut_dir / pattern
            if lut_path.exists():
                try:
                    lut = np.load(lut_path)
                    self.luts.append({
                        'name': pattern,
                        'data': lut,
                        'shape': lut.shape
                    })
                    print(f"  Loaded {pattern}: shape {lut.shape}")
                except Exception as e:
                    print(f"  Failed to load {pattern}: {e}")
        
        # If no specific patterns found, load all .npy files
        if not self.luts:
            for lut_path in self.lut_dir.glob("*.npy"):
                try:
                    lut = np.load(lut_path)
                    self.luts.append({
                        'name': lut_path.name,
                        'data': lut,
                        'shape': lut.shape
                    })
                    print(f"  Loaded {lut_path.name}: shape {lut.shape}")
                except Exception as e:
                    print(f"  Failed to load {lut_path.name}: {e}")
        
        if not self.luts:
            raise ValueError(f"No LUT files found in {self.lut_dir}")
        
        print(f"Total LUTs loaded: {len(self.luts)}")
    
    def apply_lut_interpolation(self, patch, lut_data):
        """Apply LUT with interpolation for smooth results"""
        try:
            #print(f"    LUT shape: {lut_data.shape}, Patch shape: {patch.shape}, Scale: {self.scale}")
            
            if len(lut_data.shape) == 1:
                # 1D LUT case - direct lookup table
                h, w = patch.shape
                result = np.zeros((h * self.scale, w * self.scale), dtype=np.float32)
                
                for i in range(h):
                    for j in range(w):
                        pixel_val = int(np.clip(patch[i, j], 0, len(lut_data) - 1))
                        lut_result = lut_data[pixel_val]
                        
                        # Place single value in corresponding output location
                        start_i = i * self.scale
                        start_j = j * self.scale
                        end_i = min(start_i + self.scale, result.shape[0])
                        end_j = min(start_j + self.scale, result.shape[1])
                        
                        result[start_i:end_i, start_j:end_j] = lut_result
                
                return result
            
            elif len(lut_data.shape) == 2:
                # 2D LUT case - each entry maps to multiple output values
                h, w = patch.shape
                result = np.zeros((h * self.scale, w * self.scale), dtype=np.float32)
                
                # Determine LUT indexing method
                if lut_data.shape[0] == 256:  # Standard 8-bit LUT
                    for i in range(h):
                        for j in range(w):
                            pixel_val = int(np.clip(patch[i, j], 0, 255))
                            lut_result = lut_data[pixel_val]
                            
                            # Place the LUT result in the appropriate location
                            start_i = i * self.scale
                            start_j = j * self.scale
                            end_i = min(start_i + self.scale, result.shape[0])
                            end_j = min(start_j + self.scale, result.shape[1])
                            
                            # Handle different LUT result sizes
                            expected_size = (end_i - start_i) * (end_j - start_j)
                            
                            if len(lut_result) == expected_size:
                                # Perfect match - reshape directly
                                lut_reshaped = lut_result.reshape(end_i - start_i, end_j - start_j)
                            elif len(lut_result) == self.scale * self.scale:
                                # Standard scale x scale output
                                lut_reshaped = lut_result.reshape(self.scale, self.scale)
                                lut_reshaped = lut_reshaped[:end_i-start_i, :end_j-start_j]
                            else:
                                # Fallback - use first value or interpolate
                                print(f"    Warning: LUT result size {len(lut_result)} doesn't match expected {expected_size}")
                                if len(lut_result) > 0:
                                    # Use average of LUT values
                                    avg_val = np.mean(lut_result)
                                    lut_reshaped = np.full((end_i - start_i, end_j - start_j), avg_val)
                                else:
                                    lut_reshaped = np.zeros((end_i - start_i, end_j - start_j))
                            
                            result[start_i:end_i, start_j:end_j] = lut_reshaped
                else:
                    # Non-standard LUT size - use modular indexing
                    for i in range(h):
                        for j in range(w):
                            pixel_val = int(patch[i, j]) % lut_data.shape[0]
                            lut_result = lut_data[pixel_val]
                            
                            start_i = i * self.scale
                            start_j = j * self.scale
                            end_i = min(start_i + self.scale, result.shape[0])
                            end_j = min(start_j + self.scale, result.shape[1])
                            
                            # Use average if multiple values, otherwise broadcast single value
                            if len(lut_result) > 1:
                                avg_val = np.mean(lut_result)
                            else:
                                avg_val = lut_result[0] if len(lut_result) > 0 else 0
                            
                            result[start_i:end_i, start_j:end_j] = avg_val
                
                return result
            
            elif len(lut_data.shape) == 3:
                # 3D LUT case - patch-based lookup
                h, w = patch.shape
                result = np.zeros((h * self.scale, w * self.scale), dtype=np.float32)
                
                for i in range(h):
                    for j in range(w):
                        pixel_val = int(np.clip(patch[i, j], 0, lut_data.shape[0] - 1))
                        lut_result = lut_data[pixel_val]
                        
                        start_i = i * self.scale
                        start_j = j * self.scale
                        end_i = min(start_i + self.scale, result.shape[0])
                        end_j = min(start_j + self.scale, result.shape[1])
                        
                        # Handle 3D LUT result
                        if lut_result.shape == (self.scale, self.scale):
                            result[start_i:end_i, start_j:end_j] = lut_result[:end_i-start_i, :end_j-start_j]
                        else:
                            # Use average or first value
                            avg_val = np.mean(lut_result)
                            result[start_i:end_i, start_j:end_j] = avg_val
                
                return result
            
            else:
                print(f"    Warning: Unsupported LUT shape {lut_data.shape}, using fallback")
                return cv2.resize(patch, None, fx=self.scale, fy=self.scale, 
                                interpolation=cv2.INTER_CUBIC)
        
        except Exception as e:
            print(f"    Error in LUT application: {e}")
            print(f"    LUT shape: {lut_data.shape}, Patch shape: {patch.shape}")
            # Fallback to simple upsampling
            return cv2.resize(patch, None, fx=self.scale, fy=self.scale, 
                            interpolation=cv2.INTER_CUBIC)
    
    def process_image_patches(self, image, patch_size=4):
        """Process image using patch-based LUT inference - expects single channel image"""
        # Ensure we're working with a single-channel image
        if len(image.shape) == 3:
            print(f"    Warning: Multi-channel image passed to process_image_patches, using first channel")
            image = image[:, :, 0]
        
        h, w = image.shape
        result_h, result_w = h * self.scale, w * self.scale
        result = np.zeros((result_h, result_w), dtype=np.float32)
        
        print(f"    Processing {h}x{w} image in {patch_size}x{patch_size} patches -> {result_h}x{result_w}")
        
        # Process image in patches
        patches_processed = 0
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                # Extract patch
                patch = image[i:i+patch_size, j:j+patch_size]
                
                # Apply LUTs (use the first available LUT for simplicity)
                if self.luts:
                    try:
                        lut_result = self.apply_lut_interpolation(patch, self.luts[0]['data'])
                    except Exception as e:
                        print(f"    LUT processing failed for patch at ({i},{j}): {e}")
                        # Fallback upsampling
                        lut_result = cv2.resize(patch, None, fx=self.scale, fy=self.scale,
                                              interpolation=cv2.INTER_CUBIC)
                else:
                    # Fallback upsampling
                    lut_result = cv2.resize(patch, None, fx=self.scale, fy=self.scale,
                                          interpolation=cv2.INTER_CUBIC)
                
                # Place result in output image
                start_i, start_j = i * self.scale, j * self.scale
                end_i = min(start_i + lut_result.shape[0], result_h)
                end_j = min(start_j + lut_result.shape[1], result_w)
                
                result[start_i:end_i, start_j:end_j] = lut_result[:end_i-start_i, :end_j-start_j]
                patches_processed += 1
        
        print(f"    Processed {patches_processed} patches")
        return result
    
    def super_resolve(self, image):
        """Main super-resolution function with proper color handling"""
        # Normalize input
        if image.dtype == np.uint8:
            image = image.astype(np.float32)
        
        # Handle color images properly
        if len(image.shape) == 3 and image.shape[2] == 3:
            print(f"    Processing color image: {image.shape}")
            # Convert RGB to YUV for proper super-resolution
            # Y channel will be super-resolved, U and V will be upscaled
            yuv_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2YUV)
            y_channel = yuv_image[:, :, 0].astype(np.float32)
            u_channel = yuv_image[:, :, 1].astype(np.float32)
            v_channel = yuv_image[:, :, 2].astype(np.float32)
            
            print(f"    Y channel shape: {y_channel.shape}")
            print(f"    U channel shape: {u_channel.shape}")
            print(f"    V channel shape: {v_channel.shape}")
            
            # Super-resolve Y channel using LUTs
            if len(self.luts) > 1:
                # Multi-stage processing on Y channel
                sr_y = y_channel.copy()
                for stage_idx in range(min(self.stages, len(self.luts))):
                    sr_y = self.process_image_patches(sr_y)
                    print(f"    Stage {stage_idx} Y result shape: {sr_y.shape}")
            else:
                # Single-stage processing on Y channel
                sr_y = self.process_image_patches(y_channel)
            
            print(f"    Super-resolved Y shape: {sr_y.shape}")
            
            # Upscale U and V channels using bicubic interpolation
            target_h, target_w = sr_y.shape
            sr_u = cv2.resize(u_channel, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            sr_v = cv2.resize(v_channel, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            
            print(f"    Upscaled U shape: {sr_u.shape}")
            print(f"    Upscaled V shape: {sr_v.shape}")
            
            # Ensure all channels have the same shape
            min_h = min(sr_y.shape[0], sr_u.shape[0], sr_v.shape[0])
            min_w = min(sr_y.shape[1], sr_u.shape[1], sr_v.shape[1])
            
            sr_y = sr_y[:min_h, :min_w]
            sr_u = sr_u[:min_h, :min_w]
            sr_v = sr_v[:min_h, :min_w]
            
            # Combine YUV channels
            sr_yuv = np.stack([sr_y, sr_u, sr_v], axis=2)
            sr_yuv = np.clip(sr_yuv, 0, 255).astype(np.uint8)
            
            print(f"    Combined YUV shape: {sr_yuv.shape}")
            
            # Convert back to RGB
            result = cv2.cvtColor(sr_yuv, cv2.COLOR_YUV2RGB)
            print(f"    Final RGB result shape: {result.shape}")
            
        else:
            # Grayscale image processing
            print(f"    Processing grayscale image: {image.shape}")
            if len(image.shape) == 3:
                # Convert to grayscale if it's a 3-channel grayscale
                gray_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            else:
                gray_image = image.copy()
            
            # Process with LUTs
            if len(self.luts) > 1:
                # Multi-stage processing
                result = gray_image.copy()
                for stage_idx in range(min(self.stages, len(self.luts))):
                    result = self.process_image_patches(result)
            else:
                # Single-stage processing
                result = self.process_image_patches(gray_image)
            
            # Convert back to uint8
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result

def load_test_images(test_dir):
    """Load test images from directory"""
    test_images = []
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"Test directory not found: {test_dir}")
        return []
    
    for ext in image_extensions:
        for img_path in test_path.glob(f"*{ext}"):
            test_images.append(img_path)
        for img_path in test_path.glob(f"*{ext.upper()}"):
            test_images.append(img_path)
    
    print(f"Found {len(test_images)} test images in {test_dir}")
    return sorted(test_images)

def find_corresponding_hr_image(lr_path, lr_dir, hr_dir):
    """Find corresponding HR image for a given LR image"""
    if not hr_dir.exists():
        return None
    
    # Get relative path from LR directory
    try:
        rel_path = lr_path.relative_to(lr_dir)
        hr_path = hr_dir / rel_path
        
        if hr_path.exists():
            return hr_path
        
        # Try different naming conventions
        lr_name = lr_path.stem
        
        # Common naming patterns for LR/HR pairs
        patterns_to_try = [
            lr_name,  # Same name
            lr_name.replace('_LR', '').replace('_lr', ''),  # Remove LR suffix
            lr_name.replace('LR_', '').replace('lr_', ''),  # Remove LR prefix
            lr_name + '_HR',  # Add HR suffix
            'HR_' + lr_name,  # Add HR prefix
        ]
        
        for pattern in patterns_to_try:
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                hr_candidate = hr_dir / (pattern + ext)
                if hr_candidate.exists():
                    return hr_candidate
                hr_candidate = hr_dir / (pattern + ext.upper())
                if hr_candidate.exists():
                    return hr_candidate
        
    except ValueError:
        pass
    
    return None

def create_lr_hr_pairs_from_lr_dir(lr_dir, scale=2):
    """Create LR-HR image pairs from LR directory structure"""
    pairs = []
    lr_path = Path(lr_dir)
    
    if not lr_path.exists():
        print(f"LR directory not found: {lr_dir}")
        return pairs
    
    # Try to find corresponding HR directory
    possible_hr_dirs = [
        lr_path.parent.parent / "HR" / f"X{scale}",
        lr_path.parent.parent / "HR",
        lr_path.parent / "HR",
        lr_path.parent.parent / "GT" / f"X{scale}",  # Ground Truth
        lr_path.parent.parent / "GT",
        lr_path.parent / "GT",
        lr_path.parent.parent / "GroundTruth" / f"X{scale}",
        lr_path.parent.parent / "GroundTruth",
    ]
    
    hr_dir = None
    for possible_hr_dir in possible_hr_dirs:
        if possible_hr_dir.exists():
            hr_dir = possible_hr_dir
            print(f"Found HR directory: {hr_dir}")
            break
    
    # Load LR images
    lr_images = load_test_images(lr_dir)
    
    for lr_image_path in lr_images:
        try:
            # Load LR image
            lr_img = cv2.imread(str(lr_image_path))
            if lr_img is None:
                print(f"Could not load LR image: {lr_image_path}")
                continue
                
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
            
            # Try to find corresponding HR image
            hr_img = None
            hr_path = None
            
            if hr_dir:
                hr_path = find_corresponding_hr_image(lr_image_path, lr_path, hr_dir)
                if hr_path:
                    hr_img = cv2.imread(str(hr_path))
                    if hr_img is not None:
                        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
                        print(f"  Found HR pair for {lr_image_path.name}: {hr_path.name}")
                    else:
                        print(f"  Could not load HR image: {hr_path}")
                        hr_img = None
                        hr_path = None
            
            if hr_img is None:
                print(f"  No HR image found for {lr_image_path.name}, will create synthetic HR for metrics")
                # Create a synthetic HR by upsampling LR (for basic comparison)
                hr_img = cv2.resize(lr_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            pairs.append({
                'name': lr_image_path.stem,
                'hr': hr_img,
                'lr': lr_img,
                'lr_path': lr_image_path,
                'hr_path': hr_path,
                'has_ground_truth': hr_path is not None
            })
            
        except Exception as e:
            print(f"Error processing {lr_image_path}: {e}")
    
    print(f"Created {len(pairs)} LR-HR pairs")
    ground_truth_count = sum(1 for p in pairs if p['has_ground_truth'])
    print(f"  {ground_truth_count} pairs have ground truth HR images")
    print(f"  {len(pairs) - ground_truth_count} pairs use synthetic HR for comparison")
    
    return pairs

def create_lr_hr_pairs(hr_images, scale=2):
    """Create LR-HR image pairs for testing (legacy function for HR images)"""
    pairs = []
    
    for hr_path in hr_images:
        try:
            # Load HR image
            hr_img = cv2.imread(str(hr_path))
            if hr_img is None:
                continue
                
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
            
            # Create LR image by downsampling
            h, w = hr_img.shape[:2]
            lr_h, lr_w = h // scale, w // scale
            lr_img = cv2.resize(hr_img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
            
            pairs.append({
                'name': hr_path.stem,
                'hr': hr_img,
                'lr': lr_img,
                'hr_path': hr_path,
                'has_ground_truth': True
            })
            
        except Exception as e:
            print(f"Error processing {hr_path}: {e}")
    
    return pairs

def test_mulut_model(args):
    """Main testing function"""
    print("="*60)
    print("MuLUT Model Testing")
    print("="*60)
    
    # Initialize MuLUT inference
    lut_path = Path(args.experiment_dir)
    print(f"Experiment directory: {lut_path}")
    
    try:
        mulut_model = MuLUTInference(
            lut_dir=lut_path,
            stages=args.stages,
            modes=args.modes,
            scale=args.scale
        )
    except Exception as e:
        print(f"Error initializing MuLUT model: {e}")
        return
    
def test_mulut_model(args):
    """Main testing function"""
    print("="*60)
    print("MuLUT Model Testing")
    print("="*60)
    
    # Initialize MuLUT inference
    lut_path = Path(args.experiment_dir)
    print(f"Experiment directory: {lut_path}")
    
    try:
        mulut_model = MuLUTInference(
            lut_dir=lut_path,
            stages=args.stages,
            modes=args.modes,
            scale=args.scale
        )
    except Exception as e:
        print(f"Error initializing MuLUT model: {e}")
        return
    
    # Use hardcoded test directory path
    user_test_dir = USER_TEST_DIR
    if os.path.exists(user_test_dir):
        print(f"\nFound user test directory: {user_test_dir}")
        print("Using LR images from user-specified directory...")
        
        # Create pairs from LR directory
        image_pairs = create_lr_hr_pairs_from_lr_dir(user_test_dir, scale=args.scale)
        
        if image_pairs:
            results = {}
            dataset_name = f"UserTest_X{args.scale}"
            
            # Test the user images
            dataset_results = []
            total_time = 0
            
            print(f"\nTesting on {dataset_name} ({len(image_pairs)} images)...")
            
            for i, pair in enumerate(image_pairs):
                print(f"  Processing {pair['name']} ({i+1}/{len(image_pairs)})")
                
                try:
                    print(f"  Processing {pair['name']} ({i+1}/{len(image_pairs)})")
                    print(f"    Input LR shape: {pair['lr'].shape}, dtype: {pair['lr'].dtype}")
                    print(f"    Input HR shape: {pair['hr'].shape}, dtype: {pair['hr'].dtype}")
                    
                    # Verify input color statistics
                    verify_color_image(pair['lr'], "LR input")
                    
                    # Super-resolve LR image
                    start_time = time.time()
                    sr_img = mulut_model.super_resolve(pair['lr'])
                    inference_time = time.time() - start_time
                    total_time += inference_time
                    
                    print(f"    Output SR shape: {sr_img.shape}, dtype: {sr_img.dtype}")
                    
                    # Verify output color statistics
                    verify_color_image(sr_img, "SR output")
                    
                    # Resize HR for fair comparison (crop to match SR size)
                    hr_img = pair['hr']
                    if sr_img.shape[:2] != hr_img.shape[:2]:
                        min_h = min(sr_img.shape[0], hr_img.shape[0])
                        min_w = min(sr_img.shape[1], hr_img.shape[1])
                        sr_img = sr_img[:min_h, :min_w]
                        hr_img = hr_img[:min_h, :min_w]
                        print(f"    Cropped to match: SR {sr_img.shape}, HR {hr_img.shape}")
                    
                    # Calculate metrics on Y channel for proper evaluation
                    if len(sr_img.shape) == 3 and len(hr_img.shape) == 3:
                        # RGB images - calculate on Y channel (standard practice for SR evaluation)
                        sr_yuv = cv2.cvtColor(sr_img, cv2.COLOR_RGB2YUV)
                        hr_yuv = cv2.cvtColor(hr_img, cv2.COLOR_RGB2YUV)
                        sr_y = sr_yuv[:, :, 0]  # Y channel
                        hr_y = hr_yuv[:, :, 0]  # Y channel
                        print(f"    Evaluating on Y channel: SR {sr_y.shape}, HR {hr_y.shape}")
                    elif len(sr_img.shape) == 2 and len(hr_img.shape) == 2:
                        # Both grayscale
                        sr_y = sr_img
                        hr_y = hr_img
                        print(f"    Evaluating on grayscale: SR {sr_y.shape}, HR {hr_y.shape}")
                    else:
                        # Mixed formats - convert to grayscale
                        if len(sr_img.shape) == 3:
                            sr_y = cv2.cvtColor(sr_img, cv2.COLOR_RGB2GRAY)
                        else:
                            sr_y = sr_img
                        
                        if len(hr_img.shape) == 3:
                            hr_y = cv2.cvtColor(hr_img, cv2.COLOR_RGB2GRAY)
                        else:
                            hr_y = hr_img
                        print(f"    Evaluating on converted grayscale: SR {sr_y.shape}, HR {hr_y.shape}")
                    
                    psnr = calculate_psnr(sr_y, hr_y)
                    ssim = calculate_ssim(sr_y, hr_y)
                    
                    result = {
                        'name': pair['name'],
                        'psnr': psnr,
                        'ssim': ssim,
                        'time': inference_time,
                        'has_ground_truth': pair['has_ground_truth']
                    }
                    dataset_results.append(result)
                    
                    gt_status = "GT" if pair['has_ground_truth'] else "Synthetic"
                    print(f"    PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, Time: {inference_time:.3f}s [{gt_status}]")
                    
                    # Save result image with proper color handling
                    result_dir = Path(f"results/sr_x{args.scale}{args.modes}")
                    result_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save main super-resolved image
                    result_path = result_dir / f"{dataset_name}_{pair['name']}_sr.png"
                    if len(sr_img.shape) == 3:
                        sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(result_path), sr_img_bgr)
                        print(f"    Saved color SR image: {result_path}")
                    else:
                        cv2.imwrite(str(result_path), sr_img)
                        print(f"    Saved grayscale SR image: {result_path}")
                    
                    # Save debug images for color verification
                    if len(sr_img.shape) == 3:
                        # Save individual YUV channels for debugging
                        sr_yuv = cv2.cvtColor(sr_img, cv2.COLOR_RGB2YUV)
                        
                        # Save Y channel (should show super-resolved detail)
                        y_path = result_dir / f"{dataset_name}_{pair['name']}_sr_Y.png"
                        cv2.imwrite(str(y_path), sr_yuv[:, :, 0])
                        
                        # Save U channel (should show color info)
                        u_path = result_dir / f"{dataset_name}_{pair['name']}_sr_U.png"
                        cv2.imwrite(str(u_path), sr_yuv[:, :, 1])
                        
                        # Save V channel (should show color info)
                        v_path = result_dir / f"{dataset_name}_{pair['name']}_sr_V.png"
                        cv2.imwrite(str(v_path), sr_yuv[:, :, 2])
                        
                        print(f"    Saved debug channels: Y, U, V")
                    
                    # Also save LR input for comparison
                    lr_path = result_dir / f"{dataset_name}_{pair['name']}_lr.png"
                    if len(pair['lr'].shape) == 3:
                        lr_bgr = cv2.cvtColor(pair['lr'], cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(lr_path), lr_bgr)
                    else:
                        cv2.imwrite(str(lr_path), pair['lr'])
                    
                    # Save HR reference if available
                    if pair['has_ground_truth'] and pair['hr_path']:
                        hr_ref_path = result_dir / f"{dataset_name}_{pair['name']}_hr_ref.png"
                        if len(hr_img.shape) == 3:
                            hr_bgr = cv2.cvtColor(hr_img, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(hr_ref_path), hr_bgr)
                        else:
                            cv2.imwrite(str(hr_ref_path), hr_img)
                    
                except Exception as e:
                    print(f"    Error processing {pair['name']}: {e}")
            
            if dataset_results:
                # Calculate average metrics
                avg_psnr = np.mean([r['psnr'] for r in dataset_results if r['psnr'] != float('inf')])
                avg_ssim = np.mean([r['ssim'] for r in dataset_results])
                avg_time = np.mean([r['time'] for r in dataset_results])
                
                # Separate metrics for ground truth vs synthetic
                gt_results = [r for r in dataset_results if r['has_ground_truth']]
                syn_results = [r for r in dataset_results if not r['has_ground_truth']]
                
                results[dataset_name] = {
                    'avg_psnr': avg_psnr,
                    'avg_ssim': avg_ssim,
                    'avg_time': avg_time,
                    'num_images': len(dataset_results),
                    'num_ground_truth': len(gt_results),
                    'num_synthetic': len(syn_results),
                    'detailed_results': dataset_results
                }
                
                if gt_results:
                    gt_psnr = np.mean([r['psnr'] for r in gt_results if r['psnr'] != float('inf')])
                    gt_ssim = np.mean([r['ssim'] for r in gt_results])
                    results[dataset_name]['gt_avg_psnr'] = gt_psnr
                    results[dataset_name]['gt_avg_ssim'] = gt_ssim
                
                print(f"\n{dataset_name} Results:")
                print(f"  Average PSNR: {avg_psnr:.2f} dB")
                print(f"  Average SSIM: {avg_ssim:.4f}")
                print(f"  Average Time: {avg_time:.3f}s")
                print(f"  Total Images: {len(dataset_results)}")
                print(f"  Ground Truth: {len(gt_results)}")
                print(f"  Synthetic: {len(syn_results)}")
                
                if gt_results:
                    print(f"  GT-only PSNR: {results[dataset_name]['gt_avg_psnr']:.2f} dB")
                    print(f"  GT-only SSIM: {results[dataset_name]['gt_avg_ssim']:.4f}")
            
            # Save results
            if results:
                result_dir = Path(f"results/sr_x{args.scale}{args.modes}")
                result_dir.mkdir(parents=True, exist_ok=True)
                
                with open(result_dir / "test_results.json", 'w') as f:
                    json.dump(results, f, indent=2)
                
                print("\n" + "="*60)
                print("FINAL RESULTS SUMMARY")
                print("="*60)
                
                for dataset_name, result in results.items():
                    print(f"{dataset_name}:")
                    print(f"  PSNR: {result['avg_psnr']:.2f} dB")
                    print(f"  SSIM: {result['avg_ssim']:.4f}")
                    print(f"  Time: {result['avg_time']:.3f}s")
                    print(f"  Images: {result['num_images']} ({result['num_ground_truth']} GT + {result['num_synthetic']} Synthetic)")
                    if 'gt_avg_psnr' in result:
                        print(f"  GT-only: PSNR {result['gt_avg_psnr']:.2f} dB, SSIM {result['gt_avg_ssim']:.4f}")
                
                print(f"\nResults saved to: {result_dir}")
            
            return  # Exit after processing user test directory
        else:
            print("No valid image pairs found in user test directory.")
    else:
        print(f"User test directory not found: {user_test_dir}")
        print("Please ensure your LR test images are placed in the correct directory structure:")
        print(f"  {user_test_dir}/")
        print("    ├── image1.png")
        print("    ├── image2.jpg")
        print("    └── ...")
    
    # Fallback to standard benchmark datasets
    print("\nFalling back to standard benchmark datasets...")
    test_datasets = ['Set5', 'Set14', 'BSD100', 'Urban100', 'DIV2K_valid']
    
    results = {}
    
    for dataset_name in test_datasets:
        print(f"\nTesting on {dataset_name}...")
        
        # Look for test images in various possible locations
        possible_paths = [
            f"../data/SRBenchmark/{dataset_name}",
            f"../data/{dataset_name}",
            f"./data/{dataset_name}",
            f"./test_images/{dataset_name}",
            f"./{dataset_name}"
        ]
        
        test_images = []
        for path in possible_paths:
            test_images = load_test_images(path)
            if test_images:
                print(f"Using test images from: {path}")
                break
        
        if not test_images:
            print(f"No test images found for {dataset_name}, skipping...")
            continue
        
        # Create LR-HR pairs from HR images (legacy mode)
        image_pairs = create_lr_hr_pairs(test_images, scale=args.scale)
        
        if not image_pairs:
            print(f"No valid image pairs created for {dataset_name}")
            continue
        
        # Test each image
        dataset_results = []
        total_time = 0
        
        for i, pair in enumerate(image_pairs):
            print(f"  Processing {pair['name']} ({i+1}/{len(image_pairs)})")
            
            try:
                # Super-resolve LR image
                start_time = time.time()
                sr_img = mulut_model.super_resolve(pair['lr'])
                inference_time = time.time() - start_time
                total_time += inference_time
                
                # Resize HR for fair comparison (crop to match SR size)
                hr_img = pair['hr']
                if sr_img.shape[:2] != hr_img.shape[:2]:
                    min_h = min(sr_img.shape[0], hr_img.shape[0])
                    min_w = min(sr_img.shape[1], hr_img.shape[1])
                    sr_img = sr_img[:min_h, :min_w]
                    hr_img = hr_img[:min_h, :min_w]
                
                # Calculate metrics on Y channel for proper evaluation
                if len(sr_img.shape) == 3 and len(hr_img.shape) == 3:
                    # RGB images - calculate on Y channel (standard practice for SR evaluation)
                    sr_yuv = cv2.cvtColor(sr_img, cv2.COLOR_RGB2YUV)
                    hr_yuv = cv2.cvtColor(hr_img, cv2.COLOR_RGB2YUV)
                    sr_y = sr_yuv[:, :, 0]  # Y channel
                    hr_y = hr_yuv[:, :, 0]  # Y channel
                elif len(sr_img.shape) == 2 and len(hr_img.shape) == 2:
                    # Both grayscale
                    sr_y = sr_img
                    hr_y = hr_img
                else:
                    # Mixed formats - convert to grayscale
                    if len(sr_img.shape) == 3:
                        sr_y = cv2.cvtColor(sr_img, cv2.COLOR_RGB2GRAY)
                    else:
                        sr_y = sr_img
                    
                    if len(hr_img.shape) == 3:
                        hr_y = cv2.cvtColor(hr_img, cv2.COLOR_RGB2GRAY)
                    else:
                        hr_y = hr_img
                
                psnr = calculate_psnr(sr_y, hr_y)
                ssim = calculate_ssim(sr_y, hr_y)
                
                result = {
                    'name': pair['name'],
                    'psnr': psnr,
                    'ssim': ssim,
                    'time': inference_time,
                    'has_ground_truth': pair.get('has_ground_truth', True)
                }
                dataset_results.append(result)
                
                print(f"    PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, Time: {inference_time:.3f}s")
                
                # Save result image
                result_dir = Path(f"results/sr_x{args.scale}{args.modes}")
                result_dir.mkdir(parents=True, exist_ok=True)
                
                result_path = result_dir / f"{dataset_name}_{pair['name']}_sr.png"
                if len(sr_img.shape) == 3:
                    sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(result_path), sr_img_bgr)
                else:
                    cv2.imwrite(str(result_path), sr_img)
                
            except Exception as e:
                print(f"    Error processing {pair['name']}: {e}")
        
        if dataset_results:
            # Calculate average metrics
            avg_psnr = np.mean([r['psnr'] for r in dataset_results if r['psnr'] != float('inf')])
            avg_ssim = np.mean([r['ssim'] for r in dataset_results])
            avg_time = np.mean([r['time'] for r in dataset_results])
            
            results[dataset_name] = {
                'avg_psnr': avg_psnr,
                'avg_ssim': avg_ssim,
                'avg_time': avg_time,
                'num_images': len(dataset_results),
                'detailed_results': dataset_results
            }
            
            print(f"\n{dataset_name} Results:")
            print(f"  Average PSNR: {avg_psnr:.2f} dB")
            print(f"  Average SSIM: {avg_ssim:.4f}")
            print(f"  Average Time: {avg_time:.3f}s")
            print(f"  Total Images: {len(dataset_results)}")
    
    # Save results
    if results:
        result_dir = Path(f"results/sr_x{args.scale}{args.modes}")
        result_dir.mkdir(parents=True, exist_ok=True)
        
        with open(result_dir / "test_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        
        for dataset_name, result in results.items():
            print(f"{dataset_name}:")
            print(f"  PSNR: {result['avg_psnr']:.2f} dB")
            print(f"  SSIM: {result['avg_ssim']:.4f}")
            print(f"  Time: {result['avg_time']:.3f}s")
            print(f"  Images: {result['num_images']}")
        
        print(f"\nResults saved to: {result_dir}")
    else:
        print("No results generated. Please check your test data and LUT files.")
    if os.path.exists(user_test_dir):
        print(f"\nFound user test directory: {user_test_dir}")
        print("Using LR images from user-specified directory...")
        
        # Create pairs from LR directory
        image_pairs = create_lr_hr_pairs_from_lr_dir(user_test_dir, scale=args.scale)
        
        if image_pairs:
            results = {}
            dataset_name = f"UserTest_X{args.scale}"
            
            # Test the user images
            dataset_results = []
            total_time = 0
            
            print(f"\nTesting on {dataset_name} ({len(image_pairs)} images)...")
            
            for i, pair in enumerate(image_pairs):
                print(f"  Processing {pair['name']} ({i+1}/{len(image_pairs)})")
                
                try:
                    # Super-resolve LR image
                    start_time = time.time()
                    sr_img = mulut_model.super_resolve(pair['lr'])
                    inference_time = time.time() - start_time
                    total_time += inference_time
                    
                    # Resize HR for fair comparison (crop to match SR size)
                    hr_img = pair['hr']
                    if sr_img.shape[:2] != hr_img.shape[:2]:
                        min_h = min(sr_img.shape[0], hr_img.shape[0])
                        min_w = min(sr_img.shape[1], hr_img.shape[1])
                        sr_img = sr_img[:min_h, :min_w]
                        hr_img = hr_img[:min_h, :min_w]
                    
                    # Calculate metrics
                    if len(sr_img.shape) == 3 and len(hr_img.shape) == 3:
                        # RGB images - calculate on Y channel
                        sr_y = cv2.cvtColor(sr_img, cv2.COLOR_RGB2GRAY)
                        hr_y = cv2.cvtColor(hr_img, cv2.COLOR_RGB2GRAY)
                    else:
                        sr_y = sr_img
                        hr_y = hr_img if len(hr_img.shape) == 2 else cv2.cvtColor(hr_img, cv2.COLOR_RGB2GRAY)
                    
                    psnr = calculate_psnr(sr_y, hr_y)
                    ssim = calculate_ssim(sr_y, hr_y)
                    
                    result = {
                        'name': pair['name'],
                        'psnr': psnr,
                        'ssim': ssim,
                        'time': inference_time,
                        'has_ground_truth': pair['has_ground_truth']
                    }
                    dataset_results.append(result)
                    
                    gt_status = "GT" if pair['has_ground_truth'] else "Synthetic"
                    print(f"    PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, Time: {inference_time:.3f}s [{gt_status}]")
                    
                    # Save result image
                    result_dir = Path(f"results/sr_x{args.scale}{args.modes}")
                    result_dir.mkdir(parents=True, exist_ok=True)
                    
                    result_path = result_dir / f"{dataset_name}_{pair['name']}_sr.png"
                    if len(sr_img.shape) == 3:
                        sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(result_path), sr_img_bgr)
                    else:
                        cv2.imwrite(str(result_path), sr_img)
                    
                    # Also save LR input for comparison
                    lr_path = result_dir / f"{dataset_name}_{pair['name']}_lr.png"
                    if len(pair['lr'].shape) == 3:
                        lr_bgr = cv2.cvtColor(pair['lr'], cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(lr_path), lr_bgr)
                    else:
                        cv2.imwrite(str(lr_path), pair['lr'])
                    
                    # Save HR reference if available
                    if pair['has_ground_truth'] and pair['hr_path']:
                        hr_ref_path = result_dir / f"{dataset_name}_{pair['name']}_hr_ref.png"
                        if len(hr_img.shape) == 3:
                            hr_bgr = cv2.cvtColor(hr_img, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(hr_ref_path), hr_bgr)
                        else:
                            cv2.imwrite(str(hr_ref_path), hr_img)
                    
                except Exception as e:
                    print(f"    Error processing {pair['name']}: {e}")
            
            if dataset_results:
                # Calculate average metrics
                avg_psnr = np.mean([r['psnr'] for r in dataset_results if r['psnr'] != float('inf')])
                avg_ssim = np.mean([r['ssim'] for r in dataset_results])
                avg_time = np.mean([r['time'] for r in dataset_results])
                
                # Separate metrics for ground truth vs synthetic
                gt_results = [r for r in dataset_results if r['has_ground_truth']]
                syn_results = [r for r in dataset_results if not r['has_ground_truth']]
                
                results[dataset_name] = {
                    'avg_psnr': avg_psnr,
                    'avg_ssim': avg_ssim,
                    'avg_time': avg_time,
                    'num_images': len(dataset_results),
                    'num_ground_truth': len(gt_results),
                    'num_synthetic': len(syn_results),
                    'detailed_results': dataset_results
                }
                
                if gt_results:
                    gt_psnr = np.mean([r['psnr'] for r in gt_results if r['psnr'] != float('inf')])
                    gt_ssim = np.mean([r['ssim'] for r in gt_results])
                    results[dataset_name]['gt_avg_psnr'] = gt_psnr
                    results[dataset_name]['gt_avg_ssim'] = gt_ssim
                
                print(f"\n{dataset_name} Results:")
                print(f"  Average PSNR: {avg_psnr:.2f} dB")
                print(f"  Average SSIM: {avg_ssim:.4f}")
                print(f"  Average Time: {avg_time:.3f}s")
                print(f"  Total Images: {len(dataset_results)}")
                print(f"  Ground Truth: {len(gt_results)}")
                print(f"  Synthetic: {len(syn_results)}")
                
                if gt_results:
                    print(f"  GT-only PSNR: {results[dataset_name]['gt_avg_psnr']:.2f} dB")
                    print(f"  GT-only SSIM: {results[dataset_name]['gt_avg_ssim']:.4f}")
            
            # Save results
            if results:
                result_dir = Path(f"results/sr_x{args.scale}{args.modes}")
                result_dir.mkdir(parents=True, exist_ok=True)
                
                with open(result_dir / "test_results.json", 'w') as f:
                    json.dump(results, f, indent=2)
                
                print("\n" + "="*60)
                print("FINAL RESULTS SUMMARY")
                print("="*60)
                
                for dataset_name, result in results.items():
                    print(f"{dataset_name}:")
                    print(f"  PSNR: {result['avg_psnr']:.2f} dB")
                    print(f"  SSIM: {result['avg_ssim']:.4f}")
                    print(f"  Time: {result['avg_time']:.3f}s")
                    print(f"  Images: {result['num_images']} ({result['num_ground_truth']} GT + {result['num_synthetic']} Synthetic)")
                    if 'gt_avg_psnr' in result:
                        print(f"  GT-only: PSNR {result['gt_avg_psnr']:.2f} dB, SSIM {result['gt_avg_ssim']:.4f}")
                
                print(f"\nResults saved to: {result_dir}")
            
            return  # Exit after processing user test directory
        else:
            print("No valid image pairs found in user test directory.")
    
    # Fallback to standard benchmark datasets
    print("\nFalling back to standard benchmark datasets...")
    test_datasets = ['Set5', 'Set14', 'BSD100', 'Urban100', 'DIV2K_valid']
    
    results = {}
    
    for dataset_name in test_datasets:
        print(f"\nTesting on {dataset_name}...")
        
        # Look for test images in various possible locations
        possible_paths = [
            f"../data/SRBenchmark/{dataset_name}",
            f"../data/{dataset_name}",
            f"./data/{dataset_name}",
            f"./test_images/{dataset_name}",
            f"./{dataset_name}"
        ]
        
        test_images = []
        for path in possible_paths:
            test_images = load_test_images(path)
            if test_images:
                print(f"Using test images from: {path}")
                break
        
        if not test_images:
            print(f"No test images found for {dataset_name}, skipping...")
            continue
        
        # Create LR-HR pairs from HR images (legacy mode)
        image_pairs = create_lr_hr_pairs(test_images, scale=args.scale)
        
        if not image_pairs:
            print(f"No valid image pairs created for {dataset_name}")
            continue
        
        # Test each image
        dataset_results = []
        total_time = 0
        
        for i, pair in enumerate(image_pairs):
            print(f"  Processing {pair['name']} ({i+1}/{len(image_pairs)})")
            
            try:
                # Super-resolve LR image
                start_time = time.time()
                sr_img = mulut_model.super_resolve(pair['lr'])
                inference_time = time.time() - start_time
                total_time += inference_time
                
                # Resize HR for fair comparison (crop to match SR size)
                hr_img = pair['hr']
                if sr_img.shape[:2] != hr_img.shape[:2]:
                    min_h = min(sr_img.shape[0], hr_img.shape[0])
                    min_w = min(sr_img.shape[1], hr_img.shape[1])
                    sr_img = sr_img[:min_h, :min_w]
                    hr_img = hr_img[:min_h, :min_w]
                
                # Calculate metrics
                if len(sr_img.shape) == 3 and len(hr_img.shape) == 3:
                    # RGB images - calculate on Y channel
                    sr_y = cv2.cvtColor(sr_img, cv2.COLOR_RGB2GRAY)
                    hr_y = cv2.cvtColor(hr_img, cv2.COLOR_RGB2GRAY)
                else:
                    sr_y = sr_img
                    hr_y = hr_img if len(hr_img.shape) == 2 else cv2.cvtColor(hr_img, cv2.COLOR_RGB2GRAY)
                
                psnr = calculate_psnr(sr_y, hr_y)
                ssim = calculate_ssim(sr_y, hr_y)
                
                result = {
                    'name': pair['name'],
                    'psnr': psnr,
                    'ssim': ssim,
                    'time': inference_time,
                    'has_ground_truth': pair.get('has_ground_truth', True)
                }
                dataset_results.append(result)
                
                print(f"    PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, Time: {inference_time:.3f}s")
                
                # Save result image
                result_dir = Path(f"results/sr_x{args.scale}{args.modes}")
                result_dir.mkdir(parents=True, exist_ok=True)
                
                result_path = result_dir / f"{dataset_name}_{pair['name']}_sr.png"
                if len(sr_img.shape) == 3:
                    sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(result_path), sr_img_bgr)
                else:
                    cv2.imwrite(str(result_path), sr_img)
                
            except Exception as e:
                print(f"    Error processing {pair['name']}: {e}")
        
        if dataset_results:
            # Calculate average metrics
            avg_psnr = np.mean([r['psnr'] for r in dataset_results if r['psnr'] != float('inf')])
            avg_ssim = np.mean([r['ssim'] for r in dataset_results])
            avg_time = np.mean([r['time'] for r in dataset_results])
            
            results[dataset_name] = {
                'avg_psnr': avg_psnr,
                'avg_ssim': avg_ssim,
                'avg_time': avg_time,
                'num_images': len(dataset_results),
                'detailed_results': dataset_results
            }
            
            print(f"\n{dataset_name} Results:")
            print(f"  Average PSNR: {avg_psnr:.2f} dB")
            print(f"  Average SSIM: {avg_ssim:.4f}")
            print(f"  Average Time: {avg_time:.3f}s")
            print(f"  Total Images: {len(dataset_results)}")
    
    # Save results
    if results:
        result_dir = Path(f"results/sr_x{args.scale}{args.modes}")
        result_dir.mkdir(parents=True, exist_ok=True)
        
        with open(result_dir / "test_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        
        for dataset_name, result in results.items():
            print(f"{dataset_name}:")
            print(f"  PSNR: {result['avg_psnr']:.2f} dB")
            print(f"  SSIM: {result['avg_ssim']:.4f}")
            print(f"  Time: {result['avg_time']:.3f}s")
            print(f"  Images: {result['num_images']}")
        
        print(f"\nResults saved to: {result_dir}")
    else:
        print("No results generated. Please check your test data and LUT files.")

def main():
    parser = argparse.ArgumentParser(description='Test MuLUT model with pretrained LUTs')
    parser.add_argument('--stages', type=int, default=2, help='Number of LUT stages')
    parser.add_argument('--modes', type=str, default='sdy', help='LUT modes (sdy, rotations, etc.)')
    parser.add_argument('-e', '--experiment_dir', type=str, default=None,
                       help='Path to experiment directory containing LUT files')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size for LUT processing')
    parser.add_argument('--scale', type=int, default=2, help='Super-resolution scale factor')
    
    args = parser.parse_args()
    
    # Hardcoded default paths - try these in order
    if args.experiment_dir is None:
        default_experiment_paths = [
            f"models/sr_x{args.scale}{args.modes}",           # Local models directory
            f"../models/sr_x{args.scale}{args.modes}",        # Parent directory models
            f"./models/sr_x{args.scale}{args.modes}",         # Current directory models
            f"MuLUT/models/sr_x{args.scale}{args.modes}",     # MuLUT subdirectory
            f"../MuLUT/models/sr_x{args.scale}{args.modes}",  # Parent MuLUT directory
            f"sr/models/sr_x{args.scale}{args.modes}",        # SR subdirectory
            f"models/sr_x2sdy",                               # Fallback to default 2x sdy
            f"../models/sr_x2sdy",                            # Fallback parent
        ]
        
        print("No experiment directory specified. Searching for LUT files in common locations...")
        experiment_dir_found = None
        
        for path in default_experiment_paths:
            if os.path.exists(path):
                # Check if it actually contains LUT files
                lut_files = [f for f in os.listdir(path) if f.endswith('.npy')]
                if lut_files:
                    experiment_dir_found = path
                    print(f"Found LUT files in: {path}")
                    print(f"  LUT files: {lut_files}")
                    break
                else:
                    print(f"  Directory exists but no .npy files found: {path}")
            else:
                print(f"  Directory not found: {path}")
        
        if experiment_dir_found:
            args.experiment_dir = experiment_dir_found
            print(f"\nUsing experiment directory: {args.experiment_dir}")
        else:
            print("\nError: No LUT files found in any common locations!")
            print("Please specify the experiment directory manually with -e argument")
            print("Example: python 4_test_lut.py -e /path/to/your/lut/files")
            print("\nSearched locations:")
            for path in default_experiment_paths:
                print(f"  {path}")
            sys.exit(1)
    else:
        # Validate user-specified path
        if not os.path.exists(args.experiment_dir):
            print(f"Error: Experiment directory not found: {args.experiment_dir}")
            sys.exit(1)
        
        # Check if it contains LUT files
        lut_files = [f for f in os.listdir(args.experiment_dir) if f.endswith('.npy')]
        if not lut_files:
            print(f"Warning: No .npy files found in {args.experiment_dir}")
            print("Make sure this directory contains the pretrained LUT files")
    
    # Hardcoded test directory - override the function's search
    global USER_TEST_DIR
    USER_TEST_DIR = f"../data/Test/LR/X{args.scale}"
    
    print(f"\nConfiguration:")
    print(f"  Stages: {args.stages}")
    print(f"  Modes: {args.modes}")
    print(f"  Scale: {args.scale}x")
    print(f"  Patch size: {args.patch_size}")
    print(f"  Experiment dir: {args.experiment_dir}")
    print(f"  Test images dir: {USER_TEST_DIR}")
    
    # Run testing
    test_mulut_model(args)

if __name__ == "__main__":
    main()