import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import os
import argparse
import socket
import json
from matplotlib.patches import Polygon

def is_remote_server():
    """Check if running on a remote server"""
    try:
        # Try to get display
        display = os.environ.get('DISPLAY')
        if not display:
            return True
        return False
    except:
        return True

def visualize_gtFine(gt_path, save_path=None):
    """
    Visualize a Cityscapes ground truth segmentation mask with proper color mapping.
    
    Args:
        gt_path (str): Path to the gtFine PNG file
        save_path (str, optional): Path to save the visualization. If None, will display the image.
    """
    # Cityscapes color mapping (RGB values for each class)
    cityscapes_colors = {
        0: [0, 0, 0],        # unlabeled
        1: [70, 70, 70],     # road
        2: [100, 40, 40],    # sidewalk
        3: [55, 90, 80],     # building
        4: [220, 20, 60],    # wall
        5: [153, 153, 153],  # fence
        6: [157, 234, 50],   # pole
        7: [128, 64, 128],   # traffic light
        8: [244, 35, 232],   # traffic sign
        9: [107, 142, 35],   # vegetation
        10: [0, 0, 142],     # terrain
        11: [102, 102, 156], # sky
        12: [220, 220, 0],   # person
        13: [70, 130, 180],  # rider
        14: [81, 0, 81],     # car
        15: [150, 100, 100], # truck
        16: [230, 150, 140], # bus
        17: [180, 165, 180], # train
        18: [250, 170, 30],  # motorcycle
        19: [110, 190, 160], # bicycle
        255: [0, 0, 0]       # ignore
    }
    
    # Read the ground truth mask
    gt = np.array(Image.open(gt_path))
    
    # Create RGB visualization
    h, w = gt.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Map each class to its color
    for class_id, color in cityscapes_colors.items():
        rgb_mask[gt == class_id] = color
    
    # Create figure
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb_mask)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def visualize_batch_gtFine(gt_batch, save_path=None):
    """
    Visualize a batch of ground truth segmentation masks.
    
    Args:
        gt_batch (torch.Tensor): Batch of ground truth masks [B, H, W]
        save_path (str, optional): Path to save the visualization. If None, will display the image.
    """
    # Convert to numpy if tensor
    if isinstance(gt_batch, torch.Tensor):
        gt_batch = gt_batch.cpu().numpy()
    
    # Cityscapes color mapping (RGB values for each class)
    cityscapes_colors = {
        0: [0, 0, 0],        # unlabeled
        1: [70, 70, 70],     # road
        2: [100, 40, 40],    # sidewalk
        3: [55, 90, 80],     # building
        4: [220, 20, 60],    # wall
        5: [153, 153, 153],  # fence
        6: [157, 234, 50],   # pole
        7: [128, 64, 128],   # traffic light
        8: [244, 35, 232],   # traffic sign
        9: [107, 142, 35],   # vegetation
        10: [0, 0, 142],     # terrain
        11: [102, 102, 156], # sky
        12: [220, 220, 0],   # person
        13: [70, 130, 180],  # rider
        14: [81, 0, 81],     # car
        15: [150, 100, 100], # truck
        16: [230, 150, 140], # bus
        17: [180, 165, 180], # train
        18: [250, 170, 30],  # motorcycle
        19: [110, 190, 160], # bicycle
        255: [0, 0, 0]       # ignore
    }
    
    batch_size = gt_batch.shape[0]
    fig, axes = plt.subplots(1, batch_size, figsize=(4*batch_size, 4))
    if batch_size == 1:
        axes = [axes]
    
    for i, gt in enumerate(gt_batch):
        # Create RGB visualization
        h, w = gt.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Map each class to its color
        for class_id, color in cityscapes_colors.items():
            rgb_mask[gt == class_id] = color
        
        axes[i].imshow(rgb_mask)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def visualize_image_and_gtFine(image_path, gt_path, json_path=None, save_path=None):
    """
    Visualize both input image and its ground truth segmentation mask side by side.
    
    Args:
        image_path (str): Path to the input image
        gt_path (str): Path to the gtFine_labelIds.png file
        json_path (str, optional): Path to the gtFine_polygons.json file
        save_path (str, optional): Path to save the visualization. If None, will display the image.
    """
    print(f"Loading image from: {image_path}")
    print(f"Loading ground truth from: {gt_path}")
    if json_path:
        print(f"Loading polygons from: {json_path}")
    
    # Read the input image and ground truth mask
    image = np.array(Image.open(image_path))
    gt = np.array(Image.open(gt_path))
    
    print(f"Image shape: {image.shape}")
    print(f"Ground truth shape: {gt.shape}")
    print(f"Ground truth dtype: {gt.dtype}")
    print(f"Ground truth min value: {np.min(gt)}")
    print(f"Ground truth max value: {np.max(gt)}")
    
    # For labelIds format, we need to ensure we have a single channel
    if len(gt.shape) > 2:
        print("Warning: Ground truth has multiple channels. Using first channel.")
        gt = gt[:, :, 0]
    
    # Convert to uint8 if needed
    if gt.dtype != np.uint8:
        print(f"Converting ground truth from {gt.dtype} to uint8")
        gt = gt.astype(np.uint8)
    
    print(f"After processing - shape: {gt.shape}")
    print(f"After processing - min value: {np.min(gt)}, max value: {np.max(gt)}")
    
    # Count pixels for each class
    unique_classes = np.unique(gt)
    print("\nClass distribution:")
    total_pixels = gt.size
    for class_id in unique_classes:
        pixel_count = np.sum(gt == class_id)
        percentage = (pixel_count / total_pixels) * 100
        print(f"Class {class_id}: {pixel_count} pixels ({percentage:.2f}%)")
    
    # Cityscapes color mapping (RGB values for each class) - adjusted for better visibility
    cityscapes_colors = {
        0: [0, 0, 0],        # unlabeled
        1: [128, 64, 128],   # road (purple)
        2: [244, 35, 232],   # sidewalk (pink)
        3: [70, 70, 70],     # building (dark gray)
        4: [102, 102, 156],  # wall (blue-gray)
        5: [190, 153, 153],  # fence (light brown)
        6: [153, 153, 153],  # pole (gray)
        7: [250, 170, 30],   # traffic light (orange)
        8: [220, 220, 0],    # traffic sign (yellow)
        9: [107, 142, 35],   # vegetation (green)
        10: [152, 251, 152], # terrain (light green)
        11: [70, 130, 180],  # sky (blue)
        12: [220, 20, 60],   # person (red)
        13: [255, 0, 0],     # rider (bright red)
        14: [0, 0, 142],     # car (dark blue)
        15: [0, 0, 70],      # truck (darker blue)
        16: [0, 60, 100],    # bus (navy blue)
        17: [0, 80, 100],    # train (dark navy)
        18: [0, 0, 230],     # motorcycle (bright blue)
        19: [119, 11, 32],   # bicycle (burgundy)
        255: [128, 128, 128] # out of ROI (gray)
    }
    
    # Create RGB visualization for ground truth
    h, w = gt.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Map each class to its color
    for class_id, color in cityscapes_colors.items():
        mask = (gt == class_id)
        if np.any(mask):
            print(f"Found pixels for class {class_id}")
            rgb_mask[mask] = color
    
    print(f"Created RGB mask with shape: {rgb_mask.shape}")
    print(f"RGB mask min values: {np.min(rgb_mask, axis=(0,1))}")
    print(f"RGB mask max values: {np.max(rgb_mask, axis=(0,1))}")
    
    # Create figure with three subplots (image, mask, and colorbar)
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.1])
    
    # Plot input image
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(image)
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    # Plot ground truth mask
    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(rgb_mask)
    ax2.set_title('Ground Truth Segmentation\n(255 = out of ROI)')
    ax2.axis('off')
    
    # If JSON file is provided, plot the polygons
    if json_path:
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # Plot polygons on the segmentation mask
            for obj in json_data['objects']:
                label = obj['label']
                polygon = obj['polygon']
                
                # Convert polygon to numpy array
                polygon = np.array(polygon)
                
                # Create polygon patch
                poly = Polygon(polygon, fill=False, edgecolor='red', linewidth=2)
                ax2.add_patch(poly)
                
                # Add label
                if label != "out of roi":  # Don't label out of roi
                    x, y = polygon.mean(axis=0)
                    ax2.text(x, y, label, color='white', 
                            bbox=dict(facecolor='red', alpha=0.5))
            
            print(f"\nFound {len(json_data['objects'])} objects in JSON:")
            for obj in json_data['objects']:
                print(f"- {obj['label']}")
        except Exception as e:
            print(f"Error loading JSON file: {e}")
    
    # Add colorbar
    ax3 = fig.add_subplot(gs[2])
    cbar = plt.colorbar(im, cax=ax3)
    cbar.set_label('Class ID')
    
    plt.tight_layout()
    
    # If running on remote server and no save path provided, create one
    if is_remote_server() and not save_path:
        save_path = "visualization_output.png"
        print(f"Running on remote server. Will save visualization to: {save_path}")
    
    if save_path:
        print(f"Saving visualization to: {save_path}")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print("Visualization saved successfully!")
    else:
        print("Displaying visualization...")
        plt.show()
        print("Visualization displayed successfully!")

def visualize_batch_image_and_gtFine(image_batch, gt_batch, save_path=None):
    """
    Visualize a batch of input images and their corresponding ground truth masks.
    
    Args:
        image_batch (torch.Tensor): Batch of input images [B, C, H, W]
        gt_batch (torch.Tensor): Batch of ground truth masks [B, H, W]
        save_path (str, optional): Path to save the visualization. If None, will display the image.
    """
    # Convert to numpy if tensor
    if isinstance(image_batch, torch.Tensor):
        image_batch = image_batch.cpu().numpy()
    if isinstance(gt_batch, torch.Tensor):
        gt_batch = gt_batch.cpu().numpy()
    
    # If image is in [B, C, H, W] format, convert to [B, H, W, C]
    if image_batch.shape[1] == 3:
        image_batch = np.transpose(image_batch, (0, 2, 3, 1))
    
    # Cityscapes color mapping (RGB values for each class)
    cityscapes_colors = {
        0: [0, 0, 0],        # unlabeled
        1: [70, 70, 70],     # road
        2: [100, 40, 40],    # sidewalk
        3: [55, 90, 80],     # building
        4: [220, 20, 60],    # wall
        5: [153, 153, 153],  # fence
        6: [157, 234, 50],   # pole
        7: [128, 64, 128],   # traffic light
        8: [244, 35, 232],   # traffic sign
        9: [107, 142, 35],   # vegetation
        10: [0, 0, 142],     # terrain
        11: [102, 102, 156], # sky
        12: [220, 220, 0],   # person
        13: [70, 130, 180],  # rider
        14: [81, 0, 81],     # car
        15: [150, 100, 100], # truck
        16: [230, 150, 140], # bus
        17: [180, 165, 180], # train
        18: [250, 170, 30],  # motorcycle
        19: [110, 190, 160], # bicycle
        255: [0, 0, 0]       # ignore
    }
    
    batch_size = gt_batch.shape[0]
    fig, axes = plt.subplots(2, batch_size, figsize=(4*batch_size, 8))
    
    for i in range(batch_size):
        # Plot input image
        axes[0, i].imshow(image_batch[i])
        axes[0, i].set_title(f'Input Image {i+1}')
        axes[0, i].axis('off')
        
        # Create RGB visualization for ground truth
        h, w = gt_batch[i].shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Map each class to its color
        for class_id, color in cityscapes_colors.items():
            rgb_mask[gt_batch[i] == class_id] = color
        
        # Plot ground truth mask
        axes[1, i].imshow(rgb_mask)
        axes[1, i].set_title(f'Ground Truth {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Cityscapes ground truth segmentation masks')
    parser.add_argument('--mode', type=str, choices=['single', 'pair', 'batch'], required=True,
                      help='Visualization mode: single (gt only), pair (image + gt), or batch')
    parser.add_argument('--gt_path', type=str, required=True,
                      help='Path to ground truth mask(s)')
    parser.add_argument('--image_path', type=str,
                      help='Path to input image(s) (required for pair mode)')
    parser.add_argument('--json_path', type=str,
                      help='Path to gtFine_polygons.json file (optional)')
    parser.add_argument('--save_path', type=str,
                      help='Path to save visualization (optional)')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Number of images to visualize in batch mode (default: 4)')
    
    args = parser.parse_args()
    
    print(f"\nStarting visualization in {args.mode} mode...")
    if is_remote_server():
        print("Running on remote server - visualizations will be saved to files")
    
    if args.mode == 'single':
        visualize_gtFine(args.gt_path, args.save_path)
    elif args.mode == 'pair':
        if not args.image_path:
            raise ValueError("image_path is required for pair mode")
        visualize_image_and_gtFine(args.image_path, args.gt_path, args.json_path, args.save_path)
    elif args.mode == 'batch':
        # For batch mode, gt_path should be a directory containing multiple masks
        if not os.path.isdir(args.gt_path):
            raise ValueError("gt_path must be a directory for batch mode")
        if not args.image_path or not os.path.isdir(args.image_path):
            raise ValueError("image_path must be a directory for batch mode")
            
        # Get list of files
        gt_files = sorted([f for f in os.listdir(args.gt_path) if f.endswith('_gtFine_labelIds.png')])[:args.batch_size]
        image_files = sorted([f for f in os.listdir(args.image_path) if f.endswith('_leftImg8bit.png')])[:args.batch_size]
        
        print(f"Found {len(gt_files)} ground truth files and {len(image_files)} image files")
        
        # Load images and masks
        images = []
        masks = []
        for img_file, gt_file in zip(image_files, gt_files):
            print(f"\nProcessing pair: {img_file} - {gt_file}")
            img = np.array(Image.open(os.path.join(args.image_path, img_file)))
            gt = np.array(Image.open(os.path.join(args.gt_path, gt_file)))
            images.append(img)
            masks.append(gt)
        
        # Convert to numpy arrays
        image_batch = np.stack(images)
        gt_batch = np.stack(masks)
        
        print(f"\nCreated batches with shapes: images {image_batch.shape}, masks {gt_batch.shape}")
        visualize_batch_image_and_gtFine(image_batch, gt_batch, args.save_path)
    
    print("\nVisualization completed!") 