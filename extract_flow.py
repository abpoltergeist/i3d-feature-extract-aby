import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from PIL import Image
import sys
import glob

# We need to import the I3D model definition from the cloned repo
sys.path.append('pytorch-i3d') # Assumes pytorch-i3d is in the same folder
try:
    from pytorch_i3d import InceptionI3d
except ImportError:
    print("Error: Could not import InceptionI3d.")
    print("Please make sure the 'pytorch-i3d' repository is in the same directory.")
    exit(1)

# Set up global constants
INPUT_SIZE = 224

def load_and_process_flow_chunks(flow_dir, chunk_size, stride):
    """
    Loads flow images from a directory, processes them in overlapping chunks,
    and returns a batch of pre-processed chunks ready for the I3D model.
    
    Returns a tensor of shape (N, C, T, H, W), where:
    N = number of chunks
    C = channels (2 for flow: u and v)
    T = chunk_size
    H, W = INPUT_SIZE
    """
    # Find all flow_x and flow_y files, sorted
    flow_x_files = sorted(glob.glob(os.path.join(flow_dir, 'flow_x_*.jpg')))
    flow_y_files = sorted(glob.glob(os.path.join(flow_dir, 'flow_y_*.jpg')))
    
    total_frames = min(len(flow_x_files), len(flow_y_files))

    if total_frames == 0:
        print(f"  Warning: No flow images found in {flow_dir}.")
        return None

    # 1. Create start indices for each chunk
    start_indices = list(range(0, total_frames - chunk_size + 1, stride))
    
    if not start_indices and total_frames > 0:
        start_indices = [0]
        num_to_pad = chunk_size - total_frames
        indices = list(range(total_frames)) + [total_frames - 1] * num_to_pad
    else:
        indices = []
        for start in start_indices:
            indices.extend(list(range(start, start + chunk_size)))

    num_chunks = len(start_indices)
    if num_chunks == 0:
        print(f"  Warning: Not enough frames to make a single chunk in {flow_dir}.")
        return None

    # 5. Apply transformations
    # I3D flow model expects 2-channel input, normalized with mean/std of 0.5
    transform = Compose([
        ToTensor(), # Scales images from [0, 255] to [0.0, 1.0]
        Resize(256),
        CenterCrop(INPUT_SIZE),
        Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]) # Normalizes [0,1] to [-1,1]
    ])
    
    # Pre-allocate tensor for transformed chunks
    # (N, C, T, H, W) - I3D expects C=2 first, then T
    transformed_batch = torch.zeros(num_chunks, 2, chunk_size, INPUT_SIZE, INPUT_SIZE)

    chunk_idx = 0
    for start in start_indices:
        flow_chunk = torch.zeros(chunk_size, 2, INPUT_SIZE, INPUT_SIZE)
        
        for i in range(chunk_size):
            frame_idx = start + i
            # Handle padding
            if frame_idx >= total_frames:
                frame_idx = total_frames - 1 
            
            # Load flow_x and flow_y images
            img_x = Image.open(flow_x_files[frame_idx]).convert('L') # 'L' = grayscale
            img_y = Image.open(flow_y_files[frame_idx]).convert('L')
            
            # Stack them into a 2-channel image (H, W, 2)
            # We must do this *before* ToTensor
            img_flow = np.stack([np.array(img_x), np.array(img_y)], axis=2) # Shape (H, W, 2)
            img_flow_pil = Image.fromarray(img_flow, mode='LA') # 'LA' is Luma+Alpha, acts as 2-channel
            
            # Apply transforms
            # ToTensor permutes (H, W, C) to (C, H, W)
            transformed_flow = transform(img_flow_pil) # Shape (2, H_out, W_out)
            
            flow_chunk[i] = transformed_flow

        # Permute chunk from (T, C, H, W) to (C, T, H, W)
        transformed_batch[chunk_idx] = flow_chunk.permute(1, 0, 2, 3)
        chunk_idx += 1

    return transformed_batch

def main(args):
    # ---- 1. Set up Device ----
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # ---- 2. Load Model ----
    # CRITICAL: in_channels=2 for flow
    i3d = InceptionI3d(400, in_channels=2)
    
    print(f"Loading model weights from: {args.model_path}")
    i3d.load_state_dict(torch.load(args.model_path, map_location=device))
    i3d.to(device)
    i3d.eval()

    # ---- 3. Create Output Directory ----
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 4. Find Video *Flow Folders* ----
    # We look for subdirectories in the flow_dir
    flow_folders = [d for d in os.listdir(args.flow_dir) 
                    if os.path.isdir(os.path.join(args.flow_dir, d))]
    
    print(f"Found {len(flow_folders)} flow directories to process...")

    # ---- 5. Start Extraction Loop ----
    for folder_name in flow_folders:
        flow_folder_path = os.path.join(args.flow_dir, folder_name)
        output_path = os.path.join(args.output_dir, f"{folder_name}.npy")
        
        if os.path.exists(output_path):
            print(f"Skipping {folder_name}, feature file already exists.")
            continue
            
        print(f"Processing: {folder_name}")

        # Load flow frames
        video_batch_tensor = load_and_process_flow_chunks(
            flow_folder_path, 
            chunk_size=args.chunk_size, 
            stride=args.stride
        )
        
        if video_batch_tensor is None:
            print(f"  Failed to load frames for {folder_name}. Skipping.")
            continue
            
        # ---- 6. Extract Features in Mini-Batches ----
        all_features = []
        num_chunks = video_batch_tensor.shape[0]
        
        with torch.no_grad():
            for i in range(0, num_chunks, args.batch_size):
                batch_start = i
                batch_end = min(i + args.batch_size, num_chunks)
                mini_batch = video_batch_tensor[batch_start:batch_end].to(device)
                
                features = i3d.extract_features(mini_batch)
                features_batch_np = features.squeeze().cpu().numpy()
                
                if features_batch_np.ndim == 1:
                    features_batch_np = np.expand_dims(features_batch_np, axis=0)
                all_features.append(features_batch_np)

        if not all_features:
            print("  No features extracted (list is empty). Skipping.")
            continue
            
        features_np = np.concatenate(all_features, axis=0) # Shape: (N, 1024)
            
        if features_np.ndim == 1:
            features_np = np.expand_dims(features_np, axis=0)

        # ---- 7. Save Features ----
        np.save(output_path, features_np)
        print(f"  Saved features ({features_np.shape}) to {output_path}")

    print("--- Flow feature extraction complete! ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generic I3D FLOW Feature Extractor")
    
    parser.add_argument('--flow_dir', type=str, required=True,
                        help="Directory containing subfolders of flow images.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory where .npy feature files will be saved.")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the pre-trained I3D flow model (flow_imagenet.pt).")
    
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to use ('cpu', 'cuda'). Default: 'cuda'")
    parser.add_argument('--chunk_size', type=int, default=16,
                        help="Number of frames in each chunk (T).")
    parser.add_argument('--stride', type=int, default=16,
                        help="Number of frames to slide the window. Default 16 (non-overlapping).")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Number of chunks to process at a time (to prevent OOM).")

    args = parser.parse_args()
    
    # We need PIL and OpenCV for this script
    try:
        import PIL
        import cv2
    except ImportError:
        print("Please install 'Pillow' and 'opencv-python' libraries: pip install Pillow opencv-python")
        exit(1)
        
    main(args)
