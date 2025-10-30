import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from decord import VideoReader, cpu
import sys
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import cv2
from PIL import Image

# We need to import the I3D model definition from the cloned repo
# Add the repo to our Python path (assuming it's in the same dir)
sys.path.append('pytorch-i3d')
try:
    from pytorch_i3d import InceptionI3d
except ImportError:
    print("Error: Could not import InceptionI3d.")
    print("Please make sure the 'pytorch-i3d' repository is in the same directory as this script.")
    exit(1)

# Set up global constants
INPUT_SIZE = 224


# --- STAGE 1 (NEW): RGB FEATURE EXTRACTION (READS FROM IMG FOLDERS) ---

def run_rgb_extraction(args, device):
    """Pipeline Stage 2: Extract I3D features from RGB image frames."""
    print("\n--- [Starting Stage 2: RGB Feature Extraction] ---")
    
    i3d = InceptionI3d(400, in_channels=3)
    print(f"Loading RGB model weights from: {args.model_rgb}")
    i3d.load_state_dict(torch.load(args.model_rgb, map_location=device))
    i3d.to(device)
    i3d.eval()

    # Define the transforms
    transform = Compose([
        ToTensor(), # Scales [0, 255] to [0.0, 1.0]
        Resize(256),
        CenterCrop(INPUT_SIZE),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    os.makedirs(args.rgb_feat_dir, exist_ok=True)

    # Read from the output_data_dir (where frames were saved)
    video_folders = [d for d in os.listdir(args.output_data_dir) 
                     if os.path.isdir(os.path.join(args.output_data_dir, d))]
    
    print(f"Found {len(video_folders)} image directories to process...")

    for folder_name in tqdm(video_folders, desc="Extracting RGB Features"):
        rgb_folder_path = os.path.join(args.output_data_dir, folder_name)
        output_path = os.path.join(args.rgb_feat_dir, f"{folder_name}.npy")
        
        if os.path.exists(output_path) and not args.overwrite:
            continue
        
        img_dir = os.path.join(rgb_folder_path, 'img')

        if not os.path.isdir(img_dir):
            print(f"  - WARNING: Skipping {folder_name} ('img' folder not found)")
            continue

        img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        img_files.extend(sorted(glob.glob(os.path.join(img_dir, '*.png'))))
        
        total_frames = len(img_files)

        if total_frames <= args.chunk_size:
            print(f"  - WARNING: Skipping {folder_name} (Frames {total_frames} <= chunk_size {args.chunk_size})")
            continue

        # Create all start indices for chunks
        start_indices = list(range(0, total_frames - args.chunk_size + 1, args.stride))
        if not start_indices:
            start_indices = [0] # At least one chunk
        
        all_features = []

        # Process in batches of start_indices
        with torch.no_grad():
            for i in range(0, len(start_indices), args.batch_size):
                batch_start_time = i
                batch_end_time = min(i + args.batch_size, len(start_indices))
                
                current_batch_indices = start_indices[batch_start_time:batch_end_time]
                num_chunks_in_batch = len(current_batch_indices)
                
                # Allocate tensor for this batch of chunks
                batch_tensor = torch.zeros(num_chunks_in_batch, 3, args.chunk_size, INPUT_SIZE, INPUT_SIZE)

                # Assemble chunks from the loaded frames
                for j, start_idx in enumerate(current_batch_indices):
                    rgb_chunk = torch.zeros(args.chunk_size, 3, INPUT_SIZE, INPUT_SIZE)
                    
                    for k in range(args.chunk_size):
                        frame_idx = start_idx + k
                        
                        try:
                            img = Image.open(img_files[frame_idx]).convert('RGB')
                            rgb_chunk[k] = transform(img)
                        except Exception as e:
                            print(f"  - ERROR: Failed loading {img_files[frame_idx]}. {e}")
                            break
                    
                    batch_tensor[j] = rgb_chunk.permute(1, 0, 2, 3) # (C, T, H, W)

                # Send to GPU and get features
                features = i3d.extract_features(batch_tensor.to(device))
                features_batch_np = features.squeeze().cpu().numpy()
                
                if features_batch_np.ndim == 1:
                    features_batch_np = np.expand_dims(features_batch_np, axis=0)
                all_features.append(features_batch_np)

        if not all_features:
            print(f"  - WARNING: No features extracted for {folder_name}.")
            continue
            
        features_np = np.concatenate(all_features, axis=0) # Shape: (T, 1024)
        features_np = features_np.T # Shape: (1024, T)
        np.save(output_path, features_np)

    print("--- [Stage 2 Complete] ---")



# --- STAGE 2: FRAME & FLOW GENERATION ---

# --- MODIFIED: This function now saves frames AND flow in the correct structure ---
def compute_and_save_flow_task(task_args):
    """Wrapper: Computes TV-L1 flow AND saves raw frames."""
    video_path, output_dir_flow_x, output_dir_flow_y, output_dir_img, bound = task_args
    try:
        cap = cv2.VideoCapture(video_path)
        ret, prev_frame_bgr = cap.read()
        if not ret:
            cap.release()
            return (video_path, "Error: Could not read first frame")
            
        prev_frame = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
        tvl1 = cv2.optflow.createOptFlow_DualTVL1()
        frame_idx = 0
        
        # Save the first frame (index 0, saved as ..._00001.jpg)
        save_path_img = os.path.join(output_dir_img, f"frame_{frame_idx+1:05d}.jpg")
        cv2.imwrite(save_path_img, prev_frame_bgr)
        
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            frame_idx += 1 # Frame index is now 1 (for the 2nd frame)
            
            # Save the current frame (index 1, saved as ..._00002.jpg)
            save_path_img = os.path.join(output_dir_img, f"frame_{frame_idx+1:05d}.jpg")
            cv2.imwrite(save_path_img, frame_bgr)
            
            # --- Flow calculation (between frame_idx-1 and frame_idx) ---
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            flow = tvl1.calc(prev_frame, frame, None)
            
            flow_u = np.clip(flow[..., 0], -bound, bound)
            flow_v = np.clip(flow[..., 1], -bound, bound)
            
            flow_u = ((flow_u + bound) / (2 * bound) * 255).astype(np.uint8)
            flow_v = ((flow_v + bound) / (2 * bound) * 255).astype(np.uint8)
            
            # Save flow (index 1, for flow 0->1, saved as ..._00001.jpg)
            save_path_u = os.path.join(output_dir_flow_x, f"flow_x_{frame_idx:05d}.jpg")
            save_path_v = os.path.join(output_dir_flow_y, f"flow_y_{frame_idx:05d}.jpg")
            
            cv2.imwrite(save_path_u, flow_u)
            cv2.imwrite(save_path_v, flow_v)
            
            prev_frame = frame
            
        cap.release()
        return (video_path, f"Finished ({frame_idx + 1} frames, {frame_idx} flows)")
        
    except Exception as e:
        return (video_path, f"Error: {e}")

# --- MODIFIED: This function now creates the img/flow_x/flow_y structure ---
def run_flow_generation(args):
    """Pipeline Stage 2: Generate optical flow images AND raw frames."""
    print("\n--- [Starting Stage 2: Frame & Flow Generation] ---")
    
    video_exts = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    video_paths = []
    for root, _, files in os.walk(args.video_dir):
        for file in files:
            if file.lower().endswith(video_exts):
                video_paths.append(os.path.join(root, file))

    if not video_paths:
        print(f"Error: No videos found in {args.video_dir}")
        return

    print(f"Found {len(video_paths)} videos. Starting frame/flow generation...")

    tasks = []
    print("Scanning for videos to process...")
    for video_path in tqdm(video_paths, desc="Scanning videos"):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # --- MODIFIED: Define all output paths ---
        output_base_dir = os.path.join(args.output_data_dir, base_name) # e.g., mentahan/video1
        
        output_dir_flow_x = os.path.join(output_base_dir, 'flow_x')
        output_dir_flow_y = os.path.join(output_base_dir, 'flow_y')
        output_dir_img = os.path.join(output_base_dir, 'img')
        
        # --- MODIFIED: Check for overwrite ---
        if os.path.exists(output_base_dir) and not args.overwrite:
            check_dirs = [output_dir_flow_x, output_dir_flow_y, output_dir_img]
            # Skip if all 3 dirs exist and are not empty
            if all(os.path.exists(d) and len(os.listdir(d)) > 0 for d in check_dirs):
                continue
        
        # Create all directories
        os.makedirs(output_dir_flow_x, exist_ok=True)
        os.makedirs(output_dir_flow_y, exist_ok=True)
        os.makedirs(output_dir_img, exist_ok=True)
        
        # Pass all paths to the task
        task_args = (video_path, output_dir_flow_x, output_dir_flow_y, output_dir_img, args.flow_bound)
        tasks.append(task_args)

    if not tasks:
        print("No new videos to process (all frame/flow outputs may already exist).")
        return

    print(f"Starting processing pool with {args.num_workers} workers for {len(tasks)} videos...")
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(executor.map(compute_and_save_flow_task, tasks), 
                            total=len(tasks), 
                            desc="Processing videos"))
    
    print("\n--- [Stage 2 Complete] ---")
    success_count = sum(1 for _, status in results if "Finished" in status)
    print(f"Summary: {success_count} videos successfully processed.")


# --- STAGE 3: FLOW FEATURE EXTRACTION ---

# --- MODIFIED: This function now reads from the new flow_x/flow_y subfolders ---
def load_and_process_flow_chunks(flow_dir, chunk_size, stride):
    """
    Loads flow images from a directory (e.g., .../video_name/)
    which contains 'flow_x' and 'flow_y' subfolders.
    """
    # --- MODIFIED: Point glob inside 'flow_x' and 'flow_y' subdirs ---
    flow_x_files = sorted(glob.glob(os.path.join(flow_dir, 'flow_x', '*.jpg')))
    flow_x_files.extend(sorted(glob.glob(os.path.join(flow_dir, 'flow_x', '*.png'))))
    
    flow_y_files = sorted(glob.glob(os.path.join(flow_dir, 'flow_y', '*.jpg')))
    flow_y_files.extend(sorted(glob.glob(os.path.join(flow_dir, 'flow_y', '*.png'))))
    # ----------------------------------------------------------------
    
    total_frames = min(len(flow_x_files), len(flow_y_files))

    if total_frames == 0:
        return None, "No flow images found in flow_x/flow_y subfolders"

    start_indices = list(range(0, total_frames - chunk_size + 1, stride))
    
    if not start_indices and total_frames > 0:
        start_indices = [0]
    elif not start_indices:
        return None, "Not enough frames for one chunk"

    num_chunks = len(start_indices)

    transform = Compose([
        ToTensor(), # Scales [0, 255] to [0.0, 1.0]
        Resize(256),
        CenterCrop(INPUT_SIZE),
        Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]) # Normalizes [0,1] to [-1,1]
    ])
    
    transformed_batch = torch.zeros(num_chunks, 2, chunk_size, INPUT_SIZE, INPUT_SIZE)
    chunk_idx = 0
    
    for start in start_indices:
        flow_chunk = torch.zeros(chunk_size, 2, INPUT_SIZE, INPUT_SIZE)
        
        for i in range(chunk_size):
            frame_idx = start + i
            if frame_idx >= total_frames:
                frame_idx = total_frames - 1 
            
            try:
                img_x = Image.open(flow_x_files[frame_idx]).convert('L')
                img_y = Image.open(flow_y_files[frame_idx]).convert('L')
                
                img_flow = np.stack([np.array(img_x), np.array(img_y)], axis=2)
                img_flow_pil = Image.fromarray(img_flow, mode='LA')
                
                transformed_flow = transform(img_flow_pil)
                flow_chunk[i] = transformed_flow
            except Exception as e:
                return None, f"Error loading image frame {frame_idx}: {e}"

        transformed_batch[chunk_idx] = flow_chunk.permute(1, 0, 2, 3)
        chunk_idx += 1

    return transformed_batch, f"Success ({num_chunks} chunks)"

# --- STAGE 3: FLOW FEATURE EXTRACTION (NEW MEMORY-EFFICIENT VERSION) ---

def run_flow_feature_extraction(args, device):
    """Pipeline Stage 3: Extract I3D features from flow (Memory-Efficient)."""
    print("\n--- [Starting Stage 3: Flow Feature Extraction] ---")
    
    i3d = InceptionI3d(400, in_channels=2)
    print(f"Loading Flow model weights from: {args.model_flow}")
    i3d.load_state_dict(torch.load(args.model_flow, map_location=device))
    i3d.to(device)
    i3d.eval()

    # Define the transforms
    transform = Compose([
        ToTensor(), # Scales [0, 255] to [0.0, 1.0]
        Resize(256),
        CenterCrop(INPUT_SIZE),
        Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]) # Normalizes [0,1] to [-1,1]
    ])

    os.makedirs(args.flow_feat_dir, exist_ok=True)

    flow_folders = [d for d in os.listdir(args.output_data_dir) 
                    if os.path.isdir(os.path.join(args.output_data_dir, d))]
    
    print(f"Found {len(flow_folders)} flow image directories to process...")

    for folder_name in tqdm(flow_folders, desc="Extracting Flow Features"):
        flow_folder_path = os.path.join(args.output_data_dir, folder_name)
        output_path = os.path.join(args.flow_feat_dir, f"{folder_name}.npy")
        
        if os.path.exists(output_path) and not args.overwrite:
            continue
        
        flow_x_dir = os.path.join(flow_folder_path, 'flow_x')
        flow_y_dir = os.path.join(flow_folder_path, 'flow_y')

        if not (os.path.isdir(flow_x_dir) and os.path.isdir(flow_y_dir)):
            print(f"  - WARNING: Skipping {folder_name} ('flow_x' or 'flow_y' folder not found)")
            continue

        flow_x_files = sorted(glob.glob(os.path.join(flow_x_dir, '*.jpg')))
        flow_y_files = sorted(glob.glob(os.path.join(flow_y_dir, '*.jpg')))
        
        total_frames = min(len(flow_x_files), len(flow_y_files))

        if total_frames <= args.chunk_size:
            print(f"  - WARNING: Skipping {folder_name} (Frames {total_frames} <= chunk_size {args.chunk_size})")
            continue

        # Create all start indices for chunks
        start_indices = list(range(0, total_frames - args.chunk_size + 1, args.stride))
        if not start_indices:
            start_indices = [0] # At least one chunk
        
        all_features = []

        # Process in batches of start_indices
        with torch.no_grad():
            for i in range(0, len(start_indices), args.batch_size):
                batch_start_time = i
                batch_end_time = min(i + args.batch_size, len(start_indices))
                
                current_batch_indices = start_indices[batch_start_time:batch_end_time]
                num_chunks_in_batch = len(current_batch_indices)
                
                # Allocate tensor for this batch of chunks
                batch_tensor = torch.zeros(num_chunks_in_batch, 2, args.chunk_size, INPUT_SIZE, INPUT_SIZE)

                # Assemble chunks from the loaded frames
                for j, start_idx in enumerate(current_batch_indices):
                    flow_chunk = torch.zeros(args.chunk_size, 2, INPUT_SIZE, INPUT_SIZE)
                    
                    for k in range(args.chunk_size):
                        frame_idx = start_idx + k
                        
                        try:
                            img_x = Image.open(flow_x_files[frame_idx]).convert('L')
                            img_y = Image.open(flow_y_files[frame_idx]).convert('L')
                            
                            img_flow = np.stack([np.array(img_x), np.array(img_y)], axis=2)
                            img_flow_pil = Image.fromarray(img_flow, mode='LA')
                            
                            flow_chunk[k] = transform(img_flow_pil)
                        except Exception as e:
                            print(f"  - ERROR: Failed loading {flow_x_files[frame_idx]}. {e}")
                            break
                    
                    batch_tensor[j] = flow_chunk.permute(1, 0, 2, 3) # (C, T, H, W)

                # Send to GPU and get features
                features = i3d.extract_features(batch_tensor.to(device))
                features_batch_np = features.squeeze().cpu().numpy()
                
                if features_batch_np.ndim == 1:
                    features_batch_np = np.expand_dims(features_batch_np, axis=0)
                all_features.append(features_batch_np)

        if not all_features:
            print(f"  - WARNING: No features extracted for {folder_name}.")
            continue
            
        features_np = np.concatenate(all_features, axis=0) # Shape: (T, 1024)
        features_np = features_np.T # Shape: (1024, T)
        np.save(output_path, features_np)

    print("--- [Stage 3 Complete] ---")


# --- STAGE 4: FEATURE COMBINATION ---
# --- (No changes needed for this stage) ---

def run_feature_combination(args):
    """Pipeline Stage 4: Combine RGB and Flow features."""
    print("\n--- [Starting Stage 4: Feature Combination] ---")
    
    os.makedirs(args.combined_feat_dir, exist_ok=True)
    
    rgb_files = glob.glob(os.path.join(args.rgb_feat_dir, '*.npy'))
    
    if not rgb_files:
        print(f"Error: No .npy files found in RGB directory: {args.rgb_feat_dir}")
        return

    print(f"Found {len(rgb_files)} RGB files. Combining with Flow...")
    
    total_combined = 0
    total_skipped = 0
    
    for rgb_path in tqdm(rgb_files, desc="Combining Features"):
        basename = os.path.basename(rgb_path)
        flow_path = os.path.join(args.flow_feat_dir, basename)
        
        if not os.path.exists(flow_path):
            print(f"  - WARNING: Skipping {basename}. No matching flow file.")
            total_skipped += 1
            continue
            
        try:
            rgb_feats = np.load(rgb_path)  # Shape (1024, T1)
            flow_feats = np.load(flow_path) # Shape (1024, T2)
        except Exception as e:
            print(f"  - ERROR: Could not load {basename}. {e}")
            total_skipped += 1
            continue

        # --- FIX: Check (D, T) format ---
        if rgb_feats.shape[0] != 1024 or flow_feats.shape[0] != 1024:
            print(f"  - WARNING: Skipping {basename}. Incorrect feature dim.")
            print(f"    RGB shape: {rgb_feats.shape}, Flow shape: {flow_feats.shape}")
            total_skipped += 1
            continue
        
        # Align temporally by truncating to the minimum T
        min_t = min(rgb_feats.shape[1], flow_feats.shape[1])
        
        rgb_feats = rgb_feats[:, :min_t]
        flow_feats = flow_feats[:, :min_t]
        
        # Concatenate along the feature dimension (axis=0)
        # (1024, T) + (1024, T) -> (2048, T)
        combined_feats = np.concatenate([rgb_feats, flow_feats], axis=0)
        
        output_path = os.path.join(args.combined_feat_dir, basename)
        np.save(output_path, combined_feats)
        total_combined += 1

    print("\n--- [Stage 4 Complete] ---")
    print(f"Successfully combined: {total_combined} files.")
    print(f"Skipped: {total_skipped} files.")


# --- MAIN ---

def main(args):
    # Setup Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Define all output paths from the single output_dir
    args.output_data_dir = os.path.join(args.output_dir, 'raw_data')
    args.rgb_feat_dir = os.path.join(args.output_dir, 'features', 'rgb')
    args.flow_feat_dir = os.path.join(args.output_dir, 'features', 'flow')
    args.combined_feat_dir = os.path.join(args.output_dir, 'features', 'combined_2048')
    
    # --- MODIFIED: Re-ordered the pipeline ---
    
    # Stage 1: Preprocessing (Read video, save frames/flow)
    if args.run_all or args.run_preprocessing:
        # This is the old 'run_flow_generation'
        run_flow_generation(args) 
        
    # Stage 2: RGB Extraction (Read 'img' frames)
    if args.run_all or args.run_rgb_extraction:
        # This is the new 'run_rgb_extraction'
        run_rgb_extraction(args, device)
        
    # Stage 3: Flow Extraction (Read 'flow_x/y' frames)
    if args.run_all or args.run_flow_extraction:
        # This is the old 'run_flow_feature_extraction'
        run_flow_feature_extraction(args, device)
        
    # Stage 4: Combine Features
    if args.run_all or args.run_combine:
        run_feature_combination(args)
        
    print("\n--- Pipeline Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Full I3D Feature Extraction Pipeline (RGB + Flow)")
    
    # --- General Paths ---
    parser.add_argument('--video_dir', type=str, default='videos',
                        help="Directory containing your .mp4 videos.")
    parser.add_argument('--model_rgb', type=str, default='rgb_imagenet.pt',
                        help="Path to the pre-trained I3D RGB model (.pt file).")
    parser.add_argument('--model_flow', type=str, default='flow_imagenet.pt',
                        help="Path to the pre-trained I3D Flow model (.pt file).")

    # --- Directory Outputs (Defaults) ---
    # --- FIXED: Use the single output_dir ---
    parser.add_argument('--output_dir', type=str, default='pipeline_output',
                        help="Main directory to save all outputs (raw data and features).")

    # --- Controls ---
    parser.add_argument('--run-all', action='store_true', help="Run all 4 stages in sequence.")
    parser.add_argument('--run-preprocessing', action='store_true', help="Stage 1: Read video, save frames/flow.")
    parser.add_argument('--run-rgb-extraction', action='store_true', help="Stage 2: Extract RGB features from frames.")
    parser.add_argument('--run-flow-extraction', action='store_true', help="Stage 3: Extract Flow features from frames.")
    parser.add_argument('--run-combine', action='store_true', help="Stage 4: Combine RGB and Flow features.")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing feature/flow files.")

    # --- Extraction Parameters ---
    parser.add_argument('--chunk_size', type=int, default=16,
                        help="Number of frames in each chunk (T).")
    parser.add_argument('--stride', type=int, default=16,
                        help="Number of frames to slide the window. Default 16 (non-overlapping).")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Number of chunks to process at a time (to prevent OOM).")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to use ('cpu', 'cuda'). Default: 'cuda'")
    
    # --- Flow Gen Parameters ---
    parser.add_argument('--flow_bound', type=int, default=20,
                        help="Optical flow bound, 20 is standard for I3D.")
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                        help="Number of CPU cores to use for flow generation.")

    args = parser.parse_args()

    # --- FIXED: Added parser.print_help() and exit(1) ---
    if not (args.run_all or args.run_preprocessing or args.run_rgb_extraction or args.run_flow_extraction or args.run_combine):
        print("Error: No action specified. Please add one of:")
        print("  --run-all")
        print("  --run-preprocessing")
        print("  --run-rgb-extraction")
        print("  --run-flow-extraction")
        print("  --run-combine")
        parser.print_help()
        exit(1)
        
    main(args)