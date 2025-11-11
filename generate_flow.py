import cv2
import numpy as np
import os
import argparse
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm # Import tqdm

def compute_and_save_flow(task_args):
    """
    Computes TV-L1 optical flow for a single video and saves
    the flow images (u, v) in the output directory.
    Now takes a single tuple of arguments for executor.map.
    """
    video_path, output_dir, bound = task_args
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Read the first frame
        ret, prev_frame_bgr = cap.read()
        if not ret:
            # This error is handled by returning a status
            cap.release()
            return (video_path, "Error: Could not read first frame")
            
        prev_frame = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Create the TV-L1 optical flow object
        tvl1 = cv2.optflow.createOptFlow_DualTVL1()
        
        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break # End of video
                
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            # --- Compute Flow ---
            flow = tvl1.calc(prev_frame, frame, None) # Shape (H, W, 2)
            
            # flow[..., 0] is 'u' (horizontal)
            # flow[..., 1] is 'v' (vertical)
            
            # --- Save Flow Images ---
            # This is the standard way I3D-Flow models expect the data
            
            # 1. Normalize flow from [-bound, +bound] to [0, 255]
            flow_u = np.clip(flow[..., 0], -bound, bound)
            flow_v = np.clip(flow[..., 1], -bound, bound)
            
            flow_u = ((flow_u + bound) / (2 * bound) * 255).astype(np.uint8)
            flow_v = ((flow_v + bound) / (2 * bound) * 255).astype(np.uint8)
            
            # 2. Save as JPEG
            # File format is 'flow_x_00001.jpg', 'flow_y_00001.jpg'
            # Note: frame_idx starts at 0, so we save as idx+1
            save_path_u = os.path.join(output_dir, f"flow_x_{frame_idx+1:05d}.jpg")
            save_path_v = os.path.join(output_dir, f"flow_y_{frame_idx+1:05d}.jpg")
            
            cv2.imwrite(save_path_u, flow_u)
            cv2.imwrite(save_path_v, flow_v)
            
            # Update for next loop
            prev_frame = frame
            frame_idx += 1
            
        cap.release()
        return (video_path, f"Finished ({frame_idx} frames)")
        
    except Exception as e:
        return (video_path, f"Error: {e}")

def main(args):
    # Find all videos
    video_paths = glob.glob(os.path.join(args.video_dir, '*.mp4'))
    video_paths += glob.glob(os.path.join(args.video_dir, '*.avi'))
    video_paths += glob.glob(os.path.join(args.video_dir, '*.mov'))
    
    if not video_paths:
        print(f"Error: No videos found in {args.video_dir}")
        return

    print(f"Found {len(video_paths)} videos. Starting flow generation...")

    # Create a list of arguments for the pool
    tasks = []
    print("Scanning for videos to process...")
    # Add a tqdm progress bar for scanning videos
    for video_path in tqdm(video_paths, desc="Scanning videos"):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_subdir = os.path.join(args.output_dir, base_name)
        
        if os.path.exists(output_subdir) and len(os.listdir(output_subdir)) > 0:
            # Skipping, so don't add to tasks
            continue
            
        os.makedirs(output_subdir, exist_ok=True)
        tasks.append((video_path, output_subdir, args.bound))

    if not tasks:
        print("No new videos to process (all outputs may already exist).")
        return

    # --- Run in Parallel ---
    # This process is slow, so we use multiple CPU cores
    print(f"Starting processing pool with {args.num_workers} workers for {len(tasks)} videos...")
    
    # Use tqdm with ProcessPoolExecutor
    # We wrap the executor.map call with tqdm
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # executor.map now returns results, which we can iterate over with tqdm
        # We also need to update compute_and_save_flow to accept a single arg
        results = list(tqdm(executor.map(compute_and_save_flow, tasks), 
                            total=len(tasks), 
                            desc="Processing videos"))

    print("\n--- Python flow generation complete! ---")
    
    # Optional: Print summary of results
    success_count = 0
    error_count = 0
    for video, status in results:
        if "Finished" in status:
            success_count += 1
        else:
            error_count += 1
            print(f"  - FAILED: {os.path.basename(video)} ({status})")
            
    print(f"Summary: {success_count} succeeded, {error_count} failed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate TV-L1 Optical Flow images using Python/OpenCV")
    
    parser.add_argument('--video_dir', type=str, required=True,
                        help="Directory containing your .mp4 videos.")
    # FIX: Corrected parser.add_Falser to parser.add_argument
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Main directory to save flow subfolders (e.g., 'my_flow_output').")
    parser.add_argument('--bound', type=int, default=20,
                        help="Optical flow bound, 20 is standard for I3D.")
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                        help="Number of CPU cores to use in parallel.")
    
    args = parser.parse_args()
    
    # FIX: Removed redundant try/except block for tqdm.
    # The import at the top of the file (line 6) is all that's needed.
    
    main(args)

