import numpy as np
import cv2
import psutil
from moseq2_extract.io.video import load_movie_data, get_movie_info
from moseq2_extract.extract.proc import clean_frames
from tqdm.auto import tqdm
import gc
import os

def get_ram_usage():
    """
    Retrieves the current RAM usage of the process in megabytes.
    
    Returns:
        float: RAM usage in MB.
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  
    return mem

def visualize_raw_depth(
    input_avi, 
    output_avi, 
    frame_size=(640, 576), 
    bit_depth=16,
    colormap='jet', 
    fps=30, 
    threads=6, 
    compress=False,
    batch_size=1000,   # Number of frames per batch
    depth_min=0,       # Minimum depth value for normalization
    depth_max=660       # Maximum depth value for normalization
):
    """
    Converts a raw depth .avi video to a human-viewable .avi video with a colormap,
    processing frames in batches, tracking RAM usage, and cropping to the first 30 seconds.
    
    Args:
        input_avi (str): Path to the raw depth .avi video file.
        output_avi (str): Path to the output human-viewable .avi video file.
        frame_size (tuple): Dimensions of each frame (width, height).
        bit_depth (int): Bits per pixel (default: 16).
        colormap (str): Colormap to apply (default: 'jet').
        fps (int): Frames per second for the output video (default: 30).
        threads (int): Number of threads for processing (default: 6).
        compress (bool): Whether to compress the output video (default: False).
        batch_size (int): Number of frames to process per batch (default: 1000).
        depth_min (float): Minimum depth value for normalization (default: 0).
        depth_max (float): Maximum depth value for normalization (default: 660).
    
    Returns:
        None
    """
    # Step 1: Get video information
    finfo = get_movie_info(input_avi, frame_size=frame_size, bit_depth=bit_depth, mapping='DEPTH', threads=threads)
    nframes = finfo.get('nframes', None)
    if nframes is None:
        raise ValueError("Could not determine the number of frames in the input video.")
    
    # Retrieve actual frame size
    actual_frame_size = finfo.get('dims', frame_size)
    if actual_frame_size != frame_size:
        print(f"Warning: Provided frame_size {frame_size} does not match actual frame size {actual_frame_size}. Using actual frame size.")
        frame_size = actual_frame_size
    
    # Retrieve FPS from video info if available
    actual_fps = finfo.get('fps', fps)
    if actual_fps != fps:
        print(f"Warning: Provided FPS {fps} does not match actual FPS {actual_fps}. Using actual FPS.")
        fps = actual_fps
    
    print(f"Video Information: {finfo}")
    print(f"Total frames in input video: {nframes}")
    
    # Calculate the number of frames corresponding to the first 30 seconds
    desired_duration_seconds = 5
    num_frames_to_process = min(nframes, int(fps * desired_duration_seconds))
    print(f"Number of frames to process for first {desired_duration_seconds} seconds: {num_frames_to_process}")
    
    # Step 2: Initialize VideoWriter with correct frame size
    # Define the codec and create VideoWriter object
    if compress:
        codec = 'mpeg4'  # Compressed codec
    else:
        codec = 'XVID'   # Uncompressed or less compressed codec
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_avi, fourcc, fps, (frame_size[0], frame_size[1]))
    
    if not out.isOpened():
        raise IOError(f"Cannot open video writer for file {output_avi}")
    
    print(f"Output video will be saved to {output_avi} with codec '{codec}'")
    
    # Step 3: Process frames in batches
    total_batches = (num_frames_to_process + batch_size - 1) // batch_size  # Ceiling division
    print(f"Processing frames in {total_batches} batches of up to {batch_size} frames each.")
    
    for batch_num in tqdm(range(total_batches), desc="Processing Batches"):
        # Determine the frame indices for this batch
        start_frame = batch_num * batch_size
        end_frame = min(start_frame + batch_size, num_frames_to_process)
        current_batch_size = end_frame - start_frame
        
        # Load frames for this batch
        frames = load_movie_data(
            filename=input_avi,
            frames=range(start_frame, end_frame),
            frame_size=finfo['dims'],
            bit_depth=bit_depth,
            pixel_format='gray16be',   # Azure typically uses 'gray16be'
            movie_dtype='>u2'
        )
        
        if frames is None or len(frames) == 0:
            print(f"No frames loaded for batch {batch_num + 1}. Skipping...")
            continue
        
        print(f"Batch {batch_num + 1}: Loaded {len(frames)} frames.")

        # clean frame
        frames = clean_frames(frames)
        
        # Normalize frames to 0-1 based on depth_min and depth_max
        frames_normalized = (frames - depth_min) / (depth_max - depth_min)
        frames_normalized = np.clip(frames_normalized, 0, 1)
        
        # Apply colormap
        if colormap.lower() == 'jet':
            cmap = cv2.COLORMAP_JET
        elif colormap.lower() == 'viridis':
            cmap = cv2.COLORMAP_VIRIDIS
        elif colormap.lower() == 'hot':
            cmap = cv2.COLORMAP_HOT
        else:
            print(f"Colormap '{colormap}' not recognized. Using 'jet' by default.")
            cmap = cv2.COLORMAP_JET
        
        # Convert normalized frames to uint8 and apply colormap
        try:
            frames_uint8 = (frames_normalized * 255).astype(np.uint8)
            frames_colored = np.array([cv2.applyColorMap(frame, cmap) for frame in frames_uint8])
        except Exception as e:
            print(f"Error applying colormap in batch {batch_num + 1}: {e}")
            continue
        
        # Verify frames_colored shape and type
        # Expecting (batch_size, height, width, 3)
        if frames_colored.ndim != 4 or frames_colored.shape[3] != 3:
            print(f"Invalid frame shape in batch {batch_num + 1}: {frames_colored.shape}. Skipping...")
            continue
        
        # Optional: Save a test frame from the first batch
        if batch_num == 0:
            test_frame = frames_colored[0]
            filename = './testoutput.jpg'
            cv2.imwrite(filename, test_frame)
            print(f"Saved {filename} for verification.")
        
        # Write frames to output video
        for idx, frame in enumerate(frames_colored):
            # Ensure frame is in the correct format
            if frame.dtype != np.uint8:
                print(f"Frame {idx + 1} in batch {batch_num + 1} has incorrect dtype: {frame.dtype}. Skipping frame.")
                continue
            if frame.shape[2] != 3:
                print(f"Frame {idx + 1} in batch {batch_num + 1} has incorrect shape: {frame.shape}. Skipping frame.")
                continue
            out.write(frame)
        
        # Track RAM usage
        current_ram = get_ram_usage()
        print(f"Batch {batch_num + 1}/{total_batches}: RAM Usage = {current_ram:.2f} MB")
        
        # Clean up to free memory
        del frames, frames_normalized, frames_colored, frames_uint8
        gc.collect()
    
    # Step 4: Release resources
    out.release()
    print("Conversion complete! Human-viewable video saved.")
    
    # Verify output video
    if os.path.exists(output_avi):
        # Get file size
        file_size_bytes = os.path.getsize(output_avi)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        cap = cv2.VideoCapture(output_avi)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_length_seconds = frame_count / fps
        minutes = int(video_length_seconds // 60)
        seconds = int(video_length_seconds % 60)
        
        cap.release()
        
        print(f"Output video size: {file_size_mb:.2f} MB")
        print(f"Video length: {minutes:02d}:{seconds:02d} (mm:ss)")
        print(f"Total frames: {frame_count}")
    else:
        print("Warning: Output file was not found!")
