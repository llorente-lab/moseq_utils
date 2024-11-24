import tarfile
import cv2
import numpy as np
from tqdm import tqdm
from moseq2_extract.io.video import (
    read_frames_raw,
    read_mkv,
    get_video_info,
    get_stream_names,
)
import subprocess
import datetime
import sys


def construct_ffmpeg_command(
    filename,
    frames,
    fps,
    frame_size,
    pixel_format,
    threads,
    slices,
    slicecrc,
    mapping,
    frames_is_timestamp,
):
    """
    Constructs the FFmpeg command to extract frames.

    Args:
        filename (str): Path to the video file.
        frames (list): List of frame indices or timestamps.
        fps (int): Frames per second.
        frame_size (tuple): (width, height) of the frames.
        pixel_format (str): Pixel format for FFmpeg.
        threads (int): Number of threads for FFmpeg.
        slices (int): Number of slices for FFmpeg.
        slicecrc (int): Slice CRC for FFmpeg.
        mapping (str or int): Stream mapping.
        frames_is_timestamp (bool): Whether frames are timestamps.

    Returns:
        list: FFmpeg command as a list.
    """
    if not frames:
        raise ValueError("Frames list is empty. Cannot construct FFmpeg command.")

    # Determine the starting time
    if frames_is_timestamp:
        start_time = str(datetime.timedelta(seconds=frames[0]))
    else:
        start_time = str(datetime.timedelta(seconds=frames[0] / fps))

    cmd = [
        "ffmpeg",
        "-loglevel",
        "fatal",
        "-ss",
        start_time,
        "-i",
        filename,
        "-vframes",
        str(len(frames)),
        "-f",
        "image2pipe",
        "-s",
        f"{frame_size[0]}x{frame_size[1]}",
        "-pix_fmt",
        pixel_format,
        "-threads",
        str(threads),
        "-slices",
        str(slices),
        "-slicecrc",
        str(slicecrc),
        "-vcodec",
        "rawvideo",
    ]

    if isinstance(mapping, str):
        mapping_dict = get_stream_names(filename)
        mapping = mapping_dict.get(mapping, 0)

    if filename.lower().endswith((".mkv", ".avi")):
        cmd += ["-map", f"0:{mapping}"]
        cmd += ["-vsync", "0"]

    cmd += ["-"]
    return cmd


def execute_ffmpeg(cmd, num_frames, frame_size, movie_dtype):
    """
    Execute FFmpeg command and read frames into a numpy array.

    Args:
        cmd (list): FFmpeg command.
        num_frames (int): Number of frames to read.
        frame_size (tuple): (width, height) of the frames.
        movie_dtype (str): Numpy dtype for the frames.

    Returns:
        numpy.ndarray or None: Frames as a numpy array with shape (num_frames, height, width), or None if an error occurs.
    """
    try:
        print("Creating FFmpeg pipe...")
        pipe = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = pipe.communicate()

        if pipe.returncode != 0:
            print("FFmpeg Error:", err.decode(), file=sys.stderr)
            return None

        # Calculate total number of pixels per frame
        total_pixels = frame_size[0] * frame_size[1]

        # Calculate expected number of bytes
        dtype = np.dtype(movie_dtype)
        expected_bytes = num_frames * total_pixels * dtype.itemsize

        if len(out) != expected_bytes:
            print(
                f"Warning: Expected {expected_bytes} bytes, but got {len(out)} bytes.",
                file=sys.stderr,
            )

        # Convert buffer to numpy array
        print("Converting video to NumPy array...")
        video = np.frombuffer(out, dtype=movie_dtype)

        if len(video) != num_frames * frame_size[1] * frame_size[0]:
            print(
                f"Warning: Mismatch in frame data size. Expected {num_frames * frame_size[1] * frame_size[0]}, got {len(video)}.",
                file=sys.stderr,
            )

        # Reshape to (num_frames, height, width)
        video = video.reshape((num_frames, frame_size[1], frame_size[0]))

        return video.astype("uint16")  # Adjust dtype as needed
    except Exception as e:
        print(f"Exception during FFmpeg execution: {e}", file=sys.stderr)
        return None


def read_avi(
    filename,
    frames=range(0),
    threads=6,
    fps=30,
    frames_is_timestamp=False,
    pixel_format="gray16le",
    movie_dtype="uint16",
    frame_size=None,
    slices=24,
    slicecrc=1,
    mapping="DEPTH",
    get_cmd=False,
    finfo=None,
    batch_size=1000,  # New parameter for batch processing
    progress=None,  # New parameter for external progress bar
    **kwargs,
):
    """
    Read frames from an .avi file using FFmpeg, optionally in batches.

    Args:
        filename (str): Path to the .avi file.
        frames (int, list, range, optional): Frame indices to grab. If None, all frames are loaded.
        threads (int, optional): Number of threads for FFmpeg. Default is 6.
        fps (int, optional): Frames per second of the video. Default is 30.
        frames_is_timestamp (bool, optional): If True, frames represent timestamps in seconds. Default is False.
        pixel_format (str, optional): Pixel format for FFmpeg. Default is "gray16le".
        movie_dtype (str, optional): Numpy dtype for the frames. Default is "uint16".
        frame_size (tuple or None, optional): (width, height) of frames. If None, obtained from finfo or video metadata.
        slices (int, optional): Number of slices for FFmpeg. Default is 24.
        slicecrc (int, optional): Slice CRC for FFmpeg. Default is 1.
        mapping (str or int, optional): Stream mapping. Default is "DEPTH".
        get_cmd (bool, optional): If True, return FFmpeg commands instead of executing. Default is False.
        finfo (dict or None, optional): Video file metadata. If None, obtained via get_video_info().
        batch_size (int or None, optional): Number of frames to load at once. If None, load all frames without batching. Default is 1000.
        progress (tqdm.tqdm or None, optional): External progress bar to update. If provided, used instead of creating a new one.
        **kwargs: Additional parameters for FFmpeg command.

    Returns:
        numpy.ndarray or list: If batch_size is None, returns a NumPy array of shape (nframes, height, width).
                               If batch_size is specified, returns a list of NumPy arrays, each with shape (batch_size, height, width).
                               If get_cmd=True, returns a list of FFmpeg command lists.
    """
    # Handle single frame input
    if isinstance(frames, int):
        frames = [frames]

    # If frames is None or empty, load all frames
    print("Getting video info...")
    if frames is None or len(frames) == 0:
        finfo = get_video_info(filename, threads=threads, **kwargs)
        frames = list(range(finfo["nframes"]))
    elif isinstance(frames, (list, range, np.ndarray)):
        frames = list(frames)
    else:
        raise ValueError("frames must be an integer, a list, a range, or None.")

    if finfo is None:
        finfo = get_video_info(filename, threads=threads, **kwargs)

    if not frame_size:
        frame_size = finfo["dims"]  # (width, height)

    if get_cmd:
        # Generate and return FFmpeg commands for each batch
        cmds = []
        if batch_size is None:
            batch_size = len(frames)  # Single batch
        n_batches = (len(frames) + batch_size - 1) // batch_size  # Ceiling division
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]
            cmd = construct_ffmpeg_command(
                filename=filename,
                frames=batch_frames,
                fps=fps,
                frame_size=frame_size,
                pixel_format=pixel_format,
                threads=threads,
                slices=slices,
                slicecrc=slicecrc,
                mapping=mapping,
                frames_is_timestamp=frames_is_timestamp,
            )
            cmds.append(cmd)
        return cmds

    # If batch_size is None, load all frames at once
    if batch_size is None:
        num_frames = len(frames)
        cmd = construct_ffmpeg_command(
            filename=filename,
            frames=frames,
            fps=fps,
            frame_size=frame_size,
            pixel_format=pixel_format,
            threads=threads,
            slices=slices,
            slicecrc=slicecrc,
            mapping=mapping,
            frames_is_timestamp=frames_is_timestamp,
        )

        # Execute FFmpeg and read frames
        video = execute_ffmpeg(cmd, num_frames, frame_size, movie_dtype)
        if video is None:
            raise RuntimeError("Failed to load frames using FFmpeg.")
        return video
    else:
        # Load frames in batches with tqdm progress bar if no external progress is provided
        frame_data_list = []
        n_batches = (len(frames) + batch_size - 1) // batch_size  # Ceiling division

        # Create a local progress bar if none is provided
        created_progress = False
        if progress is None:
            progress = tqdm(total=n_batches, desc="Loading Batches")
            created_progress = True

        try:
            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(frames))
                batch_frames = frames[batch_start:batch_end]
                num_frames_in_batch = len(batch_frames)

                # Construct FFmpeg command for this batch
                cmd = construct_ffmpeg_command(
                    filename=filename,
                    frames=batch_frames,
                    fps=fps,
                    frame_size=frame_size,
                    pixel_format=pixel_format,
                    threads=threads,
                    slices=slices,
                    slicecrc=slicecrc,
                    mapping=mapping,
                    frames_is_timestamp=frames_is_timestamp,
                )

                # Execute FFmpeg and get batch data
                batch_data = execute_ffmpeg(
                    cmd, num_frames_in_batch, frame_size, movie_dtype
                )

                if batch_data is not None:
                    frame_data_list.append(batch_data)
                else:
                    print(f"Batch {batch_idx + 1} failed to load.", file=sys.stderr)

                # Update the progress bar
                progress.update(1)
        finally:
            # Close the progress bar if it was created here
            if created_progress:
                progress.close()

        if not frame_data_list:
            raise ValueError(
                "No frames were loaded. Please check the video file and frame indices."
            )

        # Concatenate all batches into a single numpy array
        video = np.concatenate(frame_data_list, axis=0)
        return video


def load_frames(
    filename,
    frames=None,
    frame_size=(512, 424),
    bit_depth=16,
    batch_size=None,
    **kwargs,
):
    """
    Parse file extension and load the movie data into a numpy array, optionally in batches with a progress bar.

    Args:
        filename (str or tarfile.TarFile): Path to the video file or a TarFile object.
        frames (int or list, optional): Frame indices to read into the output array. If None, all frames are loaded.
        frame_size (tuple, optional): Video dimensions as (width, height). Default is (512, 424).
        bit_depth (int, optional): Number of bits per pixel, corresponds to image resolution. Default is 16.
        batch_size (int, optional): Number of frames to load at once. If None, loads all frames without batching.
        **kwargs: Any additional parameters that could be required in read_frames_raw(), read_mkv(), or read_avi().

    Returns:
        numpy.ndarray: Read video as a numpy array with shape (nframes, height, width).
    """
    # Handle single frame input
    if isinstance(frames, int):
        frames = [frames]

    finfo = get_video_info(filename)
    print("total video frames", finfo["nframes"])

    # If frames is None, load all frames
    if frames is None:
        finfo = get_video_info(filename)
        total_frames = finfo["nframes"]
        frames = list(range(total_frames))
    elif isinstance(frames, list):
        total_frames = len(frames)
    elif isinstance(frames, (range, np.ndarray)):
        frames = list(frames)
        total_frames = len(frames)
    else:
        raise ValueError(
            "frames must be an integer, a list of integers, a range, or None."
        )

    print("total frames", total_frames)
    # If batch_size is not specified, load all frames at once
    if batch_size is None:
        try:
            if isinstance(filename, tarfile.TarFile):
                frame_data = read_frames_raw(
                    filename,
                    frames=frames,
                    frame_size=frame_size,
                    bit_depth=bit_depth,
                    **kwargs,
                )
            elif isinstance(filename, str) and filename.lower().endswith(".dat"):
                frame_data = read_frames_raw(
                    filename,
                    frames=frames,
                    frame_size=frame_size,
                    bit_depth=bit_depth,
                    **kwargs,
                )
            elif isinstance(filename, str) and filename.lower().endswith(".mkv"):
                frame_data = read_mkv(filename, frames, frame_size=frame_size, **kwargs)
            elif isinstance(filename, str) and filename.lower().endswith(".avi"):
                print("Reading frames from AVI file...")
                frame_data = read_avi(
                    filename,
                    frames=frames,
                    frame_size=frame_size,
                    movie_dtype=f"uint{bit_depth}",
                    batch_size=None,  # Explicitly load all frames at once
                    **kwargs,
                )
            else:
                raise ValueError(f"Unsupported file format for file: {filename}")
        except AttributeError as e:
            print("Error reading movie:", e, file=sys.stderr)
            frame_data = read_frames_raw(
                filename,
                frames=frames,
                frame_size=frame_size,
                bit_depth=bit_depth,
                **kwargs,
            )
        return frame_data
    else:
        # Load frames in batches with a single tqdm progress bar
        print("Loading frames in batches...")
        try:
            frame_data = read_avi(
                filename,
                frames=frames,
                frame_size=frame_size,
                movie_dtype=f"uint{bit_depth}",
                batch_size=batch_size,
                progress=None,  # Let read_avi handle the progress bar
                **kwargs,
            )
            return frame_data
        except Exception as e:
            print(f"Error loading frames in batches: {e}", file=sys.stderr)
            raise


if __name__ == "__main__":
    try:
        FRAMES = load_frames(
            filename="/Users/atharvp04/Downloads/depth.avi",
            frames=range(18000, 36199), # second half of video (0, 18000) for the first half
            frame_size=(640, 576),
            bit_depth=16,
            batch_size=100,
        )
        print(f"Loaded frames shape: {FRAMES.shape}")
        np.save("/Users/atharvp04/Downloads/FRAMES_depthavi_1.npy", FRAMES)
        print("Frames saved successfully.")
    except MemoryError:
        print(
            "MemoryError: The script was killed due to excessive memory usage.",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
