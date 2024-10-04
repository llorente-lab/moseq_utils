import sys
import cv2
import numpy as np
import time
import subprocess
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QLineEdit,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QProgressBar,
    QMessageBox,
    QSizePolicy,
    QSpinBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import concurrent.futures
import multiprocessing


"""
Function for processing IR videos obtained from MoSeq analysis for visualization
This allows for seamless use with DeepLabCut, SLEAP, or any keypoint tracking programs

Note that to enable GPU processing, you will need to build OpenCV with GPU support. For a simple overview:
1. You will need to first have CUDA installed. Follow NVIDIA instructions to do this.
2. You will also need cudNN (NVIDIA's Deep Neural Network) module to enable support for OpenCV.
3. You will need to build OpenCV from source with GPU support. Sadly, we have to do this since unlike PyTorch or Tensorflow, we can't just use pip or conda. These instructions might be useful for doing so:
https://medium.com/@amosstaileyyoung/build-opencv-with-dnn-and-cuda-for-gpu-accelerated-face-detection-27a3cdc7e9ce
"""


class VideoProcessor:
    def __init__(
        self,
        input_path,
        output_dir,
        display=True,
        progress_callback=None,
        frame_callback=None,
        threads=1,
        use_cuda=False,  # New parameter to indicate CUDA usage
    ):
        """
        Initialize VideoProcessor Object that processes IR videos and applies CLAHE and adjust brightness and contrast.

        Args:
            input_path (str): Path to input video file or directory containing video files
            output_dir (str): Path to output directory for processed videos
            display (bool, optional): Whether to display processed frames
            progress_callback (function, optional): Callback function for progress updates
            frame_callback (function, optional): Callback function for processed frames
            threads (int, optional): Number of threads to use for processing. Defaults to 1
            use_cuda (bool, optional): Whether to use CUDA-based GPU processing. Defaults to False

        Attributes:
            input_path (Path): The resolved input file or directory path
            output_dir (Path): The resolved output directory path where processed files are saved
            display (bool): Whether processed frames will be displayed
            progress_callback (function): Function for updating progress during processing
            frame_callback (function): Function for handling frame-by-frame updates
            active_ffmpeg_processes (list): A list to store and manage FFmpeg subprocesses
            threads (int): The number of threads (or CPU cores) to use for processing
            use_cuda (bool): Whether to use CUDA for processing
            stop_requested (bool): Flag to indicate if a stop has been requested
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.display = display
        self.progress_callback = progress_callback
        self.frame_callback = frame_callback
        self.active_ffmpeg_processes = []  # List to track active FFmpeg subprocesses
        self.threads = threads
        self.use_cuda = use_cuda and self._check_cuda_available()
        self.stop_requested = False  # Flag to indicate if a stop has been requested

    def _check_cuda_available(self):
        """
        Check if OpenCV is built with CUDA support and if CUDA-enabled devices are available.

        Returns:
            bool: True if CUDA is available, False otherwise.
        """
        try:
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_count > 0:
                print(f"CUDA is available. {cuda_count} CUDA-enabled device(s) found.")
                return True
            else:
                print("No CUDA-enabled devices found. Falling back to CPU processing.")
                return False
        except AttributeError:
            print(
                "OpenCV is not built with CUDA support. Falling back to CPU processing."
            )
            return False

    @staticmethod
    def adjust_brightness_contrast(frame, brightness=150, contrast=210):
        """
        Adjust brightness and contrast of a frame using CPU.

        Args:
            frame (numpy.array): Input frame
            brightness (int, optional): Brightness adjustment value
            contrast (int, optional): Contrast adjustment value

        Returns:
            adjusted (numpy.array): Adjusted frame
        """
        alpha = (contrast + 100) / 100.0
        beta = brightness
        adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return adjusted

    @staticmethod
    def adaptive_histogram_equalization(frame):
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an input frame using CPU.

        Args:
            frame (numpy.array): Input frame, a 3-channel BGR image as a numpy array.

        Returns:
            equalized_bgr (numpy.array): Equalized frame with enhanced contrast, in BGR format.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        return equalized_bgr

    @staticmethod
    def adjust_brightness_contrast_gpu(frame, brightness=150, contrast=210):
        """
        Adjust brightness and contrast of a frame using CUDA.

        Args:
            frame (numpy.array): Input frame
            brightness (int, optional): Brightness adjustment value
            contrast (int, optional): Contrast adjustment value

        Returns:
            adjusted (numpy.array): Adjusted frame
        """
        # Upload frame to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        # Adjust brightness and contrast
        alpha = (contrast + 100) / 100.0
        beta = brightness
        gpu_adjusted = cv2.cuda.convertScaleAbs(gpu_frame, alpha=alpha, beta=beta)

        # Download back to CPU
        adjusted = gpu_adjusted.download()
        return adjusted

    @staticmethod
    def adaptive_histogram_equalization_gpu(frame):
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an input frame using CUDA.

        Args:
            frame (numpy.array): Input frame, a 3-channel BGR image as a numpy array.

        Returns:
            equalized_bgr (numpy.array): Equalized frame with enhanced contrast, in BGR format.
        """
        # Upload frame to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        # Convert to grayscale
        gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE
        clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gpu_equalized = clahe.apply(gpu_gray)

        # Convert back to BGR
        gpu_equalized_bgr = cv2.cuda.cvtColor(gpu_equalized, cv2.COLOR_GRAY2BGR)

        # Download back to CPU
        equalized_bgr = gpu_equalized_bgr.download()
        return equalized_bgr

    def stop_processing(self):
        """
        Terminate all active FFmpeg subprocesses and set the stop flag.
        """
        self.stop_requested = True
        print("Stop requested. Terminating all active FFmpeg subprocesses...")

        for process in self.active_ffmpeg_processes:
            if process.poll() is None:  # If the process is still running
                print(f"Terminating FFmpeg subprocess with PID: {process.pid}")
                process.terminate()  # Send SIGTERM
                try:
                    process.wait(timeout=5)  # Wait for it to terminate
                    print(
                        f"FFmpeg subprocess with PID {process.pid} terminated gracefully."
                    )
                except subprocess.TimeoutExpired:
                    print(
                        f"FFmpeg subprocess with PID {process.pid} did not terminate in time. Killing it."
                    )
                    process.kill()  # Force kill if it didn't terminate

        self.active_ffmpeg_processes.clear()  # Clear the list after termination
        print("All active FFmpeg subprocesses have been terminated.")

    def process_file(self, file, file_index, total_files):
        """
        Process a single video file.

        Args:
            file (Path): Path to the video file
            file_index (int): Index of the current file being processed
            total_files (int): Total number of files to process
        """
        try:
            cap = cv2.VideoCapture(str(file))
            if not cap.isOpened():
                raise IOError(f"Cannot open video file {file}")
        except Exception as e:
            raise IOError(f"Exception occurred while opening video file {file}: {e}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if FPS is zero
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_name = file.stem + "_processed.mp4"
        output_path = self.output_dir / output_name

        # Start time for processing speed calculation
        start_time = time.time()
        processed_frames = 0

        # Start FFmpeg process with CUDA-accelerated encoding if possible
        if self.use_cuda:
            vcodec = "h264_nvenc"  # NVIDIA's hardware-accelerated H.264 encoder
            preset = "fast"  # Preset can be adjusted based on desired speed/quality
        else:
            vcodec = "libx264"  # Software-based H.264 encoder
            preset = "ultrafast"

        command = [
            "ffmpeg",  # call the ffmpeg process
            "-y",  # overwrite file if it exists (good for testing)
            "-f",
            "rawvideo",  # tell ffmpeg we're sending raw video
            "-vcodec",
            "rawvideo",  # which is raw video
            "-s",
            f"{width}x{height}",  # size of one frame
            "-pix_fmt",
            "bgr24",  # pixel format (matches OpenCV's format)
            "-r",
            str(fps),  # the video fps
            "-i",
            "-",  # The input comes from a pipe
            "-an",  # Tells FFmpeg not to expect any audio
            "-vcodec",
            vcodec,  # Encoder based on CUDA availability
            "-pix_fmt",
            "yuv420p",  # pixel format
            "-preset",
            preset,  # encoding preset
            "-tune",
            "film",  # tuning parameter
            "-threads",
            str(
                self.threads
                if self.threads > 0
                else multiprocessing.cpu_count()  # set the number of CPU cores to be used here
            ),
            str(output_path),  # where to save the encoded video
        ]

        ffmpeg_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self.active_ffmpeg_processes.append(ffmpeg_process)  # Track the subprocess
        
        import threading

        def read_ffmpeg_output(process):
            for line in iter(process.stderr.readline, b""):
                print(f"FFmpeg: {line.decode().strip()}")

        threading.Thread(
            target=read_ffmpeg_output, args=(ffmpeg_process,), daemon=True
        ).start()

        # Notify about the current file
        if self.progress_callback:
            self.progress_callback(
                "file_changed",
                {
                    "current_file": file.name,
                    "file_index": file_index,
                    "total_files": total_files,
                },
            )

        while True:
            if self.stop_requested:
                # Stop has been requested; break the loop
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Process frame using CUDA or CPU
            if self.use_cuda:
                try:
                    adjusted_frame = self.adjust_brightness_contrast_gpu(frame)
                    equalized_frame = self.adaptive_histogram_equalization_gpu(
                        adjusted_frame
                    )
                except cv2.error as e:
                    # If CUDA processing fails, fallback to CPU
                    print(f"CUDA processing failed: {e}. Falling back to CPU.")
                    self.use_cuda = False
                    adjusted_frame = self.adjust_brightness_contrast(frame)
                    equalized_frame = self.adaptive_histogram_equalization(
                        adjusted_frame
                    )
            else:
                adjusted_frame = self.adjust_brightness_contrast(frame)
                equalized_frame = self.adaptive_histogram_equalization(adjusted_frame)

            frame_bytes = equalized_frame.tobytes()

            try:
                ffmpeg_process.stdin.write(frame_bytes)
            except BrokenPipeError:
                raise IOError("FFmpeg subprocess pipe is broken.")

            processed_frames += 1

            # Compute video progress
            current_video_progress = (
                int((processed_frames / total_frames) * 100) if total_frames > 0 else 0
            )

            # Compute processing speed
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                current_fps = processed_frames / elapsed_time
                processing_speed = current_fps / fps  # speed factor
            else:
                current_fps = 0
                processing_speed = 0

            # Update progress
            if self.progress_callback:
                self.progress_callback(
                    "progress",
                    {
                        "current_video_progress": current_video_progress,
                        "current_fps": current_fps,
                        "processing_speed": processing_speed,
                        "processed_frames": processed_frames,
                        "total_frames": total_frames,
                        "file_index": file_index,
                        "total_files": total_files,
                    },
                )

            # Update frame in GUI
            if self.display and self.frame_callback:
                self.frame_callback(equalized_frame)

        cap.release()
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        self.active_ffmpeg_processes.remove(ffmpeg_process)  # Remove from tracking list

    def process_folder(self, stop_check=None):
        """
        Process all video files in the input folder using multiprocessing.

        Args:
            stop_check (function): Function to check if processing should be stopped
        """
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        files = []
        if self.input_path.is_dir():
            files = list(self.input_path.glob("*.avi")) + list(
                self.input_path.glob("*.mp4")
            )
        elif self.input_path.is_file():
            if self.input_path.suffix.lower() in [".avi", ".mp4"]:
                files = [self.input_path]
            else:
                raise ValueError(
                    "Input file is not a supported video format (.avi or .mp4)."
                )
        else:
            raise ValueError("Invalid input path.")

        total_files = len(files)
        if total_files == 0:
            raise FileNotFoundError("No video files found in the input directory.")

        # Initialize the ThreadPoolExecutor
        max_workers = self.threads if self.threads > 0 else multiprocessing.cpu_count()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = []
            for i, file in enumerate(files):
                if stop_check and stop_check():
                    break
                file_index = i + 1
                futures.append(
                    executor.submit(self.process_file, file, file_index, total_files)
                )

            # Monitor the futures
            for future in concurrent.futures.as_completed(futures):
                if stop_check and stop_check():
                    break
                try:
                    future.result()
                except Exception as e:
                    if self.progress_callback:
                        self.progress_callback("error", str(e))

        # Update overall progress to 100% only if not stopped
        if self.progress_callback and not self.stop_requested:
            self.progress_callback(
                "overall_progress",
                {
                    "overall_progress": 100,
                    "files_processed": total_files,
                    "total_files": total_files,
                },
            )


class VideoProcessorThread(QThread):
    """
    A QThread subclass for running video processing in a separate thread.
    """

    file_changed = pyqtSignal(str)
    current_video_progress = pyqtSignal(int)
    overall_progress = pyqtSignal(int)
    current_fps = pyqtSignal(float)
    processing_speed = pyqtSignal(float)
    frames_processed = pyqtSignal(int, int)  # processed_frames, total_frames
    files_processed = pyqtSignal(int, int)  # files_processed, total_files
    frame = pyqtSignal(np.ndarray)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, processor):
        """
        Initialize the VideoProcessorThread.

        Args:
            processor (VideoProcessor): The VideoProcessor instance to run in this thread
        """
        super().__init__()
        self.processor = processor
        self.processor.progress_callback = self.progress_callback
        self.processor.frame_callback = self.emit_frame
        self.is_running = True

    def run(self):
        """
        Run the actual processing operation.
        """
        try:
            self.processor.process_folder(stop_check=self.is_stopped)
            if self.is_running and not self.processor.stop_requested:
                self.finished.emit()
            elif self.processor.stop_requested:
                self.error.emit("Processing stopped by user.")
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.processor.active_ffmpeg_processes.clear()  # Clear any FFmpeg subprocess references

    def stop(self):
        """
        Safely stop the processing operation
        """
        self.is_running = False
        self.processor.stop_processing()  # Terminate FFmpeg subprocesses and set stop flag

    def is_stopped(self):
        """
        Helper function that tells us whether the process has been stopped or not
        """
        return not self.is_running

    def progress_callback(self, event_type, data):
        """
        Handles progress updates for the video processing operation by emitting corresponding signals to the GUI.
        """
        if event_type == "file_changed":
            self.file_changed.emit(data["current_file"])
            self.files_processed.emit(data["file_index"] - 1, data["total_files"])
        elif event_type == "progress":
            self.current_video_progress.emit(data["current_video_progress"])
            self.current_fps.emit(data["current_fps"])
            self.processing_speed.emit(data["processing_speed"])
            self.frames_processed.emit(data["processed_frames"], data["total_frames"])
        elif event_type == "overall_progress":
            self.overall_progress.emit(data["overall_progress"])
            self.files_processed.emit(data["files_processed"], data["total_files"])
        elif event_type == "error":
            self.error.emit(data)

    def emit_frame(self, frame):
        """
        Function to emit a signal to the PyQt GUI.

        Args:
            frame (numpy.array): NumPy Array representing the frame, to be converted into a PyQt object and displayed.
        """
        self.frame.emit(frame)


class VideoProcessorGUI(QMainWindow):
    """
    PyQt5 based GUI for an easy UI.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Processor")
        self.resize(800, 900)  # Increased height to accommodate new widgets
        self.processor = None
        self.thread = None
        self.initUI()

    def initUI(self):
        """
        Initialize the UI with widgets, labels, spinboxes, buttons, etc.
        """
        input_label = QLabel("Input Path:")
        self.input_line = QLineEdit()
        input_button = QPushButton("Browse")
        input_button.clicked.connect(self.select_input_path)

        output_label = QLabel("Output Directory:")
        self.output_line = QLineEdit()
        output_button = QPushButton("Browse")
        output_button.clicked.connect(self.select_output_dir)

        self.display_checkbox = QCheckBox("Display Video")
        self.display_checkbox.setChecked(True)

        # CUDA option checkbox
        self.cuda_checkbox = QCheckBox("Enable CUDA-based GPU Processing")
        self.cuda_checkbox.setChecked(False)  # Default to CPU processing

        # CPU Core selection
        cpu_label = QLabel("Number of CPU Cores (0 for all):")
        self.cpu_spinbox = QSpinBox()
        self.cpu_spinbox.setMinimum(0)
        self.cpu_spinbox.setMaximum(multiprocessing.cpu_count())
        self.cpu_spinbox.setValue(1)

        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)

        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)

        # Video display area -- no fixed size
        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Current file label
        self.current_file_label = QLabel("Current File: None")

        # Create progress labels
        self.current_video_progress_label = QLabel(
            "Current Video Progress: 0% (Frames: 0 / 0)"
        )
        self.current_video_progress_bar = QProgressBar()
        self.current_video_progress_bar.setAlignment(Qt.AlignCenter)
        self.current_video_progress_bar.setFormat("%p%")

        # Combined label for overall progress and files processed
        self.overall_progress_label = QLabel("Overall Progress: 0% (Files: 0 / 0)")
        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setAlignment(Qt.AlignCenter)
        self.overall_progress_bar.setFormat("%p%")

        # Create speed and fps labels
        self.fps_label = QLabel("Current FPS: 0.00")
        self.speed_label = QLabel("Processing Speed: 0.00x")

        # Processing mode label
        self.mode_label = QLabel("Processing Mode: CPU")

        # Adjust size policies for other widgets
        self.start_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.stop_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        input_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        output_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Layouts for buttons, checkboxes, etc.
        layout = QVBoxLayout()

        input_layout = QHBoxLayout()
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_line)
        input_layout.addWidget(input_button)

        output_layout = QHBoxLayout()
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_line)
        output_layout.addWidget(output_button)

        cpu_layout = QHBoxLayout()
        cpu_layout.addWidget(cpu_label)
        cpu_layout.addWidget(self.cpu_spinbox)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        # Current video progress layout
        current_video_progress_layout = QVBoxLayout()
        current_video_progress_layout.addWidget(self.current_video_progress_label)
        current_video_progress_layout.addWidget(self.current_video_progress_bar)

        # Overall progress layout
        overall_progress_layout = QVBoxLayout()
        overall_progress_layout.addWidget(self.overall_progress_label)
        overall_progress_layout.addWidget(self.overall_progress_bar)

        # Add widgets to the main layout
        layout.addLayout(input_layout)
        layout.addLayout(output_layout)
        layout.addLayout(cpu_layout)
        layout.addWidget(self.display_checkbox)
        layout.addWidget(self.cuda_checkbox)  # Added CUDA checkbox
        layout.addWidget(self.mode_label)  # Added processing mode label
        layout.addLayout(button_layout)
        layout.addWidget(self.current_file_label)
        layout.addWidget(self.video_label)
        layout.addLayout(current_video_progress_layout)
        layout.addLayout(overall_progress_layout)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.speed_label)

        # Create a central widget and set the layout
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    @pyqtSlot()
    def select_input_path(self):
        """
        Function for input path selection.

        This has a @pyqtSlot decorator to signify that it can receive a signal.
        """
        # Allow selection of both files and directories
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_or_dir = QFileDialog.getExistingDirectory(
            self, "Select Input Directory", options=options
        )
        if not file_or_dir:
            # If no directory selected, allow file selection
            file_dialog = QFileDialog(self, "Select Input File", options=options)
            file_dialog.setFileMode(QFileDialog.ExistingFiles)
            if file_dialog.exec_():
                selected_files = file_dialog.selectedFiles()
                if selected_files:
                    # If multiple files are selected, choose the directory
                    if len(selected_files) > 1:
                        self.input_line.setText(str(Path(selected_files[0]).parent))
                    else:
                        self.input_line.setText(selected_files[0])
        else:
            self.input_line.setText(file_or_dir)

    @pyqtSlot()
    def select_output_dir(self):
        """
        Function for output directory selection.

        This has a @pyqtSlot decorator to signify that it can receive a signal.
        """
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_line.setText(path)

    @pyqtSlot()
    def start_processing(self):
        """
        Start the processing, update video progress, and handle multithreading as needed.

        This has a @pyqtSlot decorator to signify that it can receive a signal (in this case, from the VideoProcessorThread class and the FFmpeg process).
        """
        input_path = self.input_line.text()
        output_dir = self.output_line.text()
        display = self.display_checkbox.isChecked()
        use_cuda = self.cuda_checkbox.isChecked()
        threads = self.cpu_spinbox.value()

        if not input_path or not output_dir:
            QMessageBox.warning(
                self,
                "Input Error",
                "Please specify both input path and output directory.",
            )
            return

        # Reset progress bars and labels
        self.current_video_progress_bar.setValue(0)
        self.overall_progress_bar.setValue(0)
        self.fps_label.setText("Current FPS: 0.00")
        self.speed_label.setText("Processing Speed: 0.00x")
        self.current_file_label.setText("Current File: None")
        self.current_video_progress_label.setText(
            "Current Video Progress: 0% (Frames: 0 / 0)"
        )
        self.overall_progress_label.setText("Overall Progress: 0% (Files: 0 / 0)")

        # Initialize VideoProcessor with CUDA option
        self.processor = VideoProcessor(
            input_path, output_dir, display=display, threads=threads, use_cuda=use_cuda
        )

        # Update processing mode label
        if self.processor.use_cuda:
            self.mode_label.setText("Processing Mode: CUDA-enabled GPU")
            print("CUDA-based GPU processing is enabled.")
        else:
            self.mode_label.setText("Processing Mode: CPU")
            print("CPU processing is enabled.")

        # Set start and stop button defaults
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # Start processing in a separate thread
        self.thread = VideoProcessorThread(self.processor)
        self.thread.file_changed.connect(self.update_current_file)
        self.thread.current_video_progress.connect(self.update_current_video_progress)
        self.thread.overall_progress.connect(self.update_overall_progress)
        self.thread.current_fps.connect(self.update_fps)
        self.thread.processing_speed.connect(self.update_speed)
        self.thread.frames_processed.connect(self.update_frames_processed)
        self.thread.files_processed.connect(self.update_files_processed)
        self.thread.finished.connect(self.processing_finished)
        self.thread.error.connect(self.handle_error)
        self.thread.frame.connect(self.update_frame)
        self.thread.start()

    @pyqtSlot()
    def stop_processing(self):
        """
        Destructor for safe video process stopping.
        """
        if self.thread and self.thread.isRunning():
            self.stop_button.setEnabled(False)  # Disable to prevent multiple clicks
            self.thread.stop()
            self.thread.wait()  # Wait for the thread to finish
            self.processing_finished()

    @pyqtSlot(str)
    def update_current_file(self, current_file):
        """
        Helper function to update the current file in the GUI.

        This has a @pyqtSlot decorator to signify that it can receive a signal.
        """
        self.current_file_label.setText(f"Current File: {current_file}")

    @pyqtSlot(int)
    def update_current_video_progress(self, value):
        """
        Helper function to update the number of frames that have processed.

        This has a @pyqtSlot decorator to signify that it can receive a signal.
        """
        self.current_video_progress_bar.setValue(value)
        # Update the label text to include the progress percentage
        current_text = self.current_video_progress_label.text()
        if "Frames:" in current_text:
            frames_info = current_text.split("Frames:")[-1]
            updated_text = f"Current Video Progress: {value}% (Frames:{frames_info})"
        else:
            updated_text = f"Current Video Progress: {value}%"
        self.current_video_progress_label.setText(updated_text)

    @pyqtSlot(int)
    def update_overall_progress(self, value):
        """
        Helper function to update the number of files in the GUI that have been processed.

        Args:
            value (float): The percentage of files that have been processed.

        This has a @pyqtSlot decorator to signify that it can receive a signal.
        """
        self.overall_progress_bar.setValue(value)
        # Update the label text to include the progress percentage
        current_text = self.overall_progress_label.text()
        if "Files:" in current_text:
            files_info = current_text.split("Files:")[-1]
            updated_text = f"Overall Progress: {value}% (Files:{files_info})"
        else:
            updated_text = f"Overall Progress: {value}%"
        self.overall_progress_label.setText(updated_text)

    @pyqtSlot(float)
    def update_fps(self, fps):
        """
        Helper function to update the fps.

        Args:
            fps (float): Current fps of the video processing operation.

        This has a @pyqtSlot decorator to signify that it can receive a signal.
        """
        self.fps_label.setText(f"Current FPS: {fps:.2f}")

    @pyqtSlot(float)
    def update_speed(self, speed):
        """
        Helper function to update the video speed.

        Args:
            speed (float): Current speed of the video processing operation.

        This has a @pyqtSlot decorator to signify that it can receive a signal.
        """
        self.speed_label.setText(f"Processing Speed: {speed:.2f}x")

    @pyqtSlot(int, int)
    def update_frames_processed(self, processed_frames, total_frames):
        """
        Helper function to update the number of frames that have been processed in the GUI.

        Args:
            processed_frames (int): Total number of frames that have been processed.
            total_frames (int): Total number of frames in the video.

        This has a @pyqtSlot decorator to signify that it can receive a signal.
        """
        percentage = (processed_frames / total_frames) * 100 if total_frames > 0 else 0
        self.current_video_progress_label.setText(
            f"Current Video Progress: {percentage:.2f}% (Frames: {processed_frames} / {total_frames})"
        )

    @pyqtSlot(int, int)
    def update_files_processed(self, files_processed, total_files):
        """
        Helper function to update the percentage of files that have been processed.

        Args:
            files_processed (int): Number of files that have been processed.
            total_files (int): Total number of files.

        This has a @pyqtSlot decorator to signify that it can receive a signal.
        """
        percentage = (files_processed / total_files) * 100 if total_files > 0 else 0
        self.overall_progress_label.setText(
            f"Overall Progress: {percentage:.2f}% (Files: {files_processed} / {total_files})"
        )

    @pyqtSlot()
    def processing_finished(self):
        """
        Function to signify whether video processing has been completed.
        """
        QMessageBox.information(
            self, "Processing Finished", "Video processing is complete."
        )
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.video_label.clear()
        self.mode_label.setText("Processing Mode: CPU")  # Reset to default mode

    @pyqtSlot(str)
    def handle_error(self, error_message):
        """
        Function for basic error handling.

        Args:
            error_message (str): The error message to be displayed.

        This has a @pyqtSlot decorator to signify that it can receive a signal.
        """
        QMessageBox.critical(self, "Error", error_message)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.video_label.clear()
        self.mode_label.setText("Processing Mode: CPU")  # Reset to default mode

    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        """
        Function for updating the frame in the display GUI. We use PyQt here instead of OpenCV for better memory handling. This function converts a numpy array into a QImage and then creates a PixMap to be displayed within the GUI.

        Args:
            frame (numpy.array): A 3D numpy array with shape (height, width, channels) representing the video frame.
        """
        # Convert the frame to QImage and display it
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            frame.data, width, height, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def closeEvent(self, event):
        """
        Function for handling PyQt window closing.

        Args:
            event (QCloseEvent): An event generated when the user attempts to close a window.
        """
        # Ensure the thread and ffmpeg process are properly terminated when the GUI is closed
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
        if self.processor:
            self.processor.stop_processing()  # Ensure all subprocesses are terminated
        event.accept()


def main():
    app = QApplication(sys.argv)  # Initialize the Qt Application
    gui = VideoProcessorGUI()  # Create the GUI
    gui.show()  # Display the GUI
    sys.exit(app.exec_())  # Execute the application loop


if __name__ == "__main__":
    main()
