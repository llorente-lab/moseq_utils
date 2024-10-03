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
    QGridLayout,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import concurrent.futures
import multiprocessing


"""
Function for processing IR videos obtained from MoSeq analysis for visualization
This allows for seamless use with DeepLabCut, SLEAP, or any keypoint tracking programs
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

        Attributes:
            input_path (Path): The resolved input file or directory path
            output_dir (Path): The resolved output directory path where processed files are saved
            display (bool): Whether processed frames will be displayed
            progress_callback (function): Function for updating progress during processing
            frame_callback (function): Function for handling frame-by-frame updates
            ffmpeg_processes (dict): A dictionary to store and manage ffmpeg subprocesses for encoding videos
            threads (int): The number of threads (or CPU cores) to use for processing
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.display = display
        self.progress_callback = progress_callback
        self.frame_callback = frame_callback
        self.ffmpeg_processes = {}
        self.threads = threads

    @staticmethod
    def adjust_brightness_contrast(frame, brightness=150, contrast=210):
        """
        Adjust brightness and contrast of a frame. This method applies a linear transformation to the pixel values of the input frame
        to adjust its brightness and contrast. The transformation is of the form:
        output = alpha * input + beta

        We assign this as a static method which means we don't need to instantiate the whole class before using this.

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
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an input frame.

        This method enhances the contrast of the input frame using CLAHE. The process involves:
        1. Converting the input frame to grayscale.
        2. Applying CLAHE to the grayscale image.
        3. Converting the result back to a 3-channel BGR image.

        We assign this as a static method which means we don't need to instantiate the whole class before using this.
        
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

        # Start FFmpeg process
        # What we are basically doing here is use multithreading (GIL-bound) to process video files, however, for each video file,
        # we are allowing FFmpeg's built in video processing capabilities, which more less bypass the GIL since this involves a manual subprocess call.
        # This makes this quite efficient and allows for faster video processing, which is useful when we have large (20 min) videos
        # that also consume a large amount of memory
        command = [
            "ffmpeg",  # call the ffmpeg process
            "-y",  # overwrite file if it exists (good for testing)
            "-f",
            "rawvideo",  # tell ffmpeg we're sending raw video
            "-vcodec",  # tell ffmpeg the codec
            "rawvideo",  # which is raw video
            "-s",  # specify the video resolution (or size)
            f"{width}x{height}",  # size of one frame
            "-pix_fmt",  # pixel format
            "bgr24",  # ... which is bgr24 (since this is opencv, which represents np array as bgr24)
            "-r",  # frame rate
            str(fps),  # the video fps
            "-i",  # the input
            "-",  # The input comes from a pipe
            "-an",  # Tells FFmpeg not to expect any audio
            "-vcodec",  # the video codec to be encoded to
            "libx264",  # which is libx264 (good for DeepLabCut/SLEAP and for most applications)
            "-pix_fmt",  # set pixel format
            "yuv420p",  # as yuv420p (usually the default)
            "-preset",  # set encoding to compression ratio
            "ultrafast",  # to ultrafast (we care more about speed than memory)
            "-tune",  # sets how to optimize the encoding process
            "film",  # we set this to film (we want a high quality video)
            "-threads",  # number of threads to use
            str(
                self.threads
                if self.threads > 0
                else multiprocessing.cpu_count()  # set the number of CPU cores to be used here
            ),  # number of CPU threads to use
            str(
                output_path
            ),  # where to save the encoded video (convert to str from pathlib object)
        ]

        ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE)

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

        while True:  # our processing loop (while the video is still there)
            ret, frame = cap.read()
            if not ret:
                break

            adjusted_frame = self.adjust_brightness_contrast(frame)
            equalized_frame = self.adaptive_histogram_equalization(adjusted_frame)
            frame_bytes = equalized_frame.tobytes()

            ffmpeg_process.stdin.write(frame_bytes)

            processed_frames += 1

            # compute video progress
            current_video_progress = (
                int((processed_frames / total_frames) * 100) if total_frames > 0 else 0
            )

            # compute processing speed
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                current_fps = processed_frames / elapsed_time
                processing_speed = current_fps / fps  # speed factor
            else:
                current_fps = 0
                processing_speed = 0

            # update progress
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

        # initialize the ThreadPoolExecutor
        # we use this for multithreading file handling
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

        # Update overall progress to 100%
        if self.progress_callback:
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
            if self.is_running:
                self.finished.emit()
            else:
                self.error.emit("Processing stopped by user.")
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.processor.stop_processing()  # Safe stop ffmpeg processes

    def stop(self):
        """
        Safely stop the processing operation
        """
        self.is_running = False
        self.processor.stop_processing()  # stop ffmpeg processes

    def is_stopped(self):
        """
        Helper function that tells us whether the process has been stopped or not
        """
        return not self.is_running

    def progress_callback(self, event_type, data):
        """
        Handles progress updates for the video processing operation by emitting corresponding signals to the GUI.

        This method processes various types of events related to the progress of video processing tasks, such as file changes,
        progress updates, overall progress, and errors. Depending on the event type, it triggers different signals to update
        the GUI with the latest information.

        Args:
        event_type (str): A string indicating the type of event. Expected values are:
            - "file_changed": Indicates that the currently processed file has changed.
            - "progress": Updates progress for the currently processed video file.
            - "overall_progress": Updates overall progress across all video files being processed.
            - "error": Indicates that an error has occurred.

        data (dict): A dictionary containing relevant data for the specified event type. The expected structure of `data`
                     varies based on the event type:
            - For "file_changed":
                {
                    "current_file": str,    # The name of the current file being processed
                    "file_index": int,      # The index of the current file in the list of files
                    "total_files": int      # The total number of files to process
                }
            - For "progress":
                {
                    "current_video_progress": int,   # The progress of the current video in percentage
                    "current_fps": float,            # The current frames per second (FPS) processing rate
                    "processing_speed": float,       # The processing speed as a multiplier (e.g., 2x)
                    "processed_frames": int,         # The number of frames processed so far
                    "total_frames": int              # The total number of frames in the video
                }
            - For "overall_progress":
                {
                    "overall_progress": int,         # The overall processing progress in percentage
                    "files_processed": int,          # The number of files processed so far
                    "total_files": int               # The total number of files to process
                }
            - For "error": A string containing the error message.

        Emits:
            - file_changed: Signal to update the current file being processed.
            - current_video_progress: Signal to update the progress of the current video.
            - overall_progress: Signal to update the overall progress of all files.
            - current_fps: Signal to update the current FPS being processed.
            - processing_speed: Signal to update the processing speed multiplier.
            - frames_processed: Signal to update the number of frames processed in the current video.
            - files_processed: Signal to update the number of files processed so far.
            - error: Signal to handle and display any error that occurred during processing.
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
        self.resize(600, 800)
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

        # video display area -- no fixed size
        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # current file label
        self.current_file_label = QLabel("Current File: None")

        # create progress labels
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

        # create speed and fps labels
        self.fps_label = QLabel("Current FPS: 0.00")
        self.speed_label = QLabel("Processing Speed: 0.00x")

        # adjust size policies for other widgets
        self.start_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.stop_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        input_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        output_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # layouts for button, boxes, etc.
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

        # current video progress layout
        current_video_progress_layout = QVBoxLayout()
        current_video_progress_layout.addWidget(self.current_video_progress_label)
        current_video_progress_layout.addWidget(self.current_video_progress_bar)

        # overall progress layout
        overall_progress_layout = QVBoxLayout()
        overall_progress_layout.addWidget(self.overall_progress_label)
        overall_progress_layout.addWidget(self.overall_progress_bar)

        # add widgets to the main layout
        layout.addLayout(input_layout)
        layout.addLayout(output_layout)
        layout.addLayout(cpu_layout)
        layout.addWidget(self.display_checkbox)
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

        self.processor = VideoProcessor(
            input_path, output_dir, display=display, threads=threads
        )

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

        This has a @pyqtSlot decorator to signify that it can receive a signal.
        """
        if self.thread and self.thread.isRunning():
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

        This has a @pyqtSlot decorator to signify that it can receive a signal (in this case, from an FFmpeg process).
        """
        self.speed_label.setText(f"Processing Speed: {speed:.2f}x")

    @pyqtSlot(int, int)
    def update_frames_processed(self, processed_frames, total_frames):
        """
        Helper function to update the number of frames that have been processed in the GUI.

        Args:
        processed_frames (int): Total number of frames that have been processed.
        total_frames (int): Total number of frames in the video.

        This has a @pyqtSlot decorator to signify that it can receive a signal (in this case, from an FFmpeg process).
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

        This has a @pyqtSlot decorator to signify that it can receive a signal (in this case, from an FFmpeg process).
        """
        percentage = (files_processed / total_files) * 100 if total_files > 0 else 0
        self.overall_progress_label.setText(
            f"Overall Progress: {percentage:.2f}% (Files: {files_processed} / {total_files})"
        )

    @pyqtSlot()
    def processing_finished(self):
        """
        Function to signify whether video processing has been completed.

        This has a @pyqtSlot decorator to signify that it can receive a signal (in this case, from an FFmpeg process).
        """
        QMessageBox.information(
            self, "Processing Finished", "Video processing is complete."
        )
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.video_label.clear()

    @pyqtSlot(str)
    def handle_error(self, error_message):
        """
        Function for basic error handling.

        Args:
        error_message (str): The error message to be displayed (this has been converted into a string by the time this function is called).

        This has a @pyqtSlot decorator to signify that it can receive a signal (in this case, from an FFmpeg process).
        """
        QMessageBox.critical(self, "Error", error_message)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.video_label.clear()

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
        event (QCloseEvent): an event generated when the user attempts to close a window (either by shortcut or explicitly)
        """
        # Ensure the thread and ffmpeg process are properly terminated when the GUI is closed
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
        if self.processor:
            self.processor.stop_processing()
        event.accept()


def main():
    app = QApplication(sys.argv)  # initialize the Qt Applicatiion
    gui = VideoProcessorGUI()  # show the GUI
    gui.show()  # we don't need an explicit .show() method to our GUI since this is a Qt Application
    sys.exit(app.exec_())  # destroy the app


if __name__ == "__main__":
    main()
