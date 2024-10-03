import sys
import cv2
import numpy as np
import time
import subprocess
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QLineEdit, QVBoxLayout, QHBoxLayout, QCheckBox, QProgressBar, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

"""
Function for processing IR videos obtained from MoSeq analysis for visualization
This allows for seamless use with DeepLabCut, SLEAP, or any keypoint tracking programs
"""

class VideoProcessor:
    def __init__(self, input_path, output_dir, display=True, progress_callback=None, frame_callback=None):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.display = display
        self.progress_callback = progress_callback
        self.frame_callback = frame_callback
        self.ffmpeg_process = None

    @staticmethod
    def adjust_brightness_contrast(frame, brightness=150, contrast=210):
        alpha = (contrast + 100) / 100.0
        beta = brightness
        adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return adjusted

    @staticmethod
    def adaptive_histogram_equalization(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        return equalized_bgr

    def process_folder(self, stop_check=None):
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        files = []
        if self.input_path.is_dir():
            files = list(self.input_path.glob('*.avi')) + list(self.input_path.glob('*.mp4'))
        elif self.input_path.is_file():
            if self.input_path.suffix.lower() in ['.avi', '.mp4']:
                files = [self.input_path]
            else:
                raise ValueError("Input file is not a supported video format (.avi or .mp4).")
        else:
            raise ValueError("Invalid input path.")

        total_files = len(files)
        if total_files == 0:
            raise FileNotFoundError("No video files found in the input directory.")

        for i, file in enumerate(files):
            if stop_check and stop_check():
                break
            file_index = i + 1
            self.process(file, file_index, total_files, stop_check)
            # update progress
            overall_progress = int((file_index / total_files) * 100)
            if self.progress_callback:
                self.progress_callback('overall_progress', {
                    'overall_progress': overall_progress,
                    'files_processed': file_index,
                    'total_files': total_files
                })

    def process(self, file, file_index, total_files, stop_check=None):
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

        output_name = file.stem + '_processed.mp4'
        output_path = self.output_dir / output_name

        # Start time for processing speed calculation
        start_time = time.time()
        processed_frames = 0

        # start FFmpeg process
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',  # size of one frame
            '-pix_fmt', 'bgr24',
            '-r', str(fps),  # fps
            '-i', '-',  # The input comes from a pipe
            '-an',  # Tells FFMPEG not to expect any audio
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-tune', 'film',
            str(output_path)
        ]

        self.ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE)

        # Notify about the current file
        if self.progress_callback:
            self.progress_callback('file_changed', {
                'current_file': file.name,
                'file_index': file_index,
                'total_files': total_files
            })

        while True:
            if stop_check and stop_check():
                break
            ret, frame = cap.read()
            if not ret:
                break

            adjusted_frame = self.adjust_brightness_contrast(frame)
            equalized_frame = self.adaptive_histogram_equalization(adjusted_frame)
            frame_bytes = equalized_frame.tobytes()

            # we need to manually organize the ffmpeg process so that we can call a stop to it later on, otherwise we get errors where we're stopping it in the gui but the thread is still being used by ffmpeg

            self.ffmpeg_process.stdin.write(frame_bytes) 

            processed_frames += 1

            # compute video progress
            current_video_progress = int((processed_frames / total_frames) * 100) if total_frames > 0 else 0

            # compute processing speed
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                current_fps = processed_frames / elapsed_time
                processing_speed = current_fps / fps  # speed factor (e.g., 2x, 3x)
            else:
                current_fps = 0
                processing_speed = 0

            # update progress
            if self.progress_callback:
                self.progress_callback('progress', {
                    'current_video_progress': current_video_progress,
                    'current_fps': current_fps,
                    'processing_speed': processing_speed,
                    'processed_frames': processed_frames,
                    'total_frames': total_frames
                })

            # update frame in GUI
            if self.display and self.frame_callback:
                self.frame_callback(equalized_frame)

        cap.release()
        self.ffmpeg_process.stdin.close()
        self.ffmpeg_process.wait()
        self.ffmpeg_process = None

    def stop_processing(self):
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None

class VideoProcessorThread(QThread):
    file_changed = pyqtSignal(str)
    current_video_progress = pyqtSignal(int)
    overall_progress = pyqtSignal(int)
    current_fps = pyqtSignal(float)
    processing_speed = pyqtSignal(float)
    frames_processed = pyqtSignal(int, int)  # processed_frames, total_frames
    files_processed = pyqtSignal(int, int)   # files_processed, total_files
    frame = pyqtSignal(np.ndarray)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.processor.progress_callback = self.progress_callback
        self.processor.frame_callback = self.emit_frame
        self.is_running = True

    def run(self):
        try:
            self.processor.process_folder(stop_check=self.is_stopped)
            if self.is_running:
                self.finished.emit()
            else:
                self.error.emit("Processing stopped by user.")
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.processor.stop_processing()  # Safe stop ffmpeg process

    def stop(self):
        self.is_running = False
        self.processor.stop_processing()  # stop ffmpeg process

    def is_stopped(self):
        return not self.is_running

    def progress_callback(self, event_type, data):
        if event_type == 'file_changed':
            self.file_changed.emit(data['current_file'])
            self.files_processed.emit(data['file_index'] - 1, data['total_files'])
        elif event_type == 'progress':
            self.current_video_progress.emit(data['current_video_progress'])
            self.current_fps.emit(data['current_fps'])
            self.processing_speed.emit(data['processing_speed'])
            self.frames_processed.emit(data['processed_frames'], data['total_frames'])
        elif event_type == 'overall_progress':
            self.overall_progress.emit(data['overall_progress'])
            self.files_processed.emit(data['files_processed'], data['total_files'])

    def emit_frame(self, frame):
        self.frame.emit(frame)

class VideoProcessorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Processor")
        self.resize(500, 700)
        self.processor = None
        self.thread = None
        self.initUI()

    def initUI(self):
        # Create widgets
        input_label = QLabel('Input Path:')
        self.input_line = QLineEdit()
        input_button = QPushButton('Browse')
        input_button.clicked.connect(self.select_input_path)

        output_label = QLabel('Output Directory:')
        self.output_line = QLineEdit()
        output_button = QPushButton('Browse')
        output_button.clicked.connect(self.select_output_dir)

        self.display_checkbox = QCheckBox('Display Video')
        self.display_checkbox.setChecked(True)

        self.start_button = QPushButton('Start Processing')
        self.start_button.clicked.connect(self.start_processing)

        self.stop_button = QPushButton('Stop Processing')
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)

        # video display area -- no fixed size
        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # current file label
        self.current_file_label = QLabel('Current File: None')

        # create progress labels
        self.current_video_progress_label = QLabel('Current Video Progress: 0% (Frames: 0 / 0)')
        self.current_video_progress_bar = QProgressBar()
        self.current_video_progress_bar.setAlignment(Qt.AlignCenter)
        self.current_video_progress_bar.setFormat('%p%')

        # Combined label for overall progress and files processed
        self.overall_progress_label = QLabel('Overall Progress: 0% (Files: 0 / 0)')
        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setAlignment(Qt.AlignCenter)
        self.overall_progress_bar.setFormat('%p%')

        # create speed and fps labels
        self.fps_label = QLabel('Current FPS: 0.00')
        self.speed_label = QLabel('Processing Speed: 0.00x')

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
        path = QFileDialog.getExistingDirectory(self, 'Select Input Directory')
        if path:
            self.input_line.setText(path)

    @pyqtSlot()
    def select_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, 'Select Output Directory')
        if path:
            self.output_line.setText(path)

    @pyqtSlot()
    def start_processing(self):
        input_path = self.input_line.text()
        output_dir = self.output_line.text()
        display = self.display_checkbox.isChecked()

        if not input_path or not output_dir:
            QMessageBox.warning(self, 'Input Error', 'Please specify both input path and output directory.')
            return

        # Reset progress bars and labels
        self.current_video_progress_bar.setValue(0)
        self.overall_progress_bar.setValue(0)
        self.fps_label.setText('Current FPS: 0.00')
        self.speed_label.setText('Processing Speed: 0.00x')
        self.current_file_label.setText('Current File: None')
        self.current_video_progress_label
        self.current_video_progress_label.setText('Current Video Progress: 0% (Frames: 0 / 0)')
        self.overall_progress_label.setText('Overall Progress: 0% (Files: 0 / 0)')

        self.processor = VideoProcessor(input_path, output_dir, display=display)

        # set start and stop button defaults
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # start processing in a separate thread
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
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()  # Wait for the thread to finish
            self.processing_finished()

    @pyqtSlot(str)
    def update_current_file(self, current_file):
        self.current_file_label.setText(f'Current File: {current_file}')

    @pyqtSlot(int)
    def update_current_video_progress(self, value):
        self.current_video_progress_bar.setValue(value)
        # Update the label text to include the progress percentage
        current_text = self.current_video_progress_label.text()
        updated_text = f'Current Video Progress: {value}%' + current_text[current_text.index(' (Frames:'):]
        self.current_video_progress_label.setText(updated_text)

    @pyqtSlot(int)
    def update_overall_progress(self, value):
        self.overall_progress_bar.setValue(value)
        # Update the label text to include the progress percentage
        current_text = self.overall_progress_label.text()
        updated_text = f'Overall Progress: {value}%' + current_text[current_text.index(' (Files:'):]
        self.overall_progress_label.setText(updated_text)

    @pyqtSlot(float)
    def update_fps(self, fps):
        self.fps_label.setText(f'Current FPS: {fps:.2f}')

    @pyqtSlot(float)
    def update_speed(self, speed):
        self.speed_label.setText(f'Processing Speed: {speed:.2f}x')

    @pyqtSlot(int, int)
    def update_frames_processed(self, processed_frames, total_frames):
        percentage = (processed_frames / total_frames) * 100 if total_frames > 0 else 0
        self.current_video_progress_label.setText(f'Current Video Progress: {percentage:.2f}% (Frames: {processed_frames} / {total_frames})')

    @pyqtSlot(int, int)
    def update_files_processed(self, files_processed, total_files):
        percentage = (files_processed / total_files) * 100 if total_files > 0 else 0
        self.overall_progress_label.setText(f'Overall Progress: {percentage:.2f}% (Files: {files_processed} / {total_files})')

    @pyqtSlot()
    def processing_finished(self):
        # Update files processed to total
        self.update_files_processed(self.overall_progress_bar.value(), self.overall_progress_bar.maximum())
        QMessageBox.information(self, 'Processing Finished', 'Video processing is complete.')
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.video_label.clear()

    @pyqtSlot(str)
    def handle_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.video_label.clear()

    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        # Convert the frame to QImage and display it
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        # Ensure the thread and ffmpeg process are properly terminated when the GUI is closed
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
        if self.processor:
            self.processor.stop_processing()
        event.accept()

def main():
    app = QApplication(sys.argv)
    gui = VideoProcessorGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
