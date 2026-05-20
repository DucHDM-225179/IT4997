import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QPushButton, QLabel, 
                             QFileDialog, QComboBox, QStackedWidget)
from PyQt6.QtCore import Qt

from gui_backend import VideoDecoderThread
from gui_view import VideoGraphicsView
from gui_tools import TrimTool, PreprocessTool, ProcessVideoTool


class VideoEditorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Video Editor")
        self.resize(1024, 768)
        
        # Core Components
        self.decoder_thread = VideoDecoderThread()
        self.decoder_thread.frameReady.connect(self.on_frame_ready)
        
        self._init_ui()
        self._init_tools()
        
        self.decoder_thread.start()

    def _init_ui(self):
        # Menu Bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        open_action = file_menu.addAction("Open Video")
        open_action.triggered.connect(self.open_video)
        
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top Area (Video Preview + Tools)
        top_layout = QHBoxLayout()
        
        # Preview
        self.video_view = VideoGraphicsView()
        top_layout.addWidget(self.video_view, stretch=3)
        
        # Tools Right Panel
        tools_layout = QVBoxLayout()
        self.tool_selector = QComboBox()
        self.tool_stack = QStackedWidget()
        
        tools_layout.addWidget(QLabel("<b>Tools</b>"))
        tools_layout.addWidget(self.tool_selector)
        tools_layout.addWidget(self.tool_stack)
        
        tools_widget = QWidget()
        tools_widget.setLayout(tools_layout)
        tools_widget.setMinimumWidth(250)
        top_layout.addWidget(tools_widget, stretch=1)
        
        main_layout.addLayout(top_layout, stretch=1)
        
        # Bottom Area (Timeline)
        bottom_layout = QVBoxLayout()
        
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.sliderMoved.connect(self.on_slider_moved)
        
        controls_layout = QHBoxLayout()
        self.btn_play_pause = QPushButton("Play")
        self.btn_step_back = QPushButton("< Step")
        self.btn_step_fwd = QPushButton("Step >")
        
        self.btn_play_pause.clicked.connect(self.toggle_play)
        self.btn_step_back.clicked.connect(self.decoder_thread.step_backward)
        self.btn_step_fwd.clicked.connect(self.decoder_thread.step_forward)
        
        self.lbl_time = QLabel("00:00:00 / 00:00:00")
        self.lbl_frame = QLabel("Frame: 0 / 0")
        
        controls_layout.addWidget(self.btn_play_pause)
        controls_layout.addWidget(self.btn_step_back)
        controls_layout.addWidget(self.btn_step_fwd)
        controls_layout.addStretch()
        controls_layout.addWidget(self.lbl_time)
        controls_layout.addWidget(self.lbl_frame)
        
        bottom_layout.addWidget(self.timeline_slider)
        bottom_layout.addLayout(controls_layout)
        
        main_layout.addLayout(bottom_layout)

    def _init_tools(self):
        self.tools = []
        
        # Initialize tools
        trim_tool = TrimTool(self)
        trim_tool.trim_applied.connect(self.apply_timeline_restriction)
        
        preprocess_tool = PreprocessTool(self)
        process_video_tool = ProcessVideoTool(self)
        
        self.tools.append(trim_tool)
        self.tools.append(preprocess_tool)
        self.tools.append(process_video_tool)

        
        for tool in self.tools:
            self.tool_selector.addItem(tool.get_name())
            self.tool_stack.addWidget(tool)
            
        self.tool_selector.currentIndexChanged.connect(self.tool_stack.setCurrentIndex)
        
        self.current_tool = self.tools[0]
        self.tool_selector.currentIndexChanged.connect(self._on_tool_changed)

    def _on_tool_changed(self, index):
        self.current_tool = self.tools[index]

    def open_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if filename:
            self.load_video(filename)

    def load_video(self, filename):
        self.decoder_thread.pause()
        self.btn_play_pause.setText("Play")
        
        # Save path for other tools
        self.current_video_path = filename
        
        if self.decoder_thread.open_video(filename):
            metadata = self.decoder_thread.get_metadata()
            
            # Reset timeline
            self.timeline_slider.setEnabled(True)
            self.timeline_slider.setMinimum(0)
            self.timeline_slider.setMaximum(metadata['total_frames'] - 1)
            self.timeline_slider.setValue(0)
            
            # Inform tools
            for tool in self.tools:
                tool.on_video_loaded(metadata)
                
            # Seek to first frame
            self.decoder_thread.seek_frame(0)
            return True
        return False

    def toggle_play(self):
        if not self.decoder_thread.container:
            return
            
        if self.decoder_thread._is_playing:
            self.decoder_thread.pause()
            self.btn_play_pause.setText("Play")
        else:
            self.decoder_thread.play()
            self.btn_play_pause.setText("Pause")

    def on_slider_moved(self, value):
        self.decoder_thread.seek_frame(value)

    def on_frame_ready(self, qimage, frame_idx, time_sec):
        # Update view
        self.video_view.set_image(qimage)
        
        # Update slider without triggering seek
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(frame_idx)
        self.timeline_slider.blockSignals(False)
        
        # Update labels
        metadata = self.decoder_thread.get_metadata()
        tot_frames = metadata.get('total_frames', 0)
        tot_sec = metadata.get('duration_sec', 0)
        
        self.lbl_frame.setText(f"Frame: {frame_idx} / {tot_frames}")
        self.lbl_time.setText(f"{self.format_time(time_sec)} / {self.format_time(tot_sec)}")
        
        # Stop playback if we reached the end of the restricted timeline
        if self.decoder_thread._is_playing and frame_idx >= self.timeline_slider.maximum():
            self.decoder_thread.pause()
            self.btn_play_pause.setText("Play")
            
        # Stop playback if we reached the total end
        elif self.decoder_thread._is_playing and frame_idx >= tot_frames - 1:
            self.decoder_thread.pause()
            self.btn_play_pause.setText("Play")
        
        # Notify current tool
        if self.current_tool:
            self.current_tool.on_frame_changed(frame_idx, time_sec)

    def apply_timeline_restriction(self, start_frame, end_frame):
        """Restricts the slider to the given range."""
        self.timeline_slider.setMinimum(start_frame)
        self.timeline_slider.setMaximum(end_frame)
        
        # Snap current position if outside bounds
        current = self.timeline_slider.value()
        if current < start_frame or current > end_frame:
            self.decoder_thread.seek_frame(start_frame)

    @staticmethod
    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def closeEvent(self, event):
        self.decoder_thread.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoEditorApp()
    window.show()
    sys.exit(app.exec())
