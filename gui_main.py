import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QPushButton, QLabel, 
                             QFileDialog, QComboBox, QStackedWidget, QScrollArea, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal

from gui_view import VideoGraphicsView
from gui_tools import EXPOSED_TOOLS
from gui_state import SessionState

class VideoEditorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Video Editor")
        self.resize(1024, 768)
        
        # Initialize Shared Session State (Layer 1 uses it, but it's part of core)
        self.session = SessionState()
        self.session.changed.connect(self._on_session_changed)
        
        self._init_ui()
        
        # Register decoder in state so background threads in Layer 0/1 can use it
        self.session.set("decoder", self.video_view.decoder_thread)
        
        self._init_tools()

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
        
        # Preview View (Layer 1 component)
        self.video_view = VideoGraphicsView(session=self.session)
        self.video_view.frame_ready.connect(self.on_frame_ready)
        
        # Bridge mouse events from view to session keys reactively
        self.video_view.pixelClicked.connect(lambda x, y: self.session.set("pixel_clicked", (x, y)))
        self.video_view.boxSelected.connect(lambda x1, y1, x2, y2: self.session.set("box_selected", (x1, y1, x2, y2)))
        
        top_layout.addWidget(self.video_view, stretch=3)
        
        # Tools Right Panel
        tools_layout = QVBoxLayout()
        self.tool_selector = QComboBox()
        self.tool_stack = QStackedWidget()
        
        # Wrap self.tool_stack in a scroll area to prevent pushing the window size
        self.tool_scroll_area = QScrollArea()
        self.tool_scroll_area.setWidgetResizable(True)
        self.tool_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.tool_scroll_area.setWidget(self.tool_stack)
        
        tools_layout.addWidget(QLabel("<b>Tools</b>"))
        tools_layout.addWidget(self.tool_selector)
        tools_layout.addWidget(self.tool_scroll_area)
        
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
        self.btn_step_back.clicked.connect(self.video_view.step_backward)
        self.btn_step_fwd.clicked.connect(self.video_view.step_forward)
        
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
        
        for cls in EXPOSED_TOOLS:
            tool = cls(self.session, parent=self)
            self.tools.append(tool)
            self.tool_selector.addItem(tool.get_name())
            self.tool_stack.addWidget(tool)
            
        self.tool_selector.currentIndexChanged.connect(self.tool_stack.setCurrentIndex)
        
        self.current_tool = self.tools[0]
        self.tool_selector.currentIndexChanged.connect(self._on_tool_changed)

    def _on_tool_changed(self, index):
        self.current_tool = self.tools[index]

    def _on_session_changed(self, key, value):
        if key == "trim_range":
            if value:
                start_frame, end_frame = value
                self.timeline_slider.setMinimum(start_frame)
                self.timeline_slider.setMaximum(end_frame)
                
                current = self.timeline_slider.value()
                if current < start_frame or current > end_frame:
                    self.video_view.seek_frame(start_frame)
        elif key == "seek_frame":
            if value is not None:
                self.video_view.seek_frame(value)
        elif key == "video_path":
            if value and value != getattr(self, "current_video_path", ""):
                self.load_video(value)

    def open_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if filename:
            self.load_video(filename)

    def load_video(self, filename):
        self.video_view.pause()
        self.btn_play_pause.setText("Play")
        
        self.current_video_path = filename
        self.session.set("video_path", filename)
        
        # Clear old video session data to prevent leakage into the new session
        self.session.set("preprocess_npz", None)
        self.session.set("preprocess_json", None)
        self.session.set("tracking_result_npz", None)
        self.session.set("overlay_data", None)
        
        if self.video_view.load_video(filename):
            metadata = self.video_view.get_metadata()
            self.session.set("video_metadata", metadata)
            
            # Reset timeline state in session so all tools sync reactively
            self.session.set("trim_range", (0, metadata['total_frames'] - 1))
            self.session.set("current_frame", 0)
            
            # Reset timeline slider locally
            self.timeline_slider.setEnabled(True)
            self.timeline_slider.setMinimum(0)
            self.timeline_slider.setMaximum(metadata['total_frames'] - 1)
            self.timeline_slider.setValue(0)
            
            # Seek to first frame
            self.video_view.seek_frame(0)
            return True
        return False

    def toggle_play(self):
        if not self.video_view.has_container():
            return
            
        if self.video_view.is_playing():
            self.video_view.pause()
            self.btn_play_pause.setText("Play")
        else:
            self.video_view.play()
            self.btn_play_pause.setText("Pause")

    def on_slider_moved(self, value):
        self.video_view.seek_frame(value)

    def on_frame_ready(self, qimage, frame_idx, time_sec):
        # Update slider without triggering seek
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(frame_idx)
        self.timeline_slider.blockSignals(False)
        
        # Update labels
        metadata = self.video_view.get_metadata()
        tot_frames = metadata.get('total_frames', 0)
        tot_sec = metadata.get('duration_sec', 0)
        
        self.lbl_frame.setText(f"Frame: {frame_idx} / {tot_frames}")
        self.lbl_time.setText(f"{self.format_time(time_sec)} / {self.format_time(tot_sec)}")
        
        # Stop playback if we reached the end of the restricted timeline or absolute end
        is_playing = self.video_view.is_playing()
        if is_playing and (frame_idx >= self.timeline_slider.maximum() or frame_idx >= tot_frames - 1):
            self.video_view.pause()
            self.btn_play_pause.setText("Play")
        
        # Publish current frame index to session state so tools react
        self.session.set("current_frame", frame_idx)

    @staticmethod
    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def closeEvent(self, event):
        self.video_view.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoEditorApp()
    window.show()
    sys.exit(app.exec())
