from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from gui_tool_base import BaseTool

class TrimTool(BaseTool):
    """Tool to define an active processing window (trim)."""
    
    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        
        self.start_frame = 0
        self.end_frame = 0
        self.total_frames = 0
        self.current_frame = 0
        
        # Setup UI
        layout = QVBoxLayout(self)
        
        self.start_label = QLabel("Start: 0")
        self.end_label = QLabel("End: 0")
        
        self.btn_set_start = QPushButton("Set Start")
        self.btn_set_end = QPushButton("Set End")
        self.btn_trim = QPushButton("Trim")
        
        # Style Trim button a bit differently
        self.btn_trim.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        
        btn_layout1 = QHBoxLayout()
        btn_layout1.addWidget(self.btn_set_start)
        btn_layout1.addWidget(self.start_label)
        
        btn_layout2 = QHBoxLayout()
        btn_layout2.addWidget(self.btn_set_end)
        btn_layout2.addWidget(self.end_label)
        
        layout.addLayout(btn_layout1)
        layout.addLayout(btn_layout2)
        layout.addWidget(self.btn_trim)
        layout.addStretch()
        
        # Connect signals
        self.btn_set_start.clicked.connect(self._on_set_start)
        self.btn_set_end.clicked.connect(self._on_set_end)
        self.btn_trim.clicked.connect(self._on_trim)

        # Sync initial state if video metadata is already loaded
        meta = self.session.get("video_metadata")
        if meta:
            self.total_frames = meta.get('total_frames', 0)
            trim_range = self.session.get("trim_range")
            if trim_range:
                self.start_frame, self.end_frame = trim_range
            else:
                self.start_frame = 0
                self.end_frame = max(0, self.total_frames - 1)
            self._update_labels()
            
        current = self.session.get("current_frame")
        if current is not None:
            self.current_frame = current
        
    def get_name(self):
        return "Video Trimming"

    def _on_session_changed(self, key, value):
        if key == "video_metadata":
            if value:
                self.total_frames = value.get('total_frames', 0)
                self.start_frame = 0
                self.end_frame = max(0, self.total_frames - 1)
                self._update_labels()
        elif key == "current_frame":
            if value is not None:
                self.current_frame = value
        elif key == "trim_range":
            if value:
                self.start_frame, self.end_frame = value
                self._update_labels()
         
    def _on_set_start(self):
        if self.current_frame <= self.end_frame or self.end_frame == 0:
            self.start_frame = self.current_frame
            # Ensure end is valid
            if self.end_frame < self.start_frame:
                self.end_frame = self.total_frames - 1
            self._update_labels()
 
    def _on_set_end(self):
        if self.current_frame >= self.start_frame:
            self.end_frame = self.current_frame
            self._update_labels()
            
    def _update_labels(self):
        self.start_label.setText(f"Start: {self.start_frame}")
        self.end_label.setText(f"End: {self.end_frame}")
        
    def _on_trim(self):
        """Apply the trim, which will restrict the timeline."""
        self.session.set("trim_range", (self.start_frame, self.end_frame))
