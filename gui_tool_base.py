from PyQt6.QtWidgets import QWidget

class BaseTool(QWidget):
    """Base class for all editor tools."""
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

    def get_name(self):
        return "Base Tool"

    def on_frame_changed(self, frame_idx, current_time_sec):
        """Called when the video frame updates."""
        pass
        
    def on_video_loaded(self, metadata):
        """Called when a new video is loaded."""
        pass
