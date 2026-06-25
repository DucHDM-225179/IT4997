from PyQt6.QtWidgets import QWidget

class BaseTool(QWidget):
    """Base class for all editor tools."""
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        
        # Share session changed notifications reactively across all tools
        self.session.changed.connect(self._on_session_changed)

    def get_name(self):
        return "Base Tool"

    def _on_session_changed(self, key, value):
        """Virtual method to be overridden by subclasses."""
        pass
