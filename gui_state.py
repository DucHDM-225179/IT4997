from PyQt6.QtCore import QObject, pyqtSignal

class SessionState(QObject):
    """
    Extensible domain session state shared across presentation components.
    Provides reactive data sharing using a key-value change notification model.
    """
    changed = pyqtSignal(str, object)

    def __init__(self):
        super().__init__()
        self._data = {}

    def set(self, key: str, value):
        if key not in self._data or not self._safe_equals(self._data[key], value):
            self._data[key] = value
            self.changed.emit(key, value)

    def _safe_equals(self, a, b):
        if a is b:
            return True
        if (a is None) != (b is None):
            return False
        if type(a) != type(b):
            return False
            
        try:
            import numpy as np
            if isinstance(a, np.ndarray):
                return np.array_equal(a, b)
        except ImportError:
            pass
            
        if isinstance(a, dict):
            if len(a) != len(b):
                return False
            for k in a:
                if k not in b:
                    return False
                if not self._safe_equals(a[k], b[k]):
                    return False
            return True
            
        if isinstance(a, (list, tuple, set)):
            if len(a) != len(b):
                return False
            if isinstance(a, set):
                return a == b
            for item1, item2 in zip(a, b):
                if not self._safe_equals(item1, item2):
                    return False
            return True
            
        try:
            return bool(a == b)
        except Exception:
            return False


    def get(self, key: str, default=None):
        return self._data.get(key, default)
