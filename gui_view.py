from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QPixmap, QImage, QPen, QBrush, QColor
from PyQt6.QtCore import Qt, pyqtSignal, QPointF

class VideoGraphicsView(QGraphicsView):
    # Signal emitted when a pixel on the video frame is clicked
    pixelClicked = pyqtSignal(float, float)
    # Signal emitted when a bounding box is dragged and selected: (x_min, y_min, x_max, y_max)
    boxSelected = pyqtSignal(float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Setup GPU acceleration
        self.setViewport(QOpenGLWidget())
        
        # Setup Scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Setup background video item
        self.video_item = QGraphicsPixmapItem()
        self.scene.addItem(self.video_item)
        
        # Interaction settings
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        
        # Rendering hints
        from PyQt6.QtGui import QPainter
        self.setRenderHints(
            self.renderHints() | 
            QPainter.RenderHint.Antialiasing | 
            QPainter.RenderHint.SmoothPixmapTransform
        )
        
        self.current_image = None
        self._fit_to_view = True
        
        # Box drawing properties
        self.box_selection_mode = False
        self._is_drawing_box = False
        self._box_start_pos = None
        self._box_item = None

    def set_image(self, qimage: QImage):
        """Update the displayed video frame."""
        self.current_image = qimage
        pixmap = QPixmap.fromImage(qimage)
        self.video_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.video_item.boundingRect())
        
        if self._fit_to_view:
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_to_view and self.current_image:
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        # Implement zooming
        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor
        
        # Disable auto-fit once user starts zooming manually
        self._fit_to_view = False

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.scale(zoom_factor, zoom_factor)
        
    def reset_view(self):
        """Reset view to fit the video."""
        self._fit_to_view = True
        if self.current_image:
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_box_selection_mode(self, enabled: bool):
        """Toggle box selection mode which changes the cursor and drag behavior."""
        self.box_selection_mode = enabled
        self._is_drawing_box = False
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().unsetCursor()
            if self._box_item:
                try:
                    self.scene.removeItem(self._box_item)
                except Exception:
                    pass
                self._box_item = None

    def mousePressEvent(self, event):
        if self.box_selection_mode and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            v_rect = self.video_item.boundingRect()
            if v_rect.contains(scene_pos):
                self._is_drawing_box = True
                self._box_start_pos = scene_pos
                
                if self._box_item:
                    try:
                        self.scene.removeItem(self._box_item)
                    except Exception:
                        pass
                
                self._box_item = QGraphicsRectItem()
                self._box_item.setPen(QPen(QColor(0, 120, 240), 2, Qt.PenStyle.DashLine))
                self._box_item.setBrush(QBrush(QColor(0, 120, 240, 40)))
                self.scene.addItem(self._box_item)
                self._box_item.setRect(scene_pos.x(), scene_pos.y(), 0, 0)
                event.accept()
                return
        
        elif not self.box_selection_mode and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            if self.video_item.contains(scene_pos):
                self.pixelClicked.emit(scene_pos.x(), scene_pos.y())
                event.accept()
                return
                
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.box_selection_mode and self._is_drawing_box and self._box_start_pos:
            scene_pos = self.mapToScene(event.pos())
            v_rect = self.video_item.boundingRect()
            
            # Clamp to video boundaries
            cx = max(0.0, min(scene_pos.x(), v_rect.width()))
            cy = max(0.0, min(scene_pos.y(), v_rect.height()))
            
            x1, y1 = self._box_start_pos.x(), self._box_start_pos.y()
            x2, y2 = cx, cy
            
            rx = min(x1, x2)
            ry = min(y1, y2)
            rw = abs(x2 - x1)
            rh = abs(y2 - y1)
            
            if self._box_item:
                self._box_item.setRect(rx, ry, rw, rh)
            event.accept()
            return
            
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.box_selection_mode and self._is_drawing_box and event.button() == Qt.MouseButton.LeftButton:
            self._is_drawing_box = False
            scene_pos = self.mapToScene(event.pos())
            v_rect = self.video_item.boundingRect()
            
            # Clamp to video boundaries
            cx = max(0.0, min(scene_pos.x(), v_rect.width()))
            cy = max(0.0, min(scene_pos.y(), v_rect.height()))
            
            x1, y1 = self._box_start_pos.x(), self._box_start_pos.y()
            x2, y2 = cx, cy
            
            if self._box_item:
                try:
                    self.scene.removeItem(self._box_item)
                except Exception:
                    pass
                self._box_item = None
                
            rx_min, rx_max = min(x1, x2), max(x1, x2)
            ry_min, ry_max = min(y1, y2), max(y1, y2)
            
            if (rx_max - rx_min) > 3 and (ry_max - ry_min) > 3:
                self.boxSelected.emit(rx_min, ry_min, rx_max, ry_max)
            
            event.accept()
            return
            
        super().mouseReleaseEvent(event)

