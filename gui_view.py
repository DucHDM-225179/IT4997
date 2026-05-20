from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, pyqtSignal, QPointF

class VideoGraphicsView(QGraphicsView):
    # Signal emitted when a pixel on the video frame is clicked
    pixelClicked = pyqtSignal(float, float)

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

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            # Emit coordinate if it is inside the video frame boundaries
            if self.video_item.contains(scene_pos):
                self.pixelClicked.emit(scene_pos.x(), scene_pos.y())
                # If dragging isn't needed, we can swallow the event. But let's swallow only if
                # a listener acts on it, or just swallow to avoid drag during point placement.
                event.accept()
                return
        super().mousePressEvent(event)

