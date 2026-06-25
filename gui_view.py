import numpy as np
import math
from PyQt6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                             QGraphicsRectItem, QSizePolicy, QGraphicsLineItem, 
                             QGraphicsEllipseItem, QGraphicsSimpleTextItem)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QPixmap, QImage, QPen, QBrush, QColor
from PyQt6.QtCore import Qt, pyqtSignal, QPointF

from gui_backend import VideoDecoderThread

class VideoGraphicsView(QGraphicsView):
    # Signal emitted when a pixel on the video frame is clicked
    pixelClicked = pyqtSignal(float, float)
    # Signal emitted when a bounding box is dragged and selected: (x_min, y_min, x_max, y_max)
    boxSelected = pyqtSignal(float, float, float, float)
    
    # Signal emitted when a new frame is ready from decoder
    frame_ready = pyqtSignal(QImage, int, float)

    def __init__(self, parent=None, session=None):
        super().__init__(parent)
        self.session = session
        self.overlay_items = []
        
        # Instantiate Decoder Thread (Layer 0 dependency)
        self.decoder_thread = VideoDecoderThread()
        self.decoder_thread.frameReady.connect(self._on_decoder_frame_ready)
        self.decoder_thread.start()
        
        # Size Policy to prevent pushing the window size out
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        
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

        if self.session:
            self.session.changed.connect(self._on_session_changed)

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

    def _on_decoder_frame_ready(self, qimage, frame_idx, time_sec):
        # Default rendering behavior (if not overridden by a tool/visualizer)
        if self.session and self.session.get("override_display_image") is not None:
            pass  # Let session handler manage rendering
        else:
            self.set_image(qimage)
            
        self.frame_ready.emit(qimage, frame_idx, time_sec)

    def _on_session_changed(self, key, value):
        if key == "overlay_data":
            self.draw_overlay(value)
        elif key == "override_display_image":
            if value is not None:
                self.set_image(value)
            else:
                frame = self.session.get("current_frame") if self.session else None
                if frame is not None:
                    self.seek_frame(frame)
        elif key == "interaction_mode":
            # Example: {"mode": "add_point", "cursor": "cross", "drag": "none"}
            if value:
                mode = value.get("mode")
                cursor_type = value.get("cursor")
                drag_type = value.get("drag")
                
                self.set_box_selection_mode(mode == "box")
                if cursor_type == "cross":
                    self.viewport().setCursor(Qt.CursorShape.CrossCursor)
                else:
                    self.viewport().unsetCursor()
                    
                if drag_type == "none":
                    self.setDragMode(QGraphicsView.DragMode.NoDrag)
                elif drag_type == "scroll":
                    self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def is_playing(self):
        return self.decoder_thread._is_playing

    def has_container(self):
        return self.decoder_thread.container is not None

    def play(self):
        self.decoder_thread.play()

    def pause(self):
        self.decoder_thread.pause()

    def seek_frame(self, frame_num):
        self.decoder_thread.seek_frame(frame_num)

    def step_forward(self):
        self.decoder_thread.step_forward()

    def step_backward(self):
        self.decoder_thread.step_backward()

    def stop(self):
        self.decoder_thread.stop()

    def load_video(self, path):
        return self.decoder_thread.open_video(path)

    def get_metadata(self):
        return self.decoder_thread.get_metadata()

    def get_decoder_current_frame(self):
        return self.decoder_thread.current_frame_idx

    def clear_overlays(self):
        for item in self.overlay_items:
            try:
                self.scene.removeItem(item)
            except Exception:
                pass
        self.overlay_items.clear()

    def draw_overlay(self, overlay_data):
        self.clear_overlays()
        if not overlay_data:
            return

        points = overlay_data.get("points")
        tracks_2d = overlay_data.get("tracks_2d")
        current_frame = overlay_data.get("current_frame", 0)
        start_frame = overlay_data.get("start_frame", 0)
        end_frame = overlay_data.get("end_frame", 0)
        text_message = overlay_data.get("text")
        step = overlay_data.get("step", 1)

        def get_point_color(idx):
            colors = [
                QColor(255, 0, 0),    # Red
                QColor(0, 255, 0),    # Green
                QColor(0, 0, 255),    # Blue
                QColor(255, 255, 0),  # Yellow
                QColor(255, 0, 255),  # Magenta
                QColor(0, 255, 255),  # Cyan
                QColor(255, 165, 0),  # Orange
                QColor(128, 0, 128),  # Purple
            ]
            return colors[idx % len(colors)]

        def get_interpolated_tracks(frame_val):
            if tracks_2d is None:
                return None
            if frame_val < start_frame or frame_val > end_frame:
                return None
            float_idx = (frame_val - start_frame) / step
            k = int(math.floor(float_idx))
            if k < 0:
                return tracks_2d[0]
            if k >= len(tracks_2d) - 1:
                return tracks_2d[-1]
            ratio = float_idx - k
            pos_prev = tracks_2d[k]
            pos_next = tracks_2d[k + 1]
            return (1.0 - ratio) * pos_prev + ratio * pos_next

        if tracks_2d is not None:
            interpolated = get_interpolated_tracks(current_frame)
            if interpolated is not None:
                num_points = interpolated.shape[0]
                trail_len = 10
                for pt_idx in range(num_points):
                    past_pts = []
                    for offset in range(trail_len + 1):
                        f_past = current_frame - offset
                        if f_past < start_frame:
                            break
                        coords_past = get_interpolated_tracks(f_past)
                        if coords_past is not None:
                            past_pts.append(coords_past[pt_idx])

                    for idx, pt in enumerate(past_pts):
                        if idx == 0:
                            continue
                        pt_prev = past_pts[idx - 1]
                        opacity = max(0.15, 1.0 - (idx / trail_len))
                        base_color = get_point_color(pt_idx)
                        trail_color = QColor(base_color.red(), base_color.green(), base_color.blue(), int(opacity * 255))
                        pen = QPen(trail_color, 2)
                        line = QGraphicsLineItem(pt_prev[0], pt_prev[1], pt[0], pt[1])
                        line.setPen(pen)
                        self.scene.addItem(line)
                        self.overlay_items.append(line)

                    x_curr, y_curr = interpolated[pt_idx]
                    marker = QGraphicsEllipseItem(x_curr - 5, y_curr - 5, 10, 10)
                    marker.setBrush(QBrush(get_point_color(pt_idx)))
                    marker.setPen(QPen(QColor(0, 0, 0), 1))
                    self.scene.addItem(marker)
                    self.overlay_items.append(marker)
        else:
            if current_frame == start_frame and points:
                for pt_idx, (x, y) in enumerate(points):
                    marker = QGraphicsEllipseItem(x - 5, y - 5, 10, 10)
                    marker.setBrush(QBrush(get_point_color(pt_idx)))
                    marker.setPen(QPen(QColor(0, 0, 0), 1))
                    self.scene.addItem(marker)
                    self.overlay_items.append(marker)
            elif text_message:
                text = QGraphicsSimpleTextItem(text_message)
                text.setBrush(QBrush(QColor(255, 255, 255)))
                text.setPos(15, 15)
                self.scene.addItem(text)
                self.overlay_items.append(text)

