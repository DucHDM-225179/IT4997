# gui_tool_adding_object.py
import os
import json
import numpy as np
import trimesh

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSlider, 
                             QPushButton, QLabel, QFileDialog, QLineEdit, 
                             QMessageBox, QGroupBox, QGridLayout, QGraphicsView,
                             QCheckBox)
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QColor, QBrush, QPen, QImage, QPixmap, QPainter, QPolygonF
from PyQt6.QtWidgets import (QGraphicsEllipseItem, QGraphicsLineItem, 
                             QGraphicsPixmapItem, QGraphicsSimpleTextItem)

from gui_tool_base import BaseTool
from gui_tool_mesh_transform import MeshTransformWidget

class AddingObjectTool(BaseTool):
    """Tool to load a 3D OBJ, constrain its position and rotation to tracked points, and render it."""
    
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        
        self.video_path = ""
        self.overlay_items = []
        self.mesh_pixmap_item = None
        
        # Caches for point centers and rotations
        self.cached_centers = {}
        self.cached_rotations = {}
        self.last_coords_id = None
        
        self._init_ui()
        
        # Load default teapot if present
        self.mesh_widget.load_default_mesh_if_exist()
        self._sync_status_label()

    def get_name(self):
        return "Add 3D Object"

    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Unified Mesh transform widget
        self.mesh_widget = MeshTransformWidget(self, group_title="Transform Adjustments")
        self.mesh_widget.set_snap_button_text("Snap Center to Point Cloud Center")
        self.mesh_widget.changed.connect(self.update_overlay)
        self.mesh_widget.mesh_loaded.connect(self._on_mesh_loaded)
        layout.addWidget(self.mesh_widget)
        
        # Status
        self.lbl_status = QLabel("Status: Ready. Please load point tracking results first.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #555555;")
        layout.addWidget(self.lbl_status)
        
        layout.addStretch()

    def _on_mesh_loaded(self, success, name_or_error):
        if success:
            self.lbl_status.setText(f"Status: Loaded OBJ mesh {name_or_error}")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.lbl_status.setText(f"Status: Failed to load OBJ mesh: {name_or_error}")
            self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")

    def _sync_status_label(self):
        obj_path = self.mesh_widget.get_obj_path()
        if obj_path:
            self.lbl_status.setText(f"Status: Loaded OBJ mesh {os.path.basename(obj_path)}")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.lbl_status.setText("Status: Ready. Please load point tracking results first.")
            self.lbl_status.setStyleSheet("color: #555555;")

    def showEvent(self, event):
        super().showEvent(event)
        self.video_path = getattr(self.main_window, 'current_video_path', '')
        
        self.mesh_widget.sync_from_shared()
        self._sync_status_label()
        
        # Ensure we have standard arrow cursor or custom setup
        if hasattr(self.main_window, 'video_view') and self.main_window.video_view:
            self.main_window.video_view.set_box_selection_mode(False)
            self.main_window.video_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.main_window.video_view.viewport().unsetCursor()
            
        self.update_overlay()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._clear_overlay_items()

    def _clear_overlay_items(self):
        for item in self.overlay_items:
            try:
                self.main_window.video_view.scene.removeItem(item)
            except Exception:
                pass
        self.overlay_items.clear()
        
        if self.mesh_pixmap_item:
            try:
                self.main_window.video_view.scene.removeItem(self.mesh_pixmap_item)
            except Exception:
                pass
            self.mesh_pixmap_item = None

    def on_frame_changed(self, frame_idx, current_time_sec):
        self.update_overlay()

    def on_video_loaded(self, metadata):
        self.video_path = getattr(self.main_window, 'current_video_path', '')
        self.cached_centers.clear()
        self.cached_rotations.clear()
        self.last_coords_id = None
        self._clear_overlay_items()

    def _get_process_tool(self):
        for tool in self.main_window.tools:
            if tool.get_name() == "Process Video (Point Tracking)":
                return tool
        return None

    def _sync_tracking_cache(self, process_tool):
        # We check if coords_3d has been modified or loaded
        coords_3d = process_tool.coords_3d
        if coords_3d is None:
            self.cached_centers.clear()
            self.cached_rotations.clear()
            self.last_coords_id = None
            return False
            
        coords_id = id(coords_3d)
        if coords_id == self.last_coords_id and len(self.cached_centers) > 0:
            return True
            
        self.lbl_status.setText("Status: Syncing and precomputing camera and tracking frames...")
        self.lbl_status.setStyleSheet("color: #0078D7; font-weight: bold;")
        
        self.cached_centers.clear()
        self.cached_rotations.clear()
        
        tracks_2d = process_tool.tracks_2d
        extrinsics = process_tool.robust_squeeze_tracks(process_tool.tracking_results.get("extrinsics"))
        intrinsics = process_tool.robust_squeeze_tracks(process_tool.tracking_results.get("intrinsics"))
        
        if tracks_2d is None or extrinsics is None or intrinsics is None:
            self.lbl_status.setText("Status: Missing tracks, extrinsics, or intrinsics in Process tool.")
            self.lbl_status.setStyleSheet("color: #F44336;")
            return False
            
        T, N, _ = coords_3d.shape
        if N < 3:
            self.lbl_status.setText("Status: Need at least 3 tracked points to orient 3D object.")
            self.lbl_status.setStyleSheet("color: #F44336;")
            return False
            
        start_frame = getattr(process_tool, "start_frame", 0)
        step = getattr(process_tool, "step", 1)
        
        U_prev = None
        X_prev = None
        
        try:
            for t in range(T):
                w2c = extrinsics[t]
                K = intrinsics[t]
                pts_3d_t = coords_3d[t]
                pts_2d_t = tracks_2d[t]
                
                # 1. 2D Guided Ray Correction
                pts_3d_hom = np.hstack([pts_3d_t, np.ones((N, 1))])
                pts_cam = (w2c @ pts_3d_hom.T).T
                depth = pts_cam[:, 2]
                
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                
                ray_dir_cam = np.zeros((N, 3))
                ray_dir_cam[:, 0] = (pts_2d_t[:, 0] - cx) / fx
                ray_dir_cam[:, 1] = (pts_2d_t[:, 1] - cy) / fy
                ray_dir_cam[:, 2] = 1.0
                
                pts_cam_corr = ray_dir_cam * depth[:, None]
                
                c2w = np.linalg.inv(w2c)
                pts_cam_corr_hom = np.hstack([pts_cam_corr, np.ones((N, 1))])
                pts_world_corr = (c2w @ pts_cam_corr_hom.T).T[:, :3]
                
                # 2. PCA calculation for orientation axes
                center_t = np.mean(pts_world_corr, axis=0)
                pts_centered = pts_world_corr - center_t
                
                cov = np.cov(pts_centered.T)
                evals, evecs = np.linalg.eigh(cov)
                
                U_t = evecs[:, 0] # Minor variance = normal vector / up vector
                X_t = evecs[:, 2] # Major variance = principal direction
                
                # 3. Handedness and Sign Correction
                if t == 0:
                    P_cam_0 = c2w[:3, 3]
                    V_cam_0 = P_cam_0 - center_t
                    if np.dot(U_t, V_cam_0) < 0:
                        U_t = -U_t
                    Y_t = np.cross(U_t, X_t)
                    
                    U_t /= np.linalg.norm(U_t)
                    X_t /= np.linalg.norm(X_t)
                    Y_t /= np.linalg.norm(Y_t)
                    
                    R_t = np.stack([X_t, Y_t, U_t], axis=1)
                    U_prev = U_t
                    X_prev = X_t
                else:
                    if np.dot(U_t, U_prev) < 0:
                        U_t = -U_t
                    if np.dot(X_t, X_prev) < 0:
                        X_t = -X_t
                    Y_t = np.cross(U_t, X_t)
                    
                    U_t /= np.linalg.norm(U_t)
                    X_t /= np.linalg.norm(X_t)
                    Y_t /= np.linalg.norm(Y_t)
                    
                    R_t = np.stack([X_t, Y_t, U_t], axis=1)
                    U_prev = U_t
                    X_prev = X_t
                    
                frame_idx = start_frame + t * step
                self.cached_centers[frame_idx] = center_t
                self.cached_rotations[frame_idx] = R_t
                
            self.last_coords_id = coords_id
            self.lbl_status.setText("Status: Tracking points and rotation matrices synced successfully.")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            return True
        except Exception as e:
            self.lbl_status.setText(f"Status: PCA precomputation failed: {e}")
            self.lbl_status.setStyleSheet("color: #F44336;")
            return False

    def _get_interpolated_state(self, frame_idx, start_frame, step):
        if not self.cached_centers or not self.cached_rotations:
            return None, None
            
        if frame_idx in self.cached_centers:
            return self.cached_centers[frame_idx], self.cached_rotations[frame_idx]
            
        # Linear interpolation
        float_idx = (frame_idx - start_frame) / step
        T = len(self.cached_centers)
        
        if float_idx <= 0:
            first_key = start_frame
            return self.cached_centers[first_key], self.cached_rotations[first_key]
        if float_idx >= T - 1:
            last_key = start_frame + (T - 1) * step
            return self.cached_centers[last_key], self.cached_rotations[last_key]
            
        k = int(np.floor(float_idx))
        ratio = float_idx - k
        
        key_prev = start_frame + k * step
        key_next = start_frame + (k + 1) * step
        
        center_prev = self.cached_centers.get(key_prev)
        center_next = self.cached_centers.get(key_next)
        if center_prev is None or center_next is None:
            return None, None
            
        center_interp = (1.0 - ratio) * center_prev + ratio * center_next
        
        R_prev = self.cached_rotations[key_prev]
        R_next = self.cached_rotations[key_next]
        
        # Linear interpolation and orthonormalization
        R_interp = (1.0 - ratio) * R_prev + ratio * R_next
        U, _, Vt = np.linalg.svd(R_interp)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:
            U[:, -1] *= -1
            R_ortho = U @ Vt
            
        return center_interp, R_ortho

    def get_euler_rotation(self, yaw_deg, pitch_deg, roll_deg):
        yaw = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)
        roll = np.radians(roll_deg)
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        return Rz @ Ry @ Rx

    def project_points(self, pts_world, w2c, K):
        M = pts_world.shape[0]
        pts_world_hom = np.hstack([pts_world, np.ones((M, 1))])
        pts_cam = (w2c @ pts_world_hom.T).T
        
        xc = pts_cam[:, 0]
        yc = pts_cam[:, 1]
        zc = pts_cam[:, 2]
        
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        zc_safe = np.where(zc == 0, 1e-5, zc)
        
        u = (fx * xc) / zc_safe + cx
        v = (fy * yc) / zc_safe + cy
        
        return np.stack([u, v], axis=1), zc

    def update_overlay(self):
        self._clear_overlay_items()
        
        if not self.video_path:
            return
            
        process_tool = self._get_process_tool()
        if not process_tool:
            self.lbl_status.setText("Status: Error - Process Video tool not found.")
            return
            
        # Try to sync cache
        synced = self._sync_tracking_cache(process_tool)
        if not synced:
            return
            
        current_frame = self.main_window.timeline_slider.value()
        start_frame = process_tool.start_frame
        step = process_tool.step
        
        # 1. Get interpolated camera parameters
        extrinsics = process_tool.robust_squeeze_tracks(process_tool.tracking_results.get("extrinsics"))
        intrinsics = process_tool.robust_squeeze_tracks(process_tool.tracking_results.get("intrinsics"))
        
        w2c = self._get_interpolated_data(extrinsics, current_frame, start_frame, step)
        K = self._get_interpolated_data(intrinsics, current_frame, start_frame, step)
        
        if w2c is None or K is None:
            return
            
        # 2. Get interpolated center and rotation from the point cloud
        center_t, R_t = self._get_interpolated_state(current_frame, start_frame, step)
        if center_t is None or R_t is None:
            return
            
        # 3. Draw the 3D Object Overlay
        mesh = self.mesh_widget.get_mesh()
        face_colors = self.mesh_widget.get_face_colors()
        if mesh is not None:
            scale = self.mesh_widget.get_scale()
            offset = self.mesh_widget.get_offset()
            
            # Rotation offset matrix
            R_offset = self.mesh_widget.get_euler_rotation_matrix()
            
            # Combine rotations: R_total = R_t @ R_offset
            R_total = R_t @ R_offset
            
            # V_world = (V_centered * scale + offset) @ R_total.T + center_t
            V_world = (mesh.vertices * scale + offset) @ R_total.T + center_t
            
            # Project vertices to 2D
            pts_img, zc = self.project_points(V_world, w2c, K)
            
            # Video size for transparency pixmap
            W = 1920
            H = 1080
            if hasattr(self.main_window.decoder_thread, 'container') and self.main_window.decoder_thread.container:
                try:
                    video_stream = self.main_window.decoder_thread.container.streams.video[0]
                    W, W_h = video_stream.width, video_stream.height
                    W, H = W, W_h
                except Exception:
                    pass
            
            # Create overlay image
            overlay_img = QImage(W, H, QImage.Format.Format_ARGB32)
            overlay_img.fill(Qt.GlobalColor.transparent)
            
            painter = QPainter(overlay_img)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Painter's Algorithm: Sort faces by average depth descending (furthest first)
            face_depths = zc[mesh.faces].mean(axis=1)
            sorted_face_indices = np.argsort(face_depths)[::-1]
            
            for face_idx in sorted_face_indices:
                face = mesh.faces[face_idx]
                
                # Get projected 2D coordinates for the face's vertices
                p0 = pts_img[face[0]]
                p1 = pts_img[face[1]]
                p2 = pts_img[face[2]]
                
                # Check for nan/inf
                if not (np.isfinite(p0).all() and np.isfinite(p1).all() and np.isfinite(p2).all()):
                    continue
                    
                # Visual backface culling in image space
                cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
                if cross < 0:
                    # Skip back-facing triangles to keep clean rendering
                    continue
                    
                poly = QPolygonF([QPointF(p0[0], p0[1]), QPointF(p1[0], p1[1]), QPointF(p2[0], p2[1])])
                
                # Fill color
                face_color = face_colors[face_idx]
                painter.setBrush(QBrush(face_color))
                
                # Thin transparent border to prevent small cracks between polygons
                border_color = QColor(face_color.red(), face_color.green(), face_color.blue(), 100)
                painter.setPen(QPen(border_color, 0.5))
                
                painter.drawPolygon(poly)
                
            painter.end()
            
            # Add transparent overlay to scene
            self.mesh_pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(overlay_img))
            self.main_window.video_view.scene.addItem(self.mesh_pixmap_item)
            
        # 4. Draw the Tracked Points and Trails ON TOP of the 3D Object
        if self.mesh_widget.get_show_points() and process_tool.tracks_2d is not None:
            interpolated = process_tool.get_interpolated_tracks(current_frame)
            if interpolated is not None:
                num_points = interpolated.shape[0]
                trail_len = 10
                
                for pt_idx in range(num_points):
                    # Gather trail points
                    past_pts = []
                    for offset_f in range(trail_len + 1):
                        f_past = current_frame - offset_f
                        if f_past < start_frame:
                            break
                        coords_past = process_tool.get_interpolated_tracks(f_past)
                        if coords_past is not None:
                            past_pts.append(coords_past[pt_idx])
                            
                    # Draw trails
                    for idx, pt in enumerate(past_pts):
                        if idx == 0:
                            continue
                        pt_prev = past_pts[idx - 1]
                        opacity = max(0.15, 1.0 - (idx / trail_len))
                        base_color = process_tool.get_point_color(pt_idx)
                        trail_color = QColor(base_color.red(), base_color.green(), base_color.blue(), int(opacity * 255))
                        
                        line = QGraphicsLineItem(pt_prev[0], pt_prev[1], pt[0], pt[1])
                        line.setPen(QPen(trail_color, 2))
                        self.main_window.video_view.scene.addItem(line)
                        self.overlay_items.append(line)
                        
                    # Draw active dot
                    x_curr, y_curr = interpolated[pt_idx]
                    marker = QGraphicsEllipseItem(x_curr - 5, y_curr - 5, 10, 10)
                    marker.setBrush(QBrush(process_tool.get_point_color(pt_idx)))
                    marker.setPen(QPen(QColor(0, 0, 0), 1))
                    
                    self.main_window.video_view.scene.addItem(marker)
                    self.overlay_items.append(marker)

    def _get_interpolated_data(self, arr, frame_idx, start_frame, step):
        if arr is None:
            return None
        T = arr.shape[0]
        if frame_idx <= start_frame:
            return arr[0]
        if step == 1:
            idx = frame_idx - start_frame
            if idx >= T:
                return arr[-1]
            return arr[idx]
            
        float_idx = (frame_idx - start_frame) / step
        if float_idx <= 0:
            return arr[0]
        if float_idx >= T - 1:
            return arr[-1]
            
        k = int(np.floor(float_idx))
        ratio = float_idx - k
        return (1.0 - ratio) * arr[k] + ratio * arr[k + 1]
