# gui_tool_adding_object_2d.py
import os
import json
import numpy as np
import trimesh
import cv2

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

class AddingObject2DTool(BaseTool):
    """Tool to load a 3D OBJ, estimate camera/plane pose from 2D tracks, and render it."""
    
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        
        self.video_path = ""
        self.overlay_items = []
        self.mesh_pixmap_item = None
        
        # Caches for plane rotations and translations
        self.cached_R = {}
        self.cached_T = {}
        self.last_coords_id = None
        
        # Scaling variables anchor to point cloud spread
        self.spread = 1.0
        self.template_pts = None
        self.template_center = None
        self.K = None
        
        self._init_ui()
        
        # Load default teapot if present
        self.mesh_widget.load_default_mesh_if_exist()
        self._sync_status_label()

    def get_name(self):
        return "Add 3D Object (Planar 2D)"

    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Unified Mesh transform widget
        self.mesh_widget = MeshTransformWidget(self, group_title="Transform Adjustments (Planar)")
        self.mesh_widget.set_snap_button_text("Snap Center to 2D Tracks Center")
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
        self.cached_R.clear()
        self.cached_T.clear()
        self.last_coords_id = None
        self._clear_overlay_items()

    def _get_process_tool(self):
        for tool in self.main_window.tools:
            if tool.get_name() == "Process Video (Point Tracking)":
                return tool
        return None

    def _sync_tracking_cache(self, process_tool):
        tracks_2d = process_tool.tracks_2d
        if tracks_2d is None:
            self.cached_R.clear()
            self.cached_T.clear()
            self.last_coords_id = None
            return False
            
        coords_id = id(tracks_2d)
        if coords_id == self.last_coords_id and len(self.cached_R) > 0:
            return True
            
        self.lbl_status.setText("Status: Syncing and optimizing planar PnP camera matrices...")
        self.lbl_status.setStyleSheet("color: #0078D7; font-weight: bold;")
        
        self.cached_R.clear()
        self.cached_T.clear()
        
        T, N, _ = tracks_2d.shape
        if N < 3:
            self.lbl_status.setText("Status: Need at least 3 tracked points to orient 2D plane object.")
            self.lbl_status.setStyleSheet("color: #F44336;")
            return False
            
        start_frame = getattr(process_tool, "start_frame", 0)
        step = getattr(process_tool, "step", 1)
        
        try:
            # 1. Setup Camera Matrix using dimensions
            W = 1920
            H = 1080
            if hasattr(self.main_window.decoder_thread, 'container') and self.main_window.decoder_thread.container:
                try:
                    video_stream = self.main_window.decoder_thread.container.streams.video[0]
                    W, H = video_stream.width, video_stream.height
                except Exception:
                    pass
            
            f = float(max(W, H))
            cx = float(W) / 2.0
            cy = float(H) / 2.0
            self.K = np.array([
                [f, 0.0, cx],
                [0.0, f, cy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            
            # 2. Get points at start frame to establish local plane template
            p_0 = tracks_2d[0]
            vis_0 = process_tool.visibs[0].flatten() if getattr(process_tool, "visibs", None) is not None else np.ones(N)
            valid_0 = (vis_0 > 0.5) & np.isfinite(p_0).all(axis=1)
            
            if np.sum(valid_0) < 3:
                self.lbl_status.setText("Status: Need at least 3 valid tracked points on first frame.")
                self.lbl_status.setStyleSheet("color: #F44336;")
                return False

            extrinsics = None
            coords_3d = None
            if hasattr(process_tool, "tracking_results") and process_tool.tracking_results is not None:
                extrinsics = process_tool.robust_squeeze_tracks(process_tool.tracking_results.get("extrinsics"))
                coords_3d = process_tool.robust_squeeze_tracks(process_tool.tracking_results.get("coords"))
                intrinsics = process_tool.robust_squeeze_tracks(process_tool.tracking_results.get("intrinsics"))
                if intrinsics is not None:
                    self.K = intrinsics[0].astype(np.float32)

            if extrinsics is not None and coords_3d is not None:
                # 3D tracking data is available: use true 3D camera-space template (handles initial tilt/skew)
                w2c_0 = extrinsics[0]
                pts_3d_0 = coords_3d[0]  # (N, 3) in world space
                
                # Transform world space coordinates to camera space at frame 0
                pts_3d_0_hom = np.hstack([pts_3d_0, np.ones((N, 1))])
                pts_cam_0 = (w2c_0 @ pts_3d_0_hom.T).T[:, :3]
                
                C_0_cam = np.mean(pts_cam_0[valid_0], axis=0)
                pts_centered = pts_cam_0 - C_0_cam
                
                # Fit 3D plane to the valid points using PCA (SVD)
                pts_centered_valid = pts_centered[valid_0]
                _, _, Vt = np.linalg.svd(pts_centered_valid)
                n = Vt[2, :]  # Normal vector is the eigenvector with the smallest eigenvalue
                if n[2] > 0:
                    n = -n  # Ensure normal points generally towards the camera
                
                # Construct alignment rotation matrix R_align to map plane normal to [0, 0, 1]
                x_axis = np.array([1.0, 0.0, 0.0])
                if np.abs(np.dot(x_axis, n)) > 0.9:
                    x_axis = np.array([0.0, 1.0, 0.0])
                x_axis = x_axis - np.dot(x_axis, n) * n
                x_axis /= np.linalg.norm(x_axis)
                y_axis = np.cross(n, x_axis)
                y_axis /= np.linalg.norm(y_axis)
                
                R_align = np.stack([x_axis, y_axis, n], axis=0)  # (3, 3)
                
                # Rotate template so it lies flat on Z = 0 in local space
                P_template = (R_align @ pts_centered.T).T.astype(np.float32)
                P_template[:, 2] = 0.0
                
                self.template_pts = P_template
                self.template_center = C_0_cam
                
                # Spread in 3D camera units
                self.spread = np.mean(np.std(P_template[valid_0], axis=0))
                if self.spread <= 0:
                    self.spread = 1.0
                
                # Initialize camera pose relative to template: R = R_align^T, T = C_0_cam
                R_init = R_align.T
                rvec, _ = cv2.Rodrigues(R_init)
                tvec = C_0_cam.reshape(3, 1).astype(np.float32)
            else:
                # Fallback to 2D pixel template (assumes front-parallel at start)
                C_0 = np.mean(p_0[valid_0], axis=0)
                
                P_template = np.zeros((N, 3), dtype=np.float32)
                P_template[:, 0] = p_0[:, 0] - C_0[0]
                P_template[:, 1] = p_0[:, 1] - C_0[1]
                P_template[:, 2] = 0.0
                
                self.template_pts = P_template
                self.template_center = C_0
                
                # Spread in 2D pixels
                self.spread = np.mean(np.std(p_0[valid_0], axis=0))
                if self.spread <= 0:
                    self.spread = 1.0
                    
                rvec = np.zeros((3, 1), dtype=np.float32)
                tvec = np.array([C_0[0] - cx, C_0[1] - cy, f], dtype=np.float32).reshape(3, 1)
            
            # Store initial frame pose
            R, _ = cv2.Rodrigues(rvec)
            self.cached_R[start_frame] = R
            self.cached_T[start_frame] = tvec
            
            # 3. Propagate and optimize poses forward
            for idx in range(1, T):
                frame_idx = start_frame + idx * step
                p_t = tracks_2d[idx]
                vis_t = process_tool.visibs[idx].flatten() if getattr(process_tool, "visibs", None) is not None else np.ones(N)
                valid_t = (vis_t > 0.5) & np.isfinite(p_t).all(axis=1)
                
                rvec_guess = rvec.copy()
                tvec_guess = tvec.copy()
                
                success = False
                if np.sum(valid_t) >= 3:
                    pts_template_valid = P_template[valid_t]
                    pts_tracked_valid = p_t[valid_t]
                    
                    if extrinsics is not None and coords_3d is not None:
                        # 1. 3D tracking data is available: use it to get a robust initial guess
                        w2c_t = extrinsics[idx]
                        pts_3d_t = coords_3d[idx]
                        pts_3d_t_hom = np.hstack([pts_3d_t, np.ones((N, 1))])
                        pts_cam_t = (w2c_t @ pts_3d_t_hom.T).T[:, :3]
                        
                        C_t_cam = np.mean(pts_cam_t[valid_t], axis=0)
                        
                        # Kabsch algorithm (SVD) to find 3D-3D rotation and translation
                        pts_temp = P_template[valid_t]
                        pts_c = pts_cam_t[valid_t] - C_t_cam
                        H_matrix = pts_c.T @ pts_temp  # Y X^T matrix for Kabsch
                        U_m, _, Vt_m = np.linalg.svd(H_matrix)
                        R_guess = U_m @ Vt_m
                        if np.linalg.det(R_guess) < 0:
                            Vt_m[2, :] *= -1
                            R_guess = U_m @ Vt_m
                            
                        rvec_guess, _ = cv2.Rodrigues(R_guess)
                        tvec_guess = C_t_cam.reshape(3, 1).astype(np.float32)
                        
                        # 2. Refine using high-quality 2D tracks with solvePnP (Iterative)
                        success, r_est, t_est = cv2.solvePnP(
                            pts_template_valid, 
                            pts_tracked_valid.astype(np.float32), 
                            self.K, 
                            None, 
                            rvec_guess, 
                            tvec_guess, 
                            useExtrinsicGuess=True, 
                            flags=cv2.SOLVEPNP_ITERATIVE
                        )
                    else:
                        # Fallback if no 3D data: use IPPE for >=4 points, Iterative for 3 points
                        if np.sum(valid_t) >= 4:
                            success, r_est, t_est = cv2.solvePnP(
                                pts_template_valid, 
                                pts_tracked_valid.astype(np.float32), 
                                self.K, 
                                None, 
                                flags=cv2.SOLVEPNP_IPPE
                            )
                        else:
                            success, r_est, t_est = cv2.solvePnP(
                                pts_template_valid, 
                                pts_tracked_valid.astype(np.float32), 
                                self.K, 
                                None, 
                                rvec_guess, 
                                tvec_guess, 
                                useExtrinsicGuess=True, 
                                flags=cv2.SOLVEPNP_ITERATIVE
                            )
                            
                    if success:
                        rvec = r_est
                        tvec = t_est
                
                R, _ = cv2.Rodrigues(rvec)
                self.cached_R[frame_idx] = R
                self.cached_T[frame_idx] = tvec
                
            self.last_coords_id = coords_id
            self.lbl_status.setText("Status: Planar PnP tracker poses synced successfully.")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            return True
        except Exception as e:
            self.lbl_status.setText(f"Status: PnP precomputation failed: {e}")
            self.lbl_status.setStyleSheet("color: #F44336;")
            return False

    def _get_interpolated_state(self, frame_idx, start_frame, step):
        if not self.cached_R or not self.cached_T:
            return None, None
            
        if frame_idx in self.cached_R:
            return self.cached_R[frame_idx], self.cached_T[frame_idx]
            
        # Linear interpolation
        float_idx = (frame_idx - start_frame) / step
        T = len(self.cached_R)
        
        if float_idx <= 0:
            first_key = start_frame
            return self.cached_R[first_key], self.cached_T[first_key]
        if float_idx >= T - 1:
            last_key = start_frame + (T - 1) * step
            return self.cached_R[last_key], self.cached_T[last_key]
            
        k = int(np.floor(float_idx))
        ratio = float_idx - k
        
        key_prev = start_frame + k * step
        key_next = start_frame + (k + 1) * step
        
        R_prev = self.cached_R.get(key_prev)
        R_next = self.cached_R.get(key_next)
        T_prev = self.cached_T.get(key_prev)
        T_next = self.cached_T.get(key_next)
        
        if R_prev is None or R_next is None or T_prev is None or T_next is None:
            return None, None
            
        T_interp = (1.0 - ratio) * T_prev + ratio * T_next
        
        # Linear interpolation and orthonormalization of R
        R_interp = (1.0 - ratio) * R_prev + ratio * R_next
        U, _, Vt = np.linalg.svd(R_interp)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:
            U[:, -1] *= -1
            R_ortho = U @ Vt
            
        return R_ortho, T_interp

    def project_points(self, pts_plane, R_t, T_t, K):
        # Transform plane points to camera coordinates
        pts_cam = (R_t @ pts_plane.T).T + T_t.reshape(1, 3)
        
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
        
        # 1. Get interpolated plane pose
        R_t, T_t = self._get_interpolated_state(current_frame, start_frame, step)
        if R_t is None or T_t is None:
            return
            
        # 2. Draw the 3D Object Overlay
        mesh = self.mesh_widget.get_mesh()
        face_colors = self.mesh_widget.get_face_colors()
        if mesh is not None:
            # Scale and offset anchored to spatial spread of points (pixels)
            scale = self.mesh_widget.get_scale() * self.spread
            offset = self.mesh_widget.get_offset() * self.spread
            
            # Rotation offset matrix
            R_offset = self.mesh_widget.get_euler_rotation_matrix()
            
            # Transform vertices relative to the plane coordinate system
            V_plane = (mesh.vertices * scale + offset) @ R_offset.T
            
            # Project vertices to 2D
            pts_img, zc = self.project_points(V_plane, R_t, T_t, self.K)
            
            # Video size for transparency pixmap
            W = 1920
            H = 1080
            if hasattr(self.main_window.decoder_thread, 'container') and self.main_window.decoder_thread.container:
                try:
                    video_stream = self.main_window.decoder_thread.container.streams.video[0]
                    W, H = video_stream.width, video_stream.height
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
            
        # 3. Draw the Tracked Points and Trails ON TOP of the 3D Object
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
