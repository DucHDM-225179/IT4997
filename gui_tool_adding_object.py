# gui_tool_adding_object.py
import os
import json
import numpy as np
import trimesh
import http.server
import socketserver
import threading
import webbrowser
import colorsys

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSlider, 
                             QPushButton, QLabel, QFileDialog, QLineEdit, 
                             QMessageBox, QGroupBox, QGridLayout, QGraphicsView,
                             QCheckBox, QApplication)
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QColor, QBrush, QPen, QImage, QPixmap, QPainter, QPolygonF
from PyQt6.QtWidgets import (QGraphicsEllipseItem, QGraphicsLineItem, 
                             QGraphicsPixmapItem, QGraphicsSimpleTextItem)

from gui_tool_base import BaseTool
from gui_tool_mesh_transform import MeshTransformWidget

class HTTPServerThread(threading.Thread):
    def __init__(self, directory, port=8000):
        super().__init__()
        self.directory = directory
        self.port = port
        self.daemon = True
        self.server = None

    def run(self):
        directory_to_serve = self.directory
        class CustomHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory_to_serve, **kwargs)
                
        # Try to bind to port, if busy try next ports
        for p in range(self.port, self.port + 100):
            try:
                self.server = socketserver.TCPServer(("127.0.0.1", p), CustomHandler)
                self.port = p
                print(f"Started visualizer local server at http://127.0.0.1:{p}")
                break
            except OSError:
                continue
        if self.server:
            self.server.serve_forever()

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()

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
        
        self.server_thread = None
        
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
        
        # Checkbox to toggle 2D overlay in preview
        self.chk_show_overlay = QCheckBox("Show 2D Overlay in Preview")
        self.chk_show_overlay.setChecked(True)
        self.chk_show_overlay.stateChanged.connect(self.update_overlay)
        layout.addWidget(self.chk_show_overlay)
        
        # Button to open 3D visualizer
        self.btn_open_3d_viz = QPushButton("Open 3D WebGL Visualizer")
        self.btn_open_3d_viz.setStyleSheet("background-color: #9B59B6; color: white; font-weight: bold; padding: 8px; margin-top: 5px;")
        self.btn_open_3d_viz.clicked.connect(self._on_open_3d_viz)
        layout.addWidget(self.btn_open_3d_viz)
        
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
        
        try:
            pts_ref = None
            R_0 = None
            
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
                
                # 2. Centered coordinates
                center_t = np.mean(pts_world_corr, axis=0)
                pts_centered = pts_world_corr - center_t
                
                # 3. Handedness and Sign Correction using Kabsch relative rotation
                if t == 0:
                    cov = np.cov(pts_centered.T)
                    evals, evecs = np.linalg.eigh(cov)
                    
                    U_0 = evecs[:, 0]  # Minor variance = normal vector / up vector
                    X_0 = evecs[:, 2]  # Major variance = principal direction
                    
                    P_cam_0 = c2w[:3, 3]
                    V_cam_0 = P_cam_0 - center_t
                    if np.dot(U_0, V_cam_0) < 0:
                        U_0 = -U_0
                    
                    # Ensure orthonormal basis
                    U_0 /= np.linalg.norm(U_0)
                    X_0 = X_0 - np.dot(X_0, U_0) * U_0
                    X_0 /= np.linalg.norm(X_0)
                    Y_0 = np.cross(U_0, X_0)
                    Y_0 /= np.linalg.norm(Y_0)
                    
                    R_0 = np.stack([X_0, Y_0, U_0], axis=1)
                    
                    # Store reference points for relative alignment in subsequent frames
                    pts_ref = pts_centered.copy()
                    R_t = R_0
                else:
                    # Kabsch algorithm to find optimal relative rotation R_rel mapping pts_ref -> pts_centered
                    H = pts_ref.T @ pts_centered
                    U_svd, S_svd, Vt_svd = np.linalg.svd(H)
                    V_svd = Vt_svd.T
                    
                    # Handle reflection / handedness
                    det = np.linalg.det(V_svd @ U_svd.T)
                    if det < 0:
                        V_svd[:, 2] = -V_svd[:, 2]
                        
                    R_rel = V_svd @ U_svd.T
                    R_t = R_rel.T @ R_0
                    
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
            
        if not self.chk_show_overlay.isChecked():
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

    def _on_open_3d_viz(self):
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "No video loaded!")
            return
            
        process_tool = self._get_process_tool()
        if not process_tool or process_tool.coords_3d is None:
            QMessageBox.warning(self, "Warning", "Please run point tracking first and ensure tracking results are loaded.")
            return
            
        preprocess_npz = getattr(process_tool, "preprocess_npz", "")
        if not preprocess_npz or not os.path.exists(preprocess_npz):
            QMessageBox.warning(self, "Warning", "Preprocessed depth maps not found. Please ensure preprocessing is done.")
            return

        self.lbl_status.setText("Status: Generating 3D point cloud & trajectories...")
        self.lbl_status.setStyleSheet("color: #0078D7; font-weight: bold;")
        QApplication.processEvents()
        
        # Start server if not running
        if self.server_thread is None:
            viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_viz")
            if not os.path.exists(viz_dir):
                viz_dir = os.path.abspath("_viz")
            self.server_thread = HTTPServerThread(directory=viz_dir, port=8000)
            self.server_thread.start()
            import time
            time.sleep(0.2)
            
        try:
            filename = self._generate_viz_data(process_tool, preprocess_npz)
            if filename:
                port = self.server_thread.port
                url = f"http://127.0.0.1:{port}/viz_template.html?data={filename}"
                webbrowser.open(url)
                self.lbl_status.setText("Status: 3D Visualizer opened in browser successfully.")
                self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            else:
                self.lbl_status.setText("Status: Failed to generate visualization data.")
                self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open visualizer: {e}")
            self.lbl_status.setText("Status: Visualizer failed.")
            self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")

    def _generate_viz_data(self, process_tool, preprocess_npz):
        import zlib
        import struct
        import glob
        import time
        
        viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_viz")
        if not os.path.exists(viz_dir):
            viz_dir = os.path.abspath("_viz")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Clean up old data_*.bin files
        for old_file in glob.glob(os.path.join(viz_dir, "data_*.bin")):
            try:
                os.remove(old_file)
            except Exception:
                pass
                
        # Generate new filename
        ts = int(time.time() * 1000)
        filename = f"data_{ts}.bin"
        bin_path = os.path.join(viz_dir, filename)
        
        # 1. Retrieve data
        from gui_tool_process import get_video_dimensions
        W_orig, H_orig = get_video_dimensions(self.video_path)
        
        # Calculate aspect-ratio preserving target size (max side 256)
        aspect_ratio = W_orig / H_orig
        if W_orig >= H_orig:
            width_target = 256
            height_target = int(round(256 / aspect_ratio))
        else:
            height_target = 256
            width_target = int(round(256 * aspect_ratio))
            
        width_target = max(4, (width_target // 4) * 4)
        height_target = max(4, (height_target // 4) * 4)
        fixed_size = (width_target, height_target)
        
        extrinsics = process_tool.robust_squeeze_tracks(process_tool.tracking_results.get("extrinsics"))
        intrinsics = process_tool.robust_squeeze_tracks(process_tool.tracking_results.get("intrinsics"))
        trajs = process_tool.coords_3d
        visibs = process_tool.visibs
        confs = process_tool.tracking_results.get("unc_metric")
        
        T = trajs.shape[0]
        start_frame = process_tool.start_frame
        step = process_tool.step
        
        # 2. Decode Video Frames using PyAV with pts_map synchronization
        import av
        pts_map = getattr(self.main_window.decoder_thread, 'pts_map', [])
        
        frame_indices = [start_frame + t * step for t in range(T)]
        rgb_video = []
        
        try:
            container = av.open(self.video_path, container_options={'ignore_editlist': '1'})
            video_stream = container.streams.video[0]
            video_stream.thread_type = "AUTO"
            
            # Fast seek if pts_map is available
            if pts_map and start_frame < len(pts_map):
                container.seek(pts_map[start_frame], stream=video_stream, backward=True)
                
            count = 0
            for frame in container.decode(video=0):
                if frame.pts is None:
                    continue
                try:
                    idx = pts_map.index(frame.pts) if pts_map else count
                except ValueError:
                    continue
                    
                if idx < start_frame:
                    continue
                if idx > frame_indices[-1]:
                    break
                    
                if (idx - start_frame) % step == 0:
                    img = frame.to_rgb().to_ndarray()
                    rgb_video.append(img)
                    count += 1
                    if len(rgb_video) >= T:
                        break
            container.close()
        except Exception as e:
            print(f"PyAV synchronized decode failed: {e}. Falling back to basic decord/cv2.")
            rgb_video = []
            try:
                import decord
                vr = decord.VideoReader(self.video_path)
                frames = vr.get_batch(frame_indices).asnumpy()
                for t in range(T):
                    rgb_video.append(frames[t])
            except Exception as e2:
                print(f"Decord failed, trying OpenCV fallback: {e2}")
                import cv2
                cap = cv2.VideoCapture(self.video_path)
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret:
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
                        frame = np.zeros((h, w, 3), dtype=np.uint8)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_video.append(frame)
                cap.release()

        while len(rgb_video) < T:
            if len(rgb_video) > 0:
                rgb_video.append(rgb_video[-1])
            else:
                rgb_video.append(np.zeros((H_orig, W_orig, 3), dtype=np.uint8))
        rgb_video = rgb_video[:T]
            
        import cv2
        rgb_video_resized = np.stack([cv2.resize(f, fixed_size, interpolation=cv2.INTER_AREA) for f in rgb_video])
        
        # 3. Load and Resize Depth Maps (Refined or Preprocessed)
        has_refined_depth = (process_tool.tracking_results is not None and "depths" in process_tool.tracking_results)
        
        if has_refined_depth:
            all_depths = process_tool.tracking_results["depths"]
            uncs = process_tool.tracking_results.get("unc_metric", None)
            prep_data = {} # not used, since we have refined data
        else:
            prep_data = np.load(preprocess_npz, allow_pickle=True)
            all_depths = prep_data["depths"]
            uncs = prep_data.get("unc_metric", None)
            
        depth_video = []
        for t in range(T):
            if t < len(all_depths):
                depth_video.append(all_depths[t])
            else:
                depth_video.append(all_depths[-1] if len(all_depths) > 0 else np.zeros((fixed_size[1], fixed_size[0]), dtype=np.float32))
                
        depth_video_resized = np.stack([cv2.resize(d, fixed_size, interpolation=cv2.INTER_NEAREST) for d in depth_video])
        
        # Apply uncertainty mask if present
        if uncs is not None:
            for t in range(min(T, len(uncs))):
                unc_frame = uncs[t]
                if unc_frame.dtype == bool:
                    mask = unc_frame
                else:
                    mask = unc_frame < 0.5
                # Resize mask to fixed size
                mask_resized = cv2.resize(mask.astype(np.uint8), fixed_size, interpolation=cv2.INTER_NEAREST) > 0
                depth_video_resized[t][mask_resized] = 0.0
                
        # 4. Normalize trajectory and camera relative to first frame (tapip3d style)
        first_frame_inv = np.linalg.inv(extrinsics[0])
        normalized_extrinsics = np.array([first_frame_inv @ ext for ext in extrinsics])
        
        N = trajs.shape[1]
        normalized_trajs = np.zeros_like(trajs)
        for t in range(T):
            pts_hom = np.hstack([trajs[t], np.ones((N, 1))])
            pts_normalized = (first_frame_inv @ pts_hom.T).T
            normalized_trajs[t] = pts_normalized[:, :3]
            
        # Scale intrinsics to match resized resolution
        scale_x = fixed_size[0] / W_orig
        scale_y = fixed_size[1] / H_orig
        
        scaled_intrinsics = intrinsics.copy()
        scaled_intrinsics[:, 0, :] *= scale_x
        scaled_intrinsics[:, 1, :] *= scale_y
        
        # Calculate FOV info
        fx = intrinsics[0, 0, 0]
        fy = intrinsics[0, 1, 1]
        fov_y = 2 * np.arctan(H_orig / (2 * fy)) * (180 / np.pi)
        fov_x = 2 * np.arctan(W_orig / (2 * fx)) * (180 / np.pi)
        original_aspect_ratio = (W_orig / fx) / (H_orig / fy)
        
        # 5. Encode 16-bit depth maps
        min_depth = float(depth_video_resized.min()) * 0.8
        max_depth = float(depth_video_resized.max()) * 1.5
        if max_depth == min_depth:
            max_depth += 1e-5
            
        depth_normalized = (depth_video_resized - min_depth) / (max_depth - min_depth)
        depth_normalized = np.clip(depth_normalized, 0.0, 1.0)
        depth_int = (depth_normalized * ((1 << 16) - 1)).astype(np.uint16)
        
        depths_rgb = np.zeros((T, fixed_size[1], fixed_size[0], 3), dtype=np.uint8)
        depths_rgb[:, :, :, 0] = (depth_int & 0xFF).astype(np.uint8)
        depths_rgb[:, :, :, 1] = ((depth_int >> 8) & 0xFF).astype(np.uint8)
        
        # 6. Precompute mesh vertices, faces, and dynamic transform matrices
        mesh = self.mesh_widget.get_mesh()
        mesh_vertices = np.zeros((0, 3), dtype=np.float32)
        mesh_faces = np.zeros((0, 3), dtype=np.int32)
        mesh_vertex_colors = np.zeros((0, 3), dtype=np.float32)
        mesh_centers = np.zeros((T, 3), dtype=np.float32)
        mesh_rotations = np.zeros((T, 3, 3), dtype=np.float32)
        
        if mesh is not None:
            mesh_vertices = mesh.vertices.astype(np.float32)
            mesh_faces = mesh.faces.astype(np.int32)
            
            # Compute vertex colors based on centroid distance
            vertex_dists = np.linalg.norm(mesh_vertices, axis=1)
            min_d = vertex_dists.min() if len(vertex_dists) > 0 else 0.0
            max_d = vertex_dists.max() if len(vertex_dists) > 0 else 1.0
            if max_d == min_d:
                max_d += 1e-5
                
            mesh_vertex_colors_list = []
            for d in vertex_dists:
                t_color = (d - min_d) / (max_d - min_d)
                t_color = np.clip(t_color, 0.0, 1.0)
                hue = (240 - 240 * t_color) / 360.0
                r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.8)
                mesh_vertex_colors_list.append([r, g, b])
            mesh_vertex_colors = np.array(mesh_vertex_colors_list, dtype=np.float32)
            
            # Precompute Kabsch center and rotation per frame using normalized_trajs
            pts_ref = None
            R_0 = None
            
            for t in range(T):
                pts_3d_t = normalized_trajs[t]
                center_t = np.mean(pts_3d_t, axis=0)
                pts_centered = pts_3d_t - center_t
                
                if t == 0:
                    cov = np.cov(pts_centered.T)
                    evals, evecs = np.linalg.eigh(cov)
                    
                    U_0 = evecs[:, 0]
                    X_0 = evecs[:, 2]
                    
                    c2w_norm_0 = np.linalg.inv(normalized_extrinsics[0])
                    P_cam_0 = c2w_norm_0[:3, 3]
                    V_cam_0 = P_cam_0 - center_t
                    if np.dot(U_0, V_cam_0) < 0:
                        U_0 = -U_0
                        
                    U_0 /= np.linalg.norm(U_0)
                    X_0 = X_0 - np.dot(X_0, U_0) * U_0
                    X_0 /= np.linalg.norm(X_0)
                    Y_0 = np.cross(U_0, X_0)
                    Y_0 /= np.linalg.norm(Y_0)
                    
                    R_0 = np.stack([X_0, Y_0, U_0], axis=1)
                    pts_ref = pts_centered.copy()
                    R_t = R_0
                else:
                    H = pts_ref.T @ pts_centered
                    U_svd, S_svd, Vt_svd = np.linalg.svd(H)
                    V_svd = Vt_svd.T
                    
                    det = np.linalg.det(V_svd @ U_svd.T)
                    if det < 0:
                        V_svd[:, 2] = -V_svd[:, 2]
                        
                    R_rel = V_svd @ U_svd.T
                    R_t = R_rel.T @ R_0
                    
                mesh_centers[t] = center_t
                mesh_rotations[t] = R_t
                
        # 7. Package and Compress
        arrays = {
            "rgb_video": rgb_video_resized,
            "depths_rgb": depths_rgb,
            "intrinsics": scaled_intrinsics,
            "extrinsics": normalized_extrinsics,
            "inv_extrinsics": np.linalg.inv(normalized_extrinsics),
            "trajectories": normalized_trajs.astype(np.float32),
            "cameraZ": np.array(0.0, dtype=np.float32),
            "visibs": visibs if visibs is not None else None,
            "confs": confs if confs is not None else None,
            "mesh_vertices": mesh_vertices,
            "mesh_faces": mesh_faces,
            "mesh_vertex_colors": mesh_vertex_colors,
            "mesh_centers": mesh_centers,
            "mesh_rotations": mesh_rotations
        }
        
        header = {}
        blob_parts = []
        offset = 0
        for key, arr in arrays.items():
            if arr is not None:
                arr = np.ascontiguousarray(arr)
                arr_bytes = arr.tobytes()
                header[key] = {
                    "dtype": str(arr.dtype),
                    "shape": arr.shape,
                    "offset": offset,
                    "length": len(arr_bytes)
                }
                blob_parts.append(arr_bytes)
                offset += len(arr_bytes)
                
        raw_blob = b"".join(blob_parts)
        compressed_blob = zlib.compress(raw_blob, level=9)
        
        header["meta"] = {
            "depthRange": [min_depth, max_depth],
            "totalFrames": int(T),
            "resolution": fixed_size,
            "baseFrameRate": 4,
            "numTrajectoryPoints": normalized_trajs.shape[1],
            "fov": float(fov_y),
            "fov_x": float(fov_x),
            "original_aspect_ratio": float(original_aspect_ratio),
            "fixed_aspect_ratio": float(fixed_size[0]/fixed_size[1]),
            "mesh_scale": self.mesh_widget.get_scale() if mesh is not None else 1.0,
            "mesh_offset": self.mesh_widget.get_offset().tolist() if mesh is not None else [0.0, 0.0, 0.0],
            "mesh_rotation": [self.mesh_widget.sld_yaw.value(), self.mesh_widget.sld_pitch.value(), self.mesh_widget.sld_roll.value()] if mesh is not None else [0.0, 0.0, 0.0]
        }
        
        header_bytes = json.dumps(header).encode("utf-8")
        header_len = struct.pack("<I", len(header_bytes))
        
        with open(bin_path, "wb") as f:
            f.write(header_len)
            f.write(header_bytes)
            f.write(compressed_blob)
            
        return filename
