import os
import json
import numpy as np
from PyQt6.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QSpinBox, QWidget, QFileDialog, QGraphicsView,
                             QCheckBox, QComboBox, QMessageBox, QApplication, QProgressDialog)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import (QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsSimpleTextItem)

from gui_tool_base import BaseTool

from gui_tool_process_logic import (generate_blender_script, TrackingThread)


class ProcessVideoTool(BaseTool):
    """Tool to place points, load intermediate depth, run CUDA tracker, and visualize trajectories."""
    decode_requested = pyqtSignal(int, int, int, object)
    
    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        
        self.video_path = ""
        self.points = [] # List of (x, y) coordinates snapped to integers
        self.mode = "add" # "add" or "remove"
        
        # State
        self.preprocess_npz = ""
        self.preprocess_json = ""
        self.start_frame = 0
        self.end_frame = 0
        self.step = 1
        
        # Tracking results
        self.tracks_2d = None # (T, N, 2)
        self.coords_3d = None # (T, N, 3)
        self.visibs = None # (T, N)
        self.tracking_results = None # Full dictionary
        self.last_saved_npz = ""
        self.video_width = 0
        self.video_height = 0
        
        self.thread = None
                
        self._init_ui()
        
        # Sync initial state if already loaded
        meta = self.session.get("video_metadata")
        if meta:
            self.video_width = meta.get("width", 0)
            self.video_height = meta.get("height", 0)
            
        self.video_path = self.session.get("video_path", "")
        self._try_auto_load_preprocess()

    def get_name(self):
        return "Process Video (Point Tracking)"

    def _on_session_changed(self, key, value):
        if key == "video_metadata":
            if value:
                self.video_width = value.get("width", 0)
                self.video_height = value.get("height", 0)
        elif key == "video_path":
            self.video_path = value or ""
            
            # Reset points and tracking results for the new video
            self.points = []
            self.tracks_2d = None
            self.coords_3d = None
            self.visibs = None
            self.tracking_results = None
            self.last_saved_npz = ""
            
            self.btn_save_result.setEnabled(False)
            self.btn_gen_blender.setEnabled(False)
            self.btn_gen_blender.setStyleSheet("background-color: #E1E1E1; font-weight: bold; padding: 4px;")
            self.lbl_pts_count.setText("Selected Points: 0")
            
            self._try_auto_load_preprocess()
            self.update_overlay()
        elif key == "current_frame":
            if value is not None:
                self.update_overlay()
        elif key == "pixel_clicked":
            if self.isVisible() and value:
                x, y = value
                self._on_view_clicked(x, y)
        elif key == "box_selected":
            if self.isVisible() and value:
                x1, y1, x2, y2 = value
                self._on_box_selected(x1, y1, x2, y2)
        elif key == "preprocess_npz":
            if value:
                self._try_auto_load_preprocess()
            else:
                self.preprocess_npz = ""
                self.preprocess_json = ""
                self.lbl_prep_status.setText("Preprocess: [Not Loaded]")
                self.lbl_prep_status.setStyleSheet("color: #F44336; font-weight: bold;")

    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # 1. Preprocess Info Section
        lbl_prep_title = QLabel("<b>1. Preprocessed Data</b>")
        layout.addWidget(lbl_prep_title)
        
        self.lbl_prep_status = QLabel("Preprocess: [Not Loaded]")
        self.lbl_prep_status.setStyleSheet("color: #F44336; font-weight: bold;")
        layout.addWidget(self.lbl_prep_status)
        
        self.btn_load_prep = QPushButton("Load Preprocess [load preprocess]")
        self.btn_load_prep.setStyleSheet("background-color: #E1E1E1; font-weight: bold; padding: 5px;")
        self.btn_load_prep.clicked.connect(self._on_load_preprocess)
        layout.addWidget(self.btn_load_prep)
        
        layout.addWidget(QLabel("----------------------------------------"))
        
        # 2. Points Selector Section
        lbl_pts_title = QLabel("<b>2. Add/Remove Tracking Points</b>")
        layout.addWidget(lbl_pts_title)
        
        self.lbl_pts_count = QLabel("Selected Points: 0")
        self.lbl_pts_count.setStyleSheet("font-weight: bold; color: #333333;")
        layout.addWidget(self.lbl_pts_count)
        
        # Toggle Modes
        mode_layout = QHBoxLayout()
        self.btn_mode_add = QPushButton("Add Point")
        self.btn_mode_add.setCheckable(True)
        self.btn_mode_add.setChecked(True)
        self.btn_mode_add.setStyleSheet("background-color: #0078D7; color: white; font-weight: bold;")
        
        self.btn_mode_remove = QPushButton("Remove Point")
        self.btn_mode_remove.setCheckable(True)
        self.btn_mode_remove.setStyleSheet("background-color: #E1E1E1; color: black; font-weight: bold;")
        
        self.btn_mode_add.clicked.connect(self._set_add_mode)
        self.btn_mode_remove.clicked.connect(self._set_remove_mode)
        
        mode_layout.addWidget(self.btn_mode_add)
        mode_layout.addWidget(self.btn_mode_remove)
        layout.addLayout(mode_layout)
        
        self.btn_clear_pts = QPushButton("Clear All Points")
        self.btn_clear_pts.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")
        self.btn_clear_pts.clicked.connect(self._on_clear_points)
        layout.addWidget(self.btn_clear_pts)
        
        layout.addWidget(QLabel("----------------------------------------"))
        
        # 3. Model Configuration & Running Section
        lbl_model_title = QLabel("<b>3. Model Tracking</b>")
        layout.addWidget(lbl_model_title)
        
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model Type:"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(["SpatialTrackerV2-Online", "SpatialTrackerV2-Offline"])
        self.combo_model.setCurrentText("SpatialTrackerV2-Online")
        self.combo_model.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.combo_model)
        layout.addLayout(model_layout)
        
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("VO Points:"))
        self.spin_vo = QSpinBox()
        self.spin_vo.setRange(10, 1000)
        self.spin_vo.setValue(120)
        step_layout.addWidget(self.spin_vo)
        layout.addLayout(step_layout)
        
        swind_layout = QHBoxLayout()
        swind_layout.addWidget(QLabel("Window Size (S_wind):"))
        self.spin_swind = QSpinBox()
        self.spin_swind.setRange(5, 1000)
        self.spin_swind.setValue(30)
        swind_layout.addWidget(self.spin_swind)
        layout.addLayout(swind_layout)
        
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("Overlap Size:"))
        self.spin_overlap = QSpinBox()
        self.spin_overlap.setRange(0, 500)
        self.spin_overlap.setValue(10)
        overlap_layout.addWidget(self.spin_overlap)
        layout.addLayout(overlap_layout)
        
        self.cb_fixed_cam = QCheckBox("Fixed Camera Pose")
        self.cb_fixed_cam.setChecked(False)
        self.cb_fixed_cam.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.cb_fixed_cam)
        
        self.btn_process = QPushButton("Run Point Tracking [process]")
        self.btn_process.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 6px;")
        self.btn_process.clicked.connect(self._on_process)
        layout.addWidget(self.btn_process)
        
        self.lbl_status = QLabel("Status: Idle")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #555555;")
        layout.addWidget(self.lbl_status)
        
        layout.addWidget(QLabel("----------------------------------------"))
        
        # 4. Results Saving/Loading Section
        lbl_persist_title = QLabel("<b>4. Session / Persistence</b>")
        layout.addWidget(lbl_persist_title)
        
        self.btn_save_result = QPushButton("Save Tracking Result [save result]")
        self.btn_save_result.setEnabled(False)
        self.btn_save_result.setStyleSheet("background-color: #E1E1E1; font-weight: bold; padding: 4px;")
        self.btn_save_result.clicked.connect(self._on_save_result)
        layout.addWidget(self.btn_save_result)
        
        self.btn_gen_blender = QPushButton("Generate Blender Script")
        self.btn_gen_blender.setEnabled(False)
        self.btn_gen_blender.setStyleSheet("background-color: #E1E1E1; font-weight: bold; padding: 4px;")
        self.btn_gen_blender.clicked.connect(self._on_generate_blender_script)
        layout.addWidget(self.btn_gen_blender)
        
        self.btn_load_result = QPushButton("Load Tracking Result [load result]")
        self.btn_load_result.setStyleSheet("background-color: #E1E1E1; font-weight: bold; padding: 4px;")
        self.btn_load_result.clicked.connect(self._on_load_result)
        layout.addWidget(self.btn_load_result)
        
        layout.addStretch()

    def showEvent(self, event):
        super().showEvent(event)
        self.video_path = self.session.get('video_path', '')
        self._try_auto_load_preprocess()
        
        # Set viewport interaction mode
        self.session.set("interaction_mode", {"mode": "point", "cursor": "cross", "drag": "none"})
        self.update_overlay()

    def hideEvent(self, event):
        super().hideEvent(event)
        # Restore viewport interaction mode
        self.session.set("interaction_mode", {"mode": "scroll", "cursor": "standard", "drag": "scroll"})
        # Clear overlays
        self.session.set("overlay_data", None)

    def _on_model_changed(self, model_name):
        if model_name == "SpatialTrackerV2-Online":
            self.spin_swind.setValue(30)
            self.spin_overlap.setValue(10)
        else:
            self.spin_swind.setValue(500)
            self.spin_overlap.setValue(4)

    def get_point_color(self, pt_idx):
        if not self.points or pt_idx >= len(self.points):
            return QColor(0, 255, 0)
        
        x, y = self.points[pt_idx]
        W = self.video_width or 1920
        H = self.video_height or 1080
        
        cx, cy = W / 2.0, H / 2.0
        dx, dy = x - cx, y - cy
        
        angle = np.arctan2(dy, dx)
        hue = int(((angle + np.pi) / (2.0 * np.pi)) * 359.0)
        
        max_d = np.sqrt(cx**2 + cy**2)
        d = np.sqrt(dx**2 + dy**2)
        sat = int(180 + 75 * (d / max_d))
        sat = min(max(sat, 180), 255)
        
        return QColor.fromHsv(hue, sat, 255)

    def robust_squeeze_tracks(self, arr):
        if arr is None:
            return None
        while arr.ndim > 3:
            squeezed = False
            for axis in range(arr.ndim - 2):
                if arr.shape[axis] == 1:
                    arr = np.squeeze(arr, axis=axis)
                    squeezed = True
                    break
            if not squeezed:
                break
        return arr

    def robust_squeeze_visibs(self, arr):
        if arr is None:
            return None
        while arr.ndim > 2:
            squeezed = False
            for axis in range(arr.ndim - 1):
                if arr.shape[axis] == 1:
                    arr = np.squeeze(arr, axis=axis)
                    squeezed = True
                    break
            if not squeezed:
                break
        return arr

    def on_video_loaded(self, metadata):
        self.video_path = metadata.get("video_path", "")
        self.video_width = metadata.get("width", 0)
        self.video_height = metadata.get("height", 0)
        self.pts_map = metadata.get("pts_map", [])
        self.points.clear()
        self.tracks_2d = None
        self.coords_3d = None
        self.visibs = None
        self.tracking_results = None
        self.btn_save_result.setEnabled(False)
        self.btn_gen_blender.setEnabled(False)
        self.btn_gen_blender.setStyleSheet("background-color: #E1E1E1; font-weight: bold; padding: 4px;")
        self.lbl_pts_count.setText("Selected Points: 0")
        self._try_auto_load_preprocess()
        self.update_overlay()

    def decode_frames_via_signal(self, start, end, step):
        result = [None]
        def set_result(generator):
            result[0] = generator
        self.decode_requested.emit(start, end, step, set_result)
        return result[0]

    def on_frame_changed(self, frame_idx, current_time_sec):
        self.update_overlay()

    def _set_add_mode(self):
        self.mode = "add"
        self.btn_mode_add.setChecked(True)
        self.btn_mode_add.setStyleSheet("background-color: #0078D7; color: white; font-weight: bold;")
        self.btn_mode_remove.setChecked(False)
        self.btn_mode_remove.setStyleSheet("background-color: #E1E1E1; color: black; font-weight: bold;")

    def _set_remove_mode(self):
        self.mode = "remove"
        self.btn_mode_add.setChecked(False)
        self.btn_mode_add.setStyleSheet("background-color: #E1E1E1; color: black; font-weight: bold;")
        self.btn_mode_remove.setChecked(True)
        self.btn_mode_remove.setStyleSheet("background-color: #0078D7; color: white; font-weight: bold;")

    def _on_clear_points(self):
        self.points.clear()
        self.tracks_2d = None
        self.coords_3d = None
        self.visibs = None
        self.tracking_results = None
        self.btn_save_result.setEnabled(False)
        self.btn_gen_blender.setEnabled(False)
        self.btn_gen_blender.setStyleSheet("background-color: #E1E1E1; font-weight: bold; padding: 4px;")
        self.lbl_pts_count.setText("Selected Points: 0")
        self.lbl_status.setText("Status: Cleared all points.")
        self.update_overlay()

    def _on_view_clicked(self, scene_x, scene_y):
        if not self.video_path:
            return
            
        x = int(round(scene_x))
        y = int(round(scene_y))
        
        current_frame = self.session.get("current_frame", 0)
        if current_frame != self.start_frame:
            self.lbl_status.setText(f"Status: Jumped to first frame {self.start_frame} to manage points.")
            self.session.set("seek_frame", self.start_frame)
            return

        if self.mode == "add":
            if (x, y) not in self.points:
                self.points.append((x, y))
                self.lbl_pts_count.setText(f"Selected Points: {len(self.points)}")
                self.lbl_status.setText(f"Status: Added point at ({x}, {y})")
        else:
            if self.points:
                dists = [np.sqrt((x - px)**2 + (y - py)**2) for px, py in self.points]
                min_idx = np.argmin(dists)
                if dists[min_idx] < 15:
                    removed = self.points.pop(min_idx)
                    self.lbl_pts_count.setText(f"Selected Points: {len(self.points)}")
                    self.lbl_status.setText(f"Status: Removed point at {removed}")
                    self.tracks_2d = None
                    self.coords_3d = None
                    self.visibs = None
                    self.tracking_results = None
                    self.btn_save_result.setEnabled(False)
                    self.btn_gen_blender.setEnabled(False)
                    self.btn_gen_blender.setStyleSheet("background-color: #E1E1E1; font-weight: bold; padding: 4px;")

        self.update_overlay()

    def _try_auto_load_preprocess(self):
        npz = self.session.get("preprocess_npz")
        json_path = self.session.get("preprocess_json")
        if npz and os.path.exists(npz) and json_path and os.path.exists(json_path):
            self._apply_preprocess_paths(npz, json_path)

    def _apply_preprocess_paths(self, npz_path, json_path):
        self.preprocess_npz = npz_path
        self.preprocess_json = json_path
        
        try:
            with open(json_path, 'r') as f:
                meta = json.load(f)
            self.start_frame = meta.get("start_frame", 0)
            self.end_frame = meta.get("end_frame", 0)
            self.step = meta.get("step", 1)
            
            self.lbl_prep_status.setText(f"Preprocess: Loaded ({self.start_frame}-{self.end_frame}, step {self.step})")
            self.lbl_prep_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.lbl_status.setText("Status: Preprocess synced successfully.")
        except Exception as e:
            self.lbl_prep_status.setText("Preprocess: Error reading metadata")
            self.lbl_prep_status.setStyleSheet("color: #F44336; font-weight: bold;")
            self.lbl_status.setText(f"Error: {e}")

    def _on_load_preprocess(self):
        from gui_preprocess_loader import load_preprocess_metadata
        try:
            meta, filename = load_preprocess_metadata(self.session, self)
            if not meta:
                return
                
            npz_path = meta.get("npz_path") or meta.get("preprocess_npz")
            self.session.set("preprocess_npz", npz_path)
            self.session.set("preprocess_json", filename)
            self._apply_preprocess_paths(npz_path, filename)
        except Exception as e:
            self.lbl_status.setText(f"Status: Preprocess load failed! {e}")
            self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")

    def _on_process(self):
        if not self.video_path:
            self.lbl_status.setText("Status: Error - No video loaded!")
            return
        if not self.preprocess_npz or not os.path.exists(self.preprocess_npz):
            self.lbl_status.setText("Status: Error - Preprocessed intermediate data not loaded!")
            return
        if not self.points:
            self.lbl_status.setText("Status: Error - No tracking points added! Click on frame to add points.")
            return

        self.btn_process.setEnabled(False)
        self.btn_load_prep.setEnabled(False)
        self.btn_clear_pts.setEnabled(False)
        self.btn_load_result.setEnabled(False)
        self.btn_save_result.setEnabled(False)
        self.btn_gen_blender.setEnabled(False)
        self.btn_gen_blender.setStyleSheet("background-color: #E1E1E1; font-weight: bold; padding: 4px;")
        self.lbl_status.setText("Status: Preparing tracking background thread...")
        self.lbl_status.setStyleSheet("color: #0078D7; font-weight: bold;")

        # Fetch decoder from session state and construct synchronous decode closure
        decoder = self.session.get("decoder")
        if not decoder:
            self.lbl_status.setText("Status: Error - Decoder not initialized in session!")
            self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")
            self.btn_process.setEnabled(True)
            self.btn_load_prep.setEnabled(True)
            self.btn_clear_pts.setEnabled(True)
            self.btn_load_result.setEnabled(True)
            return
            
        def decode_fn(s, e, st):
            return decoder.decode_current_video_frames(s, e, st)

        model_type = self.combo_model.currentText()
        model_name = f"Yuxihenry/{model_type}"
        s_wind = self.spin_swind.value()
        overlap = min(self.spin_overlap.value(), s_wind - 1)

        self.thread = TrackingThread(
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            step=self.step,
            npz_preprocess_path=self.preprocess_npz,
            points=self.points,
            vo_points=self.spin_vo.value(),
            fixed_cam=self.cb_fixed_cam.isChecked(),
            model_name=model_name,
            s_wind=s_wind,
            overlap=overlap,
            decode_fn=decode_fn
        )
        self.thread.progress.connect(self._on_thread_status)
        self.thread.finished.connect(self._on_thread_finished)
        self.thread.start()

    def _on_thread_status(self, msg):
        self.lbl_status.setText(f"Status: {msg}")

    def _on_thread_finished(self, success, results, err_msg):
        self.btn_process.setEnabled(True)
        self.btn_load_prep.setEnabled(True)
        self.btn_clear_pts.setEnabled(True)
        self.btn_load_result.setEnabled(True)
        
        if success:
            self.tracking_results = results
            self.tracks_2d = self.robust_squeeze_tracks(results["tracks_2d"])
            if self.tracks_2d is not None and self.tracks_2d.shape[-1] >= 2:
                self.tracks_2d = self.tracks_2d[..., :2]
            self.coords_3d = self.robust_squeeze_tracks(results["coords"])
            self.visibs = self.robust_squeeze_visibs(results["visibs"])
            
            self.btn_save_result.setEnabled(True)
            self.btn_gen_blender.setEnabled(True)
            self.btn_gen_blender.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 4px;")
            self.lbl_status.setText("Status: Tracking completed successfully! Play video to see trajectories.")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.lbl_status.setText(f"Status: Tracking failed! Error: {err_msg}")
            self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")
        self.update_overlay()

    def get_interpolated_tracks(self, frame_idx):
        if self.tracks_2d is None:
            return None
            
        if frame_idx < self.start_frame or frame_idx > self.end_frame:
            return None
            
        num_steps = self.tracks_2d.shape[0]
        
        if self.step == 1:
            idx = frame_idx - self.start_frame
            if 0 <= idx < num_steps:
                return self.tracks_2d[idx]
            elif idx >= num_steps:
                return self.tracks_2d[-1]
            else:
                return self.tracks_2d[0]
                
        float_idx = (frame_idx - self.start_frame) / self.step
        
        if float_idx <= 0:
            return self.tracks_2d[0]
        if float_idx >= num_steps - 1:
            return self.tracks_2d[-1]
            
        k = int(np.floor(float_idx))
        ratio = float_idx - k
        
        pos_prev = self.tracks_2d[k]
        pos_next = self.tracks_2d[k + 1]
        
        return (1.0 - ratio) * pos_prev + ratio * pos_next

    def update_overlay(self):
        if not self.video_path:
            self.session.set("overlay_data", None)
            return
            
        current_frame = self.session.get("current_frame", 0)
        
        if self.tracks_2d is not None:
            overlay_data = {
                "tracks_2d": self.tracks_2d,
                "points": self.points,
                "current_frame": current_frame,
                "start_frame": self.start_frame,
                "end_frame": self.end_frame,
                "step": self.step
            }
            self.session.set("overlay_data", overlay_data)
        else:
            text_message = None
            if current_frame != self.start_frame:
                if self.points and not (self.start_frame <= current_frame <= self.end_frame):
                    pass
                elif self.points:
                    text_message = f"Go to Trim Start frame {self.start_frame} to edit points"
                    
            overlay_data = {
                "points": self.points,
                "current_frame": current_frame,
                "start_frame": self.start_frame,
                "end_frame": self.end_frame,
                "text": text_message
            }
            self.session.set("overlay_data", overlay_data)

    def _on_save_result(self):
        if self.tracking_results is None:
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Tracking Results", "", "NPZ Files (*_result.npz)"
        )
        if not filename:
            return
            
        try:
            if not filename.endswith("_result.npz"):
                base, _ = os.path.splitext(filename)
                filename = f"{base}_result.npz"
                
            np.savez(filename, **self.tracking_results)
            
            meta_path = filename.replace("_result.npz", "_result_metadata.json")
            metadata = {
                "video_path": os.path.abspath(self.video_path),
                "start_frame": self.start_frame,
                "end_frame": self.end_frame,
                "step": self.step,
                "preprocess_npz": os.path.abspath(self.preprocess_npz),
                "preprocess_json": os.path.abspath(self.preprocess_json),
                "points": self.points,
                "vo_points": self.spin_vo.value(),
                "result_npz": os.path.abspath(filename),
                "video_width": self.video_width,
                "video_height": self.video_height
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            self.last_saved_npz = os.path.abspath(filename)
            self.lbl_status.setText(f"Status: Saved result to {os.path.basename(filename)}")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        except Exception as e:
            self.lbl_status.setText(f"Status: Save failed! Error: {e}")
            self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")

    def _on_load_result(self):
        from gui_preprocess_loader import load_tracking_result
        try:
            meta, filename = load_tracking_result(self.session, self)
            if not meta:
                return
                
            result_npz = meta.get("result_npz")
            preprocess_npz = meta.get("preprocess_npz")
            preprocess_json = meta.get("preprocess_json")
            
            self.preprocess_npz = preprocess_npz
            self.preprocess_json = preprocess_json
            
            self.start_frame = meta.get("start_frame", 0)
            self.end_frame = meta.get("end_frame", 0)
            self.step = meta.get("step", 1)
            self.points = meta.get("points", [])
            self.spin_vo.setValue(meta.get("vo_points", 120))
            self.video_width = meta.get("video_width", 0)
            self.video_height = meta.get("video_height", 0)
            
            # Publish loaded preprocess paths to session state so other tools auto-detect them
            self.session.set("preprocess_npz", preprocess_npz)
            self.session.set("preprocess_json", preprocess_json)
            self.session.set("tracking_result_npz", result_npz)
            
            self.session.set("trim_range", (self.start_frame, self.end_frame))
            
            self.lbl_status.setText("Loading tracking results into memory...")
            results_npz = np.load(result_npz, allow_pickle=True)
            self.tracking_results = dict(results_npz)
            self.last_saved_npz = os.path.abspath(result_npz)
            
            self.tracks_2d = self.robust_squeeze_tracks(self.tracking_results["tracks_2d"])
            if self.tracks_2d is not None and self.tracks_2d.shape[-1] >= 2:
                self.tracks_2d = self.tracks_2d[..., :2]
            self.coords_3d = self.robust_squeeze_tracks(self.tracking_results["coords"])
            self.visibs = self.robust_squeeze_visibs(self.tracking_results["visibs"])
            
            self.lbl_pts_count.setText(f"Selected Points: {len(self.points)}")
            self.lbl_prep_status.setText(f"Preprocess: Loaded ({self.start_frame}-{self.end_frame}, step {self.step})")
            self.lbl_prep_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
            self.btn_save_result.setEnabled(True)
            self.btn_gen_blender.setEnabled(True)
            self.btn_gen_blender.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 4px;")
            self.lbl_status.setText(f"Status: Session loaded successfully! Track results ready.")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
            self.session.set("seek_frame", self.start_frame)
            self.update_overlay()
            
        except Exception as e:
            self.lbl_status.setText(f"Status: Load session failed! Error: {e}")
            self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")
            self.update_overlay()

    def _export_pc_npz(self, pc_path, stride=8, conf_threshold=0.5, min_depth=0.1, max_depth=20.0):
        if self.tracking_results is None:
            return False
            
        depths = self.tracking_results.get("depths")
        unc_metric = self.tracking_results.get("unc_metric")
        intrinsics = self.tracking_results.get("intrinsics")
        extrinsics = self.tracking_results.get("extrinsics")
        
        if depths is None or extrinsics is None or intrinsics is None:
            QMessageBox.critical(self, "Export Failed", "Missing depth, intrinsics, or extrinsics.")
            return False
            
        T = len(depths)
        H_depth, W_depth = depths.shape[1], depths.shape[2]
        progress = QProgressDialog("Decoding video frames for point cloud...", "Cancel", 0, T + 10, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.processEvents()
        
        decode_fn = None
        def set_fn(fn):
            nonlocal decode_fn
            decode_fn = fn
        self.decode_requested.emit(self.start_frame, self.end_frame, self.step, set_fn)
        if not decode_fn:
            QMessageBox.critical(self, "Export Failed", "Failed to retrieve decoder function.")
            return False
            
        needed_frame_indices = set(self.start_frame + t * self.step for t in range(T))
        decoded_frames = {}
        try:
            generator = decode_fn(self.start_frame, self.end_frame, self.step)
            for idx, frame_arr in generator:
                if progress.wasCanceled():
                    return False
                decoded_frames[idx] = frame_arr
                progress.setValue(int(len(decoded_frames) * T / len(needed_frame_indices)))
                QApplication.processEvents()
                if len(decoded_frames) == len(needed_frame_indices):
                    break
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error decoding video: {e}")
            return False
            
        if len(decoded_frames) == 0:
            QMessageBox.critical(self, "Export Failed", "No video frames could be decoded.")
            return False
        first_idx = next(iter(decoded_frames))
        H_orig, W_orig = decoded_frames[first_idx].shape[0], decoded_frames[first_idx].shape[1]
        scale_x = W_depth / W_orig
        scale_y = H_depth / H_orig
        progress.setLabelText("Lifting depth maps to 3D point cloud sequence...")
        progress.setValue(T)
        QApplication.processEvents()
        M = (H_depth // stride) * (W_depth // stride)
        points_seq = np.zeros((T, M, 3), dtype=np.float32)
        colors_seq = np.zeros((T, M, 3), dtype=np.uint8)
        for t in range(T):
            if progress.wasCanceled():
                return False
            frame_idx = self.start_frame + t * self.step
            if frame_idx not in decoded_frames:
                continue
            img_rgb = decoded_frames[frame_idx]
            depth = depths[t]
            conf = unc_metric[t] if unc_metric is not None else None
            K = intrinsics[t]
            w2c = extrinsics[t]
            try:
                c2w_cv = np.linalg.inv(w2c)
            except np.linalg.LinAlgError:
                continue
            cam_x = c2w_cv[0, 3]
            cam_y = c2w_cv[2, 3]
            cam_z = -c2w_cv[1, 3]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            fx_depth = fx * scale_x
            fy_depth = fy * scale_y
            cx_depth = cx * scale_x
            cy_depth = cy * scale_y
            pt_idx = 0
            for y in range(0, H_depth, stride):
                for x in range(0, W_depth, stride):
                    if pt_idx >= M:
                        break
                    d = depth[y, x]
                    is_valid = True
                    if conf is not None:
                        c_val = conf[y, x]
                        if isinstance(c_val, (bool, np.bool_)):
                            if not c_val:
                                is_valid = False
                        else:
                            if c_val < conf_threshold:
                                is_valid = False
                    if d < min_depth or d > max_depth:
                        is_valid = False
                    if is_valid:
                        X_cam = (x - cx_depth) * d / fx_depth
                        Y_cam = (y - cy_depth) * d / fy_depth
                        Z_cam = d
                        P_cam = np.array([X_cam, Y_cam, Z_cam, 1.0])
                        P_world_cv = c2w_cv @ P_cam
                        points_seq[t, pt_idx] = [P_world_cv[0], P_world_cv[2], -P_world_cv[1]]
                        u_orig = int(np.clip(round(x / scale_x), 0, W_orig - 1))
                        v_orig = int(np.clip(round(y / scale_y), 0, H_orig - 1))
                        colors_seq[t, pt_idx] = img_rgb[v_orig, u_orig]
                    else:
                        points_seq[t, pt_idx] = [cam_x, cam_y, cam_z]
                        colors_seq[t, pt_idx] = [0, 0, 0]
                    pt_idx += 1
            progress.setValue(T + int((t + 1) * 10 / T))
            QApplication.processEvents()
        progress.setLabelText("Writing point cloud NPZ file...")
        QApplication.processEvents()
        try:
            np.savez(pc_path, points=points_seq, colors=colors_seq)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to write NPZ file: {e}")
            return False
        finally:
            progress.close()

    def _on_generate_blender_script(self):
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "No video loaded!")
            return
            
        npz_path = getattr(self, 'last_saved_npz', '')
        if not npz_path:
            reply = QMessageBox.question(
                self, 
                "Save Tracking Result First", 
                "The tracking result needs to be saved as an NPZ file before generating the Blender script, so the script can reference the correct absolute file path.\n\nWould you like to save the tracking result now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._on_save_result()
                npz_path = getattr(self, 'last_saved_npz', '')
                if not npz_path:
                    return
            else:
                return
                
        default_pc_path = npz_path.replace("_result.npz", "_pc.npz")
        if not default_pc_path.endswith(".npz"):
            default_pc_path = os.path.splitext(npz_path)[0] + "_pc.npz"
            
        pc_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Animated Point Cloud Data (Cancel to skip)", 
            default_pc_path, 
            "NPZ Files (*.npz)"
        )
        
        if pc_path:
            ply_ok = self._export_pc_npz(pc_path)
            if not ply_ok:
                reply = QMessageBox.question(
                    self,
                    "Point Cloud Export Failed",
                    "Point cloud export failed or was cancelled. Would you like to generate the Blender script without the point cloud?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
                pc_path = ""
                
        w = getattr(self, 'video_width', 0)
        h = getattr(self, 'video_height', 0)
        if w <= 0 or h <= 0:
            w, h = get_video_dimensions(self.video_path)
            self.video_width = w
            self.video_height = h
            
        stride = self.step
        start_frame = self.start_frame
        
        try:
            script_code = generate_blender_script(
                npz_path=npz_path,
                pc_npz_path=pc_path,
                video_path=self.video_path,
                width=w,
                height=h,
                stride=stride,
                start_frame=start_frame
            )
            
            clipboard = QApplication.clipboard()
            clipboard.setText(script_code)
            
            old_style = self.btn_gen_blender.styleSheet()
            old_text = self.btn_gen_blender.text()
            
            self.btn_gen_blender.setText("Copied to Clipboard!")
            self.btn_gen_blender.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 4px;")
            self.lbl_status.setText("Status: Blender import script copied directly to clipboard!")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
            QTimer.singleShot(1500, lambda: self.btn_gen_blender.setText(old_text))
            QTimer.singleShot(1500, lambda: self.btn_gen_blender.setStyleSheet(old_style))
            
        except Exception as e:
            QMessageBox.critical(self, "Generation Failed", f"Failed to generate Blender script:\n{e}")

    def _on_box_selected(self, x1, y1, x2, y2):
        if not self.video_path:
            return
            
        current_frame = self.session.get("current_frame", 0)
        if current_frame != self.start_frame:
            self.lbl_status.setText(f"Status: Jumped to first frame {self.start_frame} to manage points.")
            self.session.set("seek_frame", self.start_frame)
            return