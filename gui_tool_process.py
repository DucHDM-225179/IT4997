import os
import gc
import json
import torch
import torch.nn.functional as F
import av
import cv2
import numpy as np
from PyQt6.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QSpinBox, QWidget, QFileDialog, QButtonGroup, QGraphicsView)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsSimpleTextItem

from gui_tool_base import BaseTool
from models.SpaTrackV2.models.predictor import Predictor
from models.monoD.depth_anything_v2.util.transform import Resize

class TrackingThread(QThread):
    """Background thread that runs the SpatialTrackerV2 tracking Predictor model on CUDA."""
    progress = pyqtSignal(str) # Status messages
    finished = pyqtSignal(bool, dict, str) # (success, results_dict, error_msg)

    def __init__(self, video_path, start_frame, end_frame, step, npz_preprocess_path, points, vo_points=120, pts_map=None):
        super().__init__()
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.step = step
        self.npz_preprocess_path = npz_preprocess_path
        self.points = points  # List of (x, y) in original HD resolution
        self.vo_points = vo_points
        self.pts_map = pts_map or []

    def run(self):
        try:
            if not self.points:
                raise ValueError("No tracking points selected. Please click on the first frame of the trim to add points.")

            self.progress.emit("Opening video with PyAV for tracking sequence...")
            container = av.open(self.video_path, container_options={'ignore_editlist': '1'})
            video_stream = container.streams.video[0]
            video_stream.thread_type = "AUTO"
            W_orig, H_orig = video_stream.width, video_stream.height

            # Count total frames to decode
            total_frames_to_decode = 0
            for i in range(self.start_frame, self.end_frame + 1):
                if (i - self.start_frame) % self.step == 0:
                    total_frames_to_decode += 1

            # Seek to start
            if self.pts_map and self.start_frame < len(self.pts_map):
                container.seek(self.pts_map[self.start_frame], stream=video_stream, backward=True)

            self.progress.emit(f"Decoding and resizing {total_frames_to_decode} frames for tracking...")
            
            resizer = Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
            )
            new_W, new_H = resizer.get_size(W_orig, H_orig)

            frames = []
            count = 0
            for frame in container.decode(video=0):
                if frame.pts is None:
                    continue
                try:
                    idx = self.pts_map.index(frame.pts) if self.pts_map else count
                except ValueError:
                    continue

                if idx < self.start_frame:
                    continue
                if idx > self.end_frame:
                    break

                if (idx - self.start_frame) % self.step == 0:
                    img = frame.to_rgb().to_ndarray()
                    img_resized = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_CUBIC)
                    frames.append(img_resized)
                    count += 1
                    if count >= total_frames_to_decode:
                        break

            container.close()

            if len(frames) == 0:
                raise ValueError("No video frames were successfully decoded for the tracking trim range.")

            video_np = np.stack(frames)
            video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float()

            self.progress.emit("Loading intermediate preprocessed depth/poses...")
            preprocess_data = np.load(self.npz_preprocess_path, allow_pickle=True)
            depth_in = preprocess_data["depths"]
            intrs_in = preprocess_data["intrinsics"]
            extrs_in = preprocess_data["extrinsics"]
            unc_metric_in = preprocess_data["unc_metric"]

            self.progress.emit("Interpolating depth and camera parameters to matched resolution...")
            # Resize depth and uncertainty to match the resized video tensor
            depth_tensor = F.interpolate(torch.from_numpy(depth_in)[:, None], 
                                         size=video_tensor.shape[2:], 
                                         mode='bilinear', align_corners=True).squeeze(1).numpy()
            unc_metric = F.interpolate(torch.from_numpy(unc_metric_in)[:, None].float(), 
                                        size=video_tensor.shape[2:], 
                                        mode='bilinear', align_corners=True).squeeze(1).numpy() > 0.5

            # Scale camera intrinsics
            scale_w = video_tensor.shape[3] / W_orig
            scale_h = video_tensor.shape[2] / H_orig
            intrs_in[:, 0, :] *= scale_w
            intrs_in[:, 1, :] *= scale_h

            # Scale query points from HD original to preprocessed resized resolution
            self.progress.emit("Preparing tracking query coordinates...")
            query_xyt = np.zeros((len(self.points), 3), dtype=np.float32)
            for idx, (x_hd, y_hd) in enumerate(self.points):
                x_proc = x_hd * scale_w
                y_proc = y_hd * scale_h
                query_xyt[idx, 0] = 0.0 # Queries correspond to frame 0 of sequence
                query_xyt[idx, 1] = x_proc
                query_xyt[idx, 2] = y_proc

            self.progress.emit("Loading Predictor model (Online mode) to CUDA...")
            model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
            model.spatrack.track_num = self.vo_points
            model.eval()
            model.to("cuda")

            if hasattr(model.spatrack, "base_model") and model.spatrack.base_model is not None:
                model.spatrack.base_model.to("cpu")
                torch.cuda.empty_cache()

            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.progress.emit(f"Running tracking forward pass (mixed precision: {dtype})...")
            
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    (
                        c2w_traj, intrs, point_map, conf_depth,
                        track3d_pred, track2d_pred, vis_pred, conf_pred, video
                    ) = model.forward(video_tensor, depth=depth_tensor,
                                        intrs=intrs_in, extrs=extrs_in, 
                                        queries=query_xyt,
                                        fps=1, full_point=False, iters_track=4,
                                        query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
                                        support_frame=len(video_tensor)-1, replace_ratio=0.2)

            self.progress.emit("Post-processing tracking results back to HD dimensions...")
            # Scale 2D tracks and camera intrinsics back to original HD resolution
            track2d_pred[..., 0] *= W_orig / video_tensor.shape[3]
            track2d_pred[..., 1] *= H_orig / video_tensor.shape[2]
            intrs[:, 0, :] *= W_orig / video_tensor.shape[3]
            intrs[:, 1, :] *= H_orig / video_tensor.shape[2]

            # Reconstruct world coords and formats for saving
            results = {}
            results["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
            results["tracks_2d"] = track2d_pred.cpu().numpy()
            results["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
            results["intrinsics"] = intrs.cpu().numpy()
            results["visibs"] = vis_pred.cpu().numpy()
            results["unc_metric"] = conf_pred.cpu().numpy()

            self.progress.emit("Unloading model and freeing GPU cache...")
            del model
            torch.cuda.empty_cache()
            gc.collect()

            self.finished.emit(True, results, "")
        except Exception as e:
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()
            gc.collect()
            self.finished.emit(False, {}, str(e))


class ProcessVideoTool(BaseTool):
    """Tool to place points, load intermediate depth, run CUDA tracker, and visualize trajectories."""
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        
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
        
        self.overlay_items = []
        self.thread = None
        
        self._init_ui()
        
        # Connect to main window view click signals
        self.main_window.video_view.pixelClicked.connect(self._on_view_clicked)

    def get_name(self):
        return "Process Video (Point Tracking)"

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
        
        # Connect toggling to maintain mutual exclusion
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
        
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("VO Points:"))
        self.spin_vo = QSpinBox()
        self.spin_vo.setRange(10, 1000)
        self.spin_vo.setValue(120)
        step_layout.addWidget(self.spin_vo)
        layout.addLayout(step_layout)
        
        self.btn_process = QPushButton("Run Point Tracking [process]")
        self.btn_process.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 6px;")
        self.btn_process.clicked.connect(self._on_process)
        layout.addWidget(self.btn_process)
        
        # Status
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
        
        self.btn_load_result = QPushButton("Load Tracking Result [load result]")
        self.btn_load_result.setStyleSheet("background-color: #E1E1E1; font-weight: bold; padding: 4px;")
        self.btn_load_result.clicked.connect(self._on_load_result)
        layout.addWidget(self.btn_load_result)
        
        layout.addStretch()

    def showEvent(self, event):
        super().showEvent(event)
        self.video_path = getattr(self.main_window, 'current_video_path', '')
        # Try to automatically pull preprocessed path from PreprocessTool if loaded
        self._try_auto_load_preprocess()
        
        # Disable panning scroll hand drag, and use crosshair cursor for point tool
        if hasattr(self.main_window, 'video_view') and self.main_window.video_view:
            self.main_window.video_view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.main_window.video_view.viewport().setCursor(Qt.CursorShape.CrossCursor)
            
        self.update_overlay()

    def hideEvent(self, event):
        super().hideEvent(event)
        # Restore standard scroll hand drag and arrow cursor
        if hasattr(self.main_window, 'video_view') and self.main_window.video_view:
            self.main_window.video_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.main_window.video_view.viewport().unsetCursor()
        
        # Clear overlay items when tool is hidden
        for item in self.overlay_items:
            try:
                self.main_window.video_view.scene.removeItem(item)
            except Exception:
                pass
        self.overlay_items.clear()

    def get_point_color(self, pt_idx):
        """Generates a premium, highly-vibrant spatial color based on starting coordinates."""
        if not self.points or pt_idx >= len(self.points):
            return QColor(0, 255, 0)
        
        x, y = self.points[pt_idx]
        
        W = 1920
        H = 1080
        if hasattr(self.main_window.decoder_thread, 'container') and self.main_window.decoder_thread.container:
            try:
                video_stream = self.main_window.decoder_thread.container.streams.video[0]
                W, H = video_stream.width, video_stream.height
            except Exception:
                pass
        
        # Center coordinates
        cx = W / 2.0
        cy = H / 2.0
        dx = x - cx
        dy = y - cy
        
        # Standard color wheel angle map to Hue
        angle = np.arctan2(dy, dx)
        hue = int(((angle + np.pi) / (2.0 * np.pi)) * 359.0)
        
        # Distance from center normalized to saturate
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
            for axis in range(arr.ndim - 2): # don't touch N (ndim-2) or coords (ndim-1)
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
            for axis in range(arr.ndim - 1): # don't touch N (ndim-1)
                if arr.shape[axis] == 1:
                    arr = np.squeeze(arr, axis=axis)
                    squeezed = True
                    break
            if not squeezed:
                break
        return arr

    def on_video_loaded(self, metadata):
        self.video_path = getattr(self.main_window, 'current_video_path', '')
        self.points.clear()
        self.tracks_2d = None
        self.coords_3d = None
        self.visibs = None
        self.tracking_results = None
        self.btn_save_result.setEnabled(False)
        self.lbl_pts_count.setText("Selected Points: 0")
        self._try_auto_load_preprocess()
        self.update_overlay()

    def on_frame_changed(self, frame_idx, current_time_sec):
        # Dynamically redraw overlay coordinates for current playback frame index
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
        self.lbl_pts_count.setText("Selected Points: 0")
        self.lbl_status.setText("Status: Cleared all points.")
        self.update_overlay()

    def _on_view_clicked(self, scene_x, scene_y):
        """Triggered from main window QGraphicsView left-click."""
        if not self.video_path:
            return
            
        # Coordinates must be snapped to nearest integer
        x = int(round(scene_x))
        y = int(round(scene_y))
        
        # Point additions are only allowed on the first frame of the trimmed range
        current_frame = self.main_window.timeline_slider.value()
        if current_frame != self.start_frame:
            # Shift video viewer to the start frame of trim automatically to guide the user
            self.lbl_status.setText(f"Status: Jumped to first frame {self.start_frame} to manage points.")
            self.main_window.decoder_thread.seek_frame(self.start_frame)
            return

        if self.mode == "add":
            # Avoid duplicate coordinates
            if (x, y) not in self.points:
                self.points.append((x, y))
                self.lbl_pts_count.setText(f"Selected Points: {len(self.points)}")
                self.lbl_status.setText(f"Status: Added point at ({x}, {y})")
        else:
            # Remove nearest point if clicked within 15px radius
            if self.points:
                dists = [np.sqrt((x - px)**2 + (y - py)**2) for px, py in self.points]
                min_idx = np.argmin(dists)
                if dists[min_idx] < 15:
                    removed = self.points.pop(min_idx)
                    self.lbl_pts_count.setText(f"Selected Points: {len(self.points)}")
                    self.lbl_status.setText(f"Status: Removed point at {removed}")
                    # Clear generated tracking since points changed
                    self.tracks_2d = None
                    self.coords_3d = None
                    self.visibs = None
                    self.tracking_results = None
                    self.btn_save_result.setEnabled(False)

        self.update_overlay()

    def _try_auto_load_preprocess(self):
        """Checks if PreprocessTool has completed outputs in-memory or on-disk to auto-sync."""
        if not self.video_path:
            return
            
        # Get from sibling PreprocessTool in main window stack
        from gui_tool_preprocess import PreprocessTool
        for tool in self.main_window.tools:
            if isinstance(tool, PreprocessTool):
                npz, json_path = tool._get_output_paths()
                if npz and os.path.exists(npz) and os.path.exists(json_path):
                    self._apply_preprocess_paths(npz, json_path)
                    return

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
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Preprocessed Metadata", "", "JSON Files (*_metadata.json)"
        )
        if not filename:
            return
            
        try:
            with open(filename, 'r') as f:
                meta = json.load(f)
                
            video_path = meta.get("video_path")
            npz_path = meta.get("npz_path")
            
            if not os.path.exists(video_path):
                # Try relative dir resolution
                meta_dir = os.path.dirname(filename)
                rel_vid = os.path.join(meta_dir, os.path.basename(video_path))
                if os.path.exists(rel_vid):
                    video_path = rel_vid
                else:
                    raise FileNotFoundError(f"Associated video not found at: {video_path}")
            
            # Load video via main window if needed
            if os.path.abspath(self.video_path) != os.path.abspath(video_path):
                success = self.main_window.load_video(video_path)
                if not success:
                    raise RuntimeError("Failed to load associated video.")
            
            # Apply bounds
            start = meta.get("start_frame", 0)
            end = meta.get("end_frame", 0)
            self.main_window.apply_timeline_restriction(start, end)
            
            # Sync TrimTool labels
            from gui_tool_trim import TrimTool
            for tool in self.main_window.tools:
                if isinstance(tool, TrimTool):
                    tool.start_frame = start
                    tool.end_frame = end
                    tool._update_labels()
                    break
                    
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

        # Disable GUI controls
        self.btn_process.setEnabled(False)
        self.btn_load_prep.setEnabled(False)
        self.btn_clear_pts.setEnabled(False)
        self.btn_load_result.setEnabled(False)
        self.lbl_status.setText("Status: Preparing tracking background thread...")
        self.lbl_status.setStyleSheet("color: #0078D7; font-weight: bold;")

        pts_map = getattr(self.main_window.decoder_thread, 'pts_map', [])
        
        self.thread = TrackingThread(
            video_path=self.video_path,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            step=self.step,
            npz_preprocess_path=self.preprocess_npz,
            points=self.points,
            vo_points=self.spin_vo.value(),
            pts_map=pts_map
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
            self.lbl_status.setText("Status: Tracking completed successfully! Play video to see trajectories.")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.lbl_status.setText(f"Status: Tracking failed! Error: {err_msg}")
            self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")
        self.update_overlay()

    def get_interpolated_tracks(self, frame_idx):
        """Finds or interpolates coordinate tuples for points at a specific frame index."""
        if self.tracks_2d is None:
            return None
            
        if frame_idx < self.start_frame or frame_idx > self.end_frame:
            return None
            
        num_steps = self.tracks_2d.shape[0]
        
        # Pure copy branch for no frame drop (step == 1)
        if self.step == 1:
            idx = frame_idx - self.start_frame
            if 0 <= idx < num_steps:
                return self.tracks_2d[idx]
            elif idx >= num_steps:
                return self.tracks_2d[-1]
            else:
                return self.tracks_2d[0]
                
        # Interpolation branch for step > 1
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
        """Redraws point markers and trail trace connecting lines on QGraphicsScene."""
        # Clean old graphics items
        for item in self.overlay_items:
            try:
                self.main_window.video_view.scene.removeItem(item)
            except Exception:
                pass
        self.overlay_items.clear()
        
        if not self.video_path:
            return
            
        current_frame = self.main_window.timeline_slider.value()
        
        # 1. We have tracking results in memory -> Draw dynamic tracking points and trails
        if self.tracks_2d is not None:
            interpolated = self.get_interpolated_tracks(current_frame)
            if interpolated is None:
                # Outside trim boundaries: draw nothing
                return
                
            num_points = interpolated.shape[0]
            trail_len = 10
            
            # Palette matching for premium trails
            for pt_idx in range(num_points):
                # Trace past coordinates for fading spatial trail
                past_pts = []
                for offset in range(trail_len + 1):
                    f_past = current_frame - offset
                    if f_past < self.start_frame:
                        break
                    coords_past = self.get_interpolated_tracks(f_past)
                    if coords_past is not None:
                        past_pts.append(coords_past[pt_idx])
                
                # Draw fading trail path lines
                for idx, pt in enumerate(past_pts):
                    if idx == 0:
                        continue
                    pt_prev = past_pts[idx - 1]
                    opacity = max(0.15, 1.0 - (idx / trail_len))
                    base_color = self.get_point_color(pt_idx)
                    trail_color = QColor(base_color.red(), base_color.green(), base_color.blue(), int(opacity * 255))
                    pen = QPen(trail_color, 2)
                    line = QGraphicsLineItem(pt_prev[0], pt_prev[1], pt[0], pt[1])
                    line.setPen(pen)
                    self.main_window.video_view.scene.addItem(line)
                    self.overlay_items.append(line)
                
                # Draw main pointer dot
                x_curr, y_curr = interpolated[pt_idx]
                marker = QGraphicsEllipseItem(x_curr - 5, y_curr - 5, 10, 10)
                marker.setBrush(QBrush(self.get_point_color(pt_idx))) # Premium spatial color active dot
                marker.setPen(QPen(QColor(0, 0, 0), 1))
                self.main_window.video_view.scene.addItem(marker)
                self.overlay_items.append(marker)
                
        # 2. No tracking results -> Draw user-selected static points on the first frame
        else:
            if current_frame == self.start_frame:
                for pt_idx, (x, y) in enumerate(self.points):
                    marker = QGraphicsEllipseItem(x - 5, y - 5, 10, 10)
                    marker.setBrush(QBrush(self.get_point_color(pt_idx))) # Beautiful spatial color
                    marker.setPen(QPen(QColor(0, 0, 0), 1))
                    self.main_window.video_view.scene.addItem(marker)
                    self.overlay_items.append(marker)
            else:
                # Prompt user that point placement/management is on start frame
                if self.points and (current_frame < self.start_frame or current_frame > self.end_frame):
                    pass
                elif self.points:
                    # Draw a semi-transparent warning overlay or message
                    text = QGraphicsSimpleTextItem(f"Go to Trim Start frame {self.start_frame} to edit points")
                    text.setBrush(QBrush(QColor(255, 255, 255)))
                    text.setPos(15, 15)
                    self.main_window.video_view.scene.addItem(text)
                    self.overlay_items.append(text)

    def _on_save_result(self):
        if self.tracking_results is None:
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Tracking Results", "", "NPZ Files (*_result.npz)"
        )
        if not filename:
            return
            
        try:
            # Ensure file ending is correctly formatted
            if not filename.endswith("_result.npz"):
                base, _ = os.path.splitext(filename)
                filename = f"{base}_result.npz"
                
            np.savez(filename, **self.tracking_results)
            
            # Save companion metadata JSON to allow effortless session loads
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
                "result_npz": os.path.abspath(filename)
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            self.lbl_status.setText(f"Status: Saved result to {os.path.basename(filename)}")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        except Exception as e:
            self.lbl_status.setText(f"Status: Save failed! Error: {e}")
            self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")

    def _on_load_result(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Tracking Session", "", "JSON Files (*_result_metadata.json)"
        )
        if not filename:
            return
            
        try:
            with open(filename, 'r') as f:
                meta = json.load(f)
                
            video_path = meta.get("video_path")
            result_npz = meta.get("result_npz")
            preprocess_npz = meta.get("preprocess_npz")
            preprocess_json = meta.get("preprocess_json")
            
            # Check files exist
            if not os.path.exists(video_path):
                # Try relative resolution path
                meta_dir = os.path.dirname(filename)
                rel_vid = os.path.join(meta_dir, os.path.basename(video_path))
                if os.path.exists(rel_vid):
                    video_path = rel_vid
                else:
                    raise FileNotFoundError(f"Associated video not found at: {video_path}")
            
            # Load video via main window if needed
            if os.path.abspath(self.video_path) != os.path.abspath(video_path):
                success = self.main_window.load_video(video_path)
                if not success:
                    raise RuntimeError("Failed to load associated video.")
            
            # Load preprocess paths
            self.preprocess_npz = preprocess_npz
            self.preprocess_json = preprocess_json
            
            # Load sequence limits
            self.start_frame = meta.get("start_frame", 0)
            self.end_frame = meta.get("end_frame", 0)
            self.step = meta.get("step", 1)
            self.points = meta.get("points", [])
            self.spin_vo.setValue(meta.get("vo_points", 120))
            
            # Apply active trim restrictions
            self.main_window.apply_timeline_restriction(self.start_frame, self.end_frame)
            
            # Sync TrimTool
            from gui_tool_trim import TrimTool
            for tool in self.main_window.tools:
                if isinstance(tool, TrimTool):
                    tool.start_frame = self.start_frame
                    tool.end_frame = self.end_frame
                    tool._update_labels()
                    break
            
            # Load results NPZ
            self.lbl_status.setText("Loading tracking results into memory...")
            results_npz = np.load(result_npz, allow_pickle=True)
            self.tracking_results = dict(results_npz)
            
            self.tracks_2d = self.robust_squeeze_tracks(self.tracking_results["tracks_2d"])
            if self.tracks_2d is not None and self.tracks_2d.shape[-1] >= 2:
                self.tracks_2d = self.tracks_2d[..., :2]
            self.coords_3d = self.robust_squeeze_tracks(self.tracking_results["coords"])
            self.visibs = self.robust_squeeze_visibs(self.tracking_results["visibs"])
            
            self.lbl_pts_count.setText(f"Selected Points: {len(self.points)}")
            self.lbl_prep_status.setText(f"Preprocess: Loaded ({self.start_frame}-{self.end_frame}, step {self.step})")
            self.lbl_prep_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
            self.btn_save_result.setEnabled(True)
            self.lbl_status.setText(f"Status: Session loaded successfully! Track results ready.")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
            # Navigate to starting frame to show overlays instantly
            self.main_window.decoder_thread.seek_frame(self.start_frame)
            self.update_overlay()
            
        except Exception as e:
            self.lbl_status.setText(f"Status: Load session failed! Error: {e}")
            self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")
            self.update_overlay()
