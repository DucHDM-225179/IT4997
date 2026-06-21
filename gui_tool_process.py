import os
import gc
import json
import torch
import torch.nn.functional as F
import av
import cv2
import numpy as np
from PyQt6.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QSpinBox, QDoubleSpinBox, QWidget, QFileDialog, QButtonGroup, QGraphicsView,
                             QGroupBox, QRadioButton, QCheckBox, QComboBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import (QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsSimpleTextItem, 
                             QGraphicsRectItem, QMessageBox, QApplication)

from gui_tool_base import BaseTool
from models.SpaTrackV2.models.predictor import Predictor
from models.monoD.depth_anything_v2.util.transform import Resize


def get_video_dimensions(video_path):
    """Fallback function to safely read video dimensions using PyAV if the cached dimensions are empty."""
    import av
    try:
        container = av.open(video_path, container_options={'ignore_editlist': '1'})
        video_stream = container.streams.video[0]
        w, h = video_stream.width, video_stream.height
        container.close()
        return w, h
    except Exception as e:
        print(f"Error getting video dimensions: {e}")
        return 1920, 1080  # Reasonable HD fallback


def generate_blender_script(npz_path, pc_npz_path, video_path, width, height, stride, start_frame):
    """
    Reads the blender_loading_script_template.py file, substitutes placeholder strings
    with resolved configuration parameters, and returns the customized Blender import script.
    """
    # Normalized paths to use forward slashes so they execute safely in Blender on Windows
    npz_norm = os.path.abspath(npz_path).replace("\\", "/")
    pc_npz_norm = os.path.abspath(pc_npz_path).replace("\\", "/") if pc_npz_path else ""
    video_norm = os.path.abspath(video_path).replace("\\", "/")
    
    # Try reading the template file from the workspace/local directory
    # Template should be located in the project's root folder
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blender_loading_script_template.py")
    
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Blender script template file not found at: {template_path}")
        
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            template_code = f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read Blender script template file: {e}")

    # Clean string replacements of placeholders
    code = (template_code
            .replace('"TO_BE_REPLACED_NPZ_PATH"', f'"{npz_norm}"')
            .replace('"TO_BE_REPLACED_PC_NPZ_PATH"', f'"{pc_npz_norm}"' if pc_npz_norm else '""')
            .replace('"TO_BE_REPLACED_VIDEO_PATH"', f'"{video_norm}"')
            .replace('TO_BE_REPLACED_ORIGINAL_RES', f'({width}, {height})')
            .replace('TO_BE_REPLACED_FRAME_STRIDE', str(stride))
            .replace('TO_BE_REPLACED_START_FRAME', str(start_frame)))
            
    return code


def farthest_point_sampling(pts_3d: np.ndarray, num_points: int) -> np.ndarray:
    """
    Performs Farthest Point Sampling (FPS) on a 3D point cloud.
    pts_3d: np.ndarray of shape (N, 3)
    num_points: int
    Returns sampled points of shape (M, 3) where M = min(N, num_points).
    """
    N = pts_3d.shape[0]
    if N <= num_points:
        return pts_3d
        
    sampled_indices = []
    # Start with the point closest to the centroid of the point cloud
    centroid = np.mean(pts_3d, axis=0)
    dists = np.sum((pts_3d - centroid) ** 2, axis=1)
    start_idx = np.argmin(dists)
    sampled_indices.append(start_idx)
    
    # Initialize min distances array
    min_dists = np.sum((pts_3d - pts_3d[start_idx]) ** 2, axis=1)
    
    for _ in range(1, num_points):
        # Find point with the maximum distance from the already selected set
        next_idx = np.argmax(min_dists)
        sampled_indices.append(next_idx)
        
        # Update min distances with the new selected point
        new_dists = np.sum((pts_3d - pts_3d[next_idx]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        
    return pts_3d[sampled_indices]


def generate_points_in_mask(mask: np.ndarray, num_points: int = 50, depth: np.ndarray = None, K: np.ndarray = None, conf: np.ndarray = None, conf_threshold: float = 0.3) -> list:
    """
    Generates a cluster of points inside the binary mask (mask: HxW boolean/uint8 array).
    If depth, K, and conf are provided, lifts the pixels to a 3D point cloud, performs
    Farthest Point Sampling (FPS) in 3D to ensure even coverage avoiding boundary/edge noise,
    and de-lifts them back to 2D.
    Otherwise, falls back to 2D random sampling.
    """
    import random
    import cv2
    
    # If 3D camera geometry inputs are provided, run the advanced 3D lift-and-project sampling
    if depth is not None and K is not None and conf is not None:
        try:
            # 1. Erode the mask to avoid staying at the edges
            mask_pixels = np.sum(mask > 0)
            if mask_pixels == 0:
                return []
                
            radius = int(np.sqrt(mask_pixels) * 0.05)
            kernel_size = max(3, radius if radius % 2 == 1 else radius + 1)
            kernel_size = min(kernel_size, 31)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            eroded_mask = cv2.erode((mask > 0).astype(np.uint8), kernel, iterations=1)
            
            if np.sum(eroded_mask) == 0:
                eroded_mask = (mask > 0).astype(np.uint8)
                
            # 2. Combine with the uncertainty metric (confidence map)
            valid_pixels = (eroded_mask > 0) & (conf > conf_threshold)
            
            # Decay confidence threshold if too strict to ensure we get enough points
            if np.sum(valid_pixels) < num_points:
                valid_pixels = (eroded_mask > 0) & (conf > (conf_threshold * 0.5))
            if np.sum(valid_pixels) < num_points:
                valid_pixels = (eroded_mask > 0)
                
            v_indices, u_indices = np.where(valid_pixels)
            if len(u_indices) == 0:
                return []
                
            # 3. Lift 2D coordinates to 3D camera coordinates
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            
            d = depth[v_indices, u_indices]
            d = np.maximum(d, 0.01) # Avoid division by zero
            
            X = (u_indices - cx) * d / fx
            Y = (v_indices - cy) * d / fy
            Z = d
            
            pts_3d = np.stack([X, Y, Z], axis=1)
            
            # 4. Farthest Point Sampling in 3D
            sampled_pts_3d = farthest_point_sampling(pts_3d, num_points)
            
            # 5. De-lift back to 2D
            u_proj = sampled_pts_3d[:, 0] * fx / sampled_pts_3d[:, 2] + cx
            v_proj = sampled_pts_3d[:, 1] * fy / sampled_pts_3d[:, 2] + cy
            
            sampled_pts_2d = []
            for u, v in zip(u_proj, v_proj):
                sampled_pts_2d.append((int(round(u)), int(round(v))))
            return sampled_pts_2d
        except Exception as e:
            print(f"Error in 3D point generation: {e}. Falling back to 2D random sampling.")
            
    # Fallback: standard 2D sampling
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0:
        return []
        
    indices = list(range(len(x_indices)))
    random.shuffle(indices)
    
    pts = []
    for idx in indices[:num_points]:
        pts.append((int(x_indices[idx]), int(y_indices[idx])))
    return pts


class TrackingThread(QThread):
    """Background thread that runs the SpatialTrackerV2 tracking Predictor model on CUDA."""
    progress = pyqtSignal(str) # Status messages
    finished = pyqtSignal(bool, dict, str) # (success, results_dict, error_msg)

    def __init__(self, video_path, start_frame, end_frame, step, npz_preprocess_path, points, vo_points=120, pts_map=None, fixed_cam=False, model_name="Yuxihenry/SpatialTrackerV2-Online", s_wind=30, overlap=10):
        super().__init__()
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.step = step
        self.npz_preprocess_path = npz_preprocess_path
        self.points = points  # List of (x, y) in original HD resolution
        self.vo_points = vo_points
        self.pts_map = pts_map or []
        self.fixed_cam = fixed_cam
        self.model_name = model_name
        self.s_wind = s_wind
        self.overlap = overlap

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
                width=336,
                height=336,
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
            extrs_in_c2w = np.linalg.inv(extrs_in).astype(np.float32)
            #extrs_in_c2w = extrs_in.astype(np.float32)
            unc_metric_in = preprocess_data["unc_metric"]

            self.progress.emit("Interpolating depth and camera parameters to matched resolution...")
            # Resize depth and uncertainty to match the resized video tensor
            depth_tensor = F.interpolate(torch.from_numpy(depth_in)[:, None], 
                                         size=video_tensor.shape[2:], 
                                         mode='bilinear', align_corners=True).squeeze(1).numpy()
            unc_metric = F.interpolate(torch.from_numpy(unc_metric_in)[:, None].float(), 
                                        size=video_tensor.shape[2:], 
                                        mode='bilinear', align_corners=True).squeeze(1).numpy() > 0.5

            # Scale camera intrinsics based on the downscaled preprocessed depth shape (backward-compatible)
            scale_w = video_tensor.shape[3] / depth_in.shape[2]
            scale_h = video_tensor.shape[2] / depth_in.shape[1]
            intrs_in[:, 0, :] *= scale_w
            intrs_in[:, 1, :] *= scale_h

            # Scale query points from HD original to preprocessed resized resolution
            self.progress.emit("Preparing tracking query coordinates...")
            scale_w_query = video_tensor.shape[3] / W_orig
            scale_h_query = video_tensor.shape[2] / H_orig
            query_xyt = np.zeros((len(self.points), 3), dtype=np.float32)
            for idx, (x_hd, y_hd) in enumerate(self.points):
                x_proc = x_hd * scale_w_query
                y_proc = y_hd * scale_h_query
                query_xyt[idx, 0] = 0.0 # Queries correspond to frame 0 of sequence
                query_xyt[idx, 1] = x_proc
                query_xyt[idx, 2] = y_proc

            self.progress.emit(f"Loading Predictor model ({self.model_name}) to CUDA...")
            model = Predictor.from_pretrained(self.model_name)
            model.spatrack.track_num = self.vo_points
            model.S_wind = self.s_wind
            model.overlap = self.overlap
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
                                        intrs=intrs_in, extrs=extrs_in_c2w, 
                                        queries=query_xyt,
                                        fps=1, full_point=False, iters_track=8,
                                        query_no_BA=True, fixed_cam=self.fixed_cam, stage=1, unc_metric=unc_metric,
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
            results["confs"] = conf_pred.cpu().numpy()
            
            # Save tracker-refined depths, uncertainties, and resized video in downscaled tracking resolution
            refined_depth = point_map[:, 2, ...].cpu().numpy()
            conf_depth_np = conf_depth.cpu().numpy()
            results["depths"] = np.where(conf_depth_np > 0.5, refined_depth, depth_tensor)
            results["unc_metric"] = np.where(conf_depth_np > 0.5, conf_depth_np, unc_metric)
            
            #results["depths"] = point_map[:, 2, ...].cpu().numpy()
            #results["unc_metric"] = conf_depth.cpu().numpy()
            #results["video"] = (video / 255.0).cpu().numpy()

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
        self.last_saved_npz = ""
        self.video_width = 0
        self.video_height = 0
        
        self.overlay_items = []
        self.thread = None
        
        # MobileSAM state properties
        self.sam_points = []
        self.sam_labels = []
        self.sam_box = None
        self.sam_interact_mode = "manual"
        
        self._init_ui()
        
        # Connect to main window view click signals
        self.main_window.video_view.pixelClicked.connect(self._on_view_clicked)
        self.main_window.video_view.boxSelected.connect(self._on_box_selected)

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
        
        # MobileSAM segmenter group box (for premium look)
        sam_group = QGroupBox("MobileSAM Point Generator")
        sam_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #CCC; border-radius: 6px; margin-top: 10px; padding-top: 15px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        sam_layout = QVBoxLayout()
        
        sam_layout.addWidget(QLabel("<b>Mouse Click Mode:</b>"))
        self.sam_radio_group = QButtonGroup(self)
        
        self.radio_manual = QRadioButton("Standard Click (Add/Remove tracking points)")
        self.radio_manual.setChecked(True)
        self.radio_sam_pos = QRadioButton("SAM Prompt: Positive Point (+ green)")
        self.radio_sam_neg = QRadioButton("SAM Prompt: Negative Point (- red)")
        self.radio_sam_box = QRadioButton("SAM Prompt: Bounding Box [drag blue]")
        
        self.sam_radio_group.addButton(self.radio_manual)
        self.sam_radio_group.addButton(self.radio_sam_pos)
        self.sam_radio_group.addButton(self.radio_sam_neg)
        self.sam_radio_group.addButton(self.radio_sam_box)
        self.sam_radio_group.buttonClicked.connect(self._on_sam_mode_changed)
        
        sam_layout.addWidget(self.radio_manual)
        sam_layout.addWidget(self.radio_sam_pos)
        sam_layout.addWidget(self.radio_sam_neg)
        sam_layout.addWidget(self.radio_sam_box)
        
        # Points to Generate
        gen_pts_layout = QHBoxLayout()
        gen_pts_layout.addWidget(QLabel("Points to Generate:"))
        self.sam_pts_count_spin = QSpinBox()
        self.sam_pts_count_spin.setRange(5, 500)
        self.sam_pts_count_spin.setValue(50)
        gen_pts_layout.addWidget(self.sam_pts_count_spin)
        sam_layout.addLayout(gen_pts_layout)
        
        # Depth Confidence Threshold (default 0.3)
        gen_conf_layout = QHBoxLayout()
        gen_conf_layout.addWidget(QLabel("SAM Depth Conf Thresh:"))
        self.sam_conf_spin = QDoubleSpinBox()
        self.sam_conf_spin.setRange(0.01, 0.99)
        self.sam_conf_spin.setSingleStep(0.05)
        self.sam_conf_spin.setValue(0.30)
        gen_conf_layout.addWidget(self.sam_conf_spin)
        sam_layout.addLayout(gen_conf_layout)
        
        # Clear existing checkbox
        self.sam_clear_existing_cb = QCheckBox("Clear existing points before generating")
        self.sam_clear_existing_cb.setChecked(True)
        sam_layout.addWidget(self.sam_clear_existing_cb)
        
        # Reset and Run buttons
        sam_buttons_layout = QHBoxLayout()
        self.btn_reset_sam = QPushButton("Reset SAM")
        self.btn_reset_sam.setStyleSheet("background-color: #E1E1E1; font-weight: bold; height: 26px;")
        self.btn_reset_sam.clicked.connect(self._on_reset_sam_prompts)
        
        self.btn_run_sam = QPushButton("Run SAM")
        self.btn_run_sam.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; height: 26px;")
        self.btn_run_sam.clicked.connect(self._run_sam_segmentation)
        
        sam_buttons_layout.addWidget(self.btn_reset_sam)
        sam_buttons_layout.addWidget(self.btn_run_sam)
        sam_layout.addLayout(sam_buttons_layout)
        
        sam_group.setLayout(sam_layout)
        layout.addWidget(sam_group)
        
        layout.addWidget(QLabel("----------------------------------------"))
        
        # 3. Model Configuration & Running Section
        lbl_model_title = QLabel("<b>3. Model Tracking</b>")
        layout.addWidget(lbl_model_title)
        
        # Model Selection
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
        
        # Window Size (S_wind)
        swind_layout = QHBoxLayout()
        swind_layout.addWidget(QLabel("Window Size (S_wind):"))
        self.spin_swind = QSpinBox()
        self.spin_swind.setRange(5, 1000)
        self.spin_swind.setValue(30)
        swind_layout.addWidget(self.spin_swind)
        layout.addLayout(swind_layout)
        
        # Overlap Size
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
        self.video_path = getattr(self.main_window, 'current_video_path', '')
        # Try to automatically pull preprocessed path from PreprocessTool if loaded
        self._try_auto_load_preprocess()
        
        # Reset SAM modes to manual tracking by default
        self.radio_manual.setChecked(True)
        self.sam_interact_mode = "manual"
        
        # Disable panning scroll hand drag, and use crosshair cursor for point tool
        if hasattr(self.main_window, 'video_view') and self.main_window.video_view:
            self.main_window.video_view.set_box_selection_mode(False)
            self.main_window.video_view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.main_window.video_view.viewport().setCursor(Qt.CursorShape.CrossCursor)
            
        self.update_overlay()

    def hideEvent(self, event):
        super().hideEvent(event)
        # Restore standard scroll hand drag and arrow cursor
        if hasattr(self.main_window, 'video_view') and self.main_window.video_view:
            self.main_window.video_view.set_box_selection_mode(False)
            self.main_window.video_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.main_window.video_view.viewport().unsetCursor()
        
        # Clear overlay items when tool is hidden
        for item in self.overlay_items:
            try:
                self.main_window.video_view.scene.removeItem(item)
            except Exception:
                pass
        self.overlay_items.clear()

    def _on_model_changed(self, model_name):
        if model_name == "SpatialTrackerV2-Online":
            self.spin_swind.setValue(30)
            self.spin_overlap.setValue(10)
        else: # Offline
            self.spin_swind.setValue(500)
            self.spin_overlap.setValue(4)

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
        self.btn_gen_blender.setEnabled(False)
        self.btn_gen_blender.setStyleSheet("background-color: #E1E1E1; font-weight: bold; padding: 4px;")
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
        self.btn_gen_blender.setEnabled(False)
        self.btn_gen_blender.setStyleSheet("background-color: #E1E1E1; font-weight: bold; padding: 4px;")
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

        if self.sam_interact_mode == "sam_pos":
            # Add positive SAM prompt
            if (x, y) not in self.sam_points:
                self.sam_points.append((x, y))
                self.sam_labels.append(1)
                self.lbl_status.setText(f"SAM: Added Positive Point at ({x}, {y})")
        elif self.sam_interact_mode == "sam_neg":
            # Add negative SAM prompt
            if (x, y) not in self.sam_points:
                self.sam_points.append((x, y))
                self.sam_labels.append(0)
                self.lbl_status.setText(f"SAM: Added Negative Point at ({x}, {y})")
        elif self.sam_interact_mode == "sam_box":
            pass
        else:
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
                        self.btn_gen_blender.setEnabled(False)
                        self.btn_gen_blender.setStyleSheet("background-color: #E1E1E1; font-weight: bold; padding: 4px;")

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
        from gui_preprocess_loader import load_preprocess_metadata
        try:
            meta, filename = load_preprocess_metadata(self.main_window, self)
            if not meta:
                return
                
            npz_path = meta.get("npz_path") or meta.get("preprocess_npz")
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
        self.btn_save_result.setEnabled(False)
        self.btn_gen_blender.setEnabled(False)
        self.btn_gen_blender.setStyleSheet("background-color: #E1E1E1; font-weight: bold; padding: 4px;")
        self.lbl_status.setText("Status: Preparing tracking background thread...")
        self.lbl_status.setStyleSheet("color: #0078D7; font-weight: bold;")

        pts_map = getattr(self.main_window.decoder_thread, 'pts_map', [])
        
        model_type = self.combo_model.currentText()
        model_name = f"Yuxihenry/{model_type}"
        s_wind = self.spin_swind.value()
        overlap = min(self.spin_overlap.value(), s_wind - 1)

        self.thread = TrackingThread(
            video_path=self.video_path,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            step=self.step,
            npz_preprocess_path=self.preprocess_npz,
            points=self.points,
            vo_points=self.spin_vo.value(),
            pts_map=pts_map,
            fixed_cam=self.cb_fixed_cam.isChecked(),
            model_name=model_name,
            s_wind=s_wind,
            overlap=overlap
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
                    
        # 3. Draw MobileSAM prompt overlays on the first frame
        if current_frame == self.start_frame:
            # Draw SAM bounding box
            if self.sam_box is not None:
                x1, y1, x2, y2 = self.sam_box
                rect_item = QGraphicsRectItem(x1, y1, x2 - x1, y2 - y1)
                rect_item.setPen(QPen(QColor(0, 150, 255), 2, Qt.PenStyle.DashLine))
                rect_item.setBrush(QBrush(QColor(0, 150, 255, 30)))
                self.main_window.video_view.scene.addItem(rect_item)
                self.overlay_items.append(rect_item)
                
            # Draw SAM point prompts (green = positive, red = negative)
            for pt_idx, (x, y) in enumerate(self.sam_points):
                label = self.sam_labels[pt_idx]
                marker = QGraphicsEllipseItem(x - 6, y - 6, 12, 12)
                if label == 1:
                    marker.setBrush(QBrush(QColor(0, 255, 0))) # green
                else:
                    marker.setBrush(QBrush(QColor(255, 0, 0))) # red
                marker.setPen(QPen(QColor(0, 0, 0), 1.5))
                self.main_window.video_view.scene.addItem(marker)
                self.overlay_items.append(marker)
                
                # Draw +/- text
                text = QGraphicsSimpleTextItem("+" if label == 1 else "-")
                text.setBrush(QBrush(QColor(0, 0, 0)))
                text.setPos(x - 3.5, y - 7.5)
                font = text.font()
                font.setBold(True)
                font.setPointSize(9)
                text.setFont(font)
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
            meta, filename = load_tracking_result(self.main_window, self)
            if not meta:
                return
                
            result_npz = meta.get("result_npz")
            preprocess_npz = meta.get("preprocess_npz")
            preprocess_json = meta.get("preprocess_json")
            
            # Load preprocess paths
            self.preprocess_npz = preprocess_npz
            self.preprocess_json = preprocess_json
            
            # Load sequence limits
            self.start_frame = meta.get("start_frame", 0)
            self.end_frame = meta.get("end_frame", 0)
            self.step = meta.get("step", 1)
            self.points = meta.get("points", [])
            self.spin_vo.setValue(meta.get("vo_points", 120))
            self.video_width = meta.get("video_width", 0)
            self.video_height = meta.get("video_height", 0)
            
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
            
            # Navigate to starting frame to show overlays instantly
            self.main_window.decoder_thread.seek_frame(self.start_frame)
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
        from PyQt6.QtWidgets import QProgressDialog
        from PyQt6.QtCore import Qt
        progress = QProgressDialog("Decoding video frames for point cloud...", "Cancel", 0, T + 10, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.processEvents()
        
        import av
        try:
            container = av.open(self.video_path, container_options={'ignore_editlist': '1'})
            video_stream = container.streams.video[0]
            video_stream.thread_type = "AUTO"
            pts_map = getattr(self.main_window.decoder_thread, 'pts_map', [])
            if pts_map and self.start_frame < len(pts_map):
                container.seek(pts_map[self.start_frame], stream=video_stream, backward=True)
            needed_frame_indices = set(self.start_frame + t * self.step for t in range(T))
            decoded_frames = {}
            count = 0
            for frame in container.decode(video=0):
                if progress.wasCanceled():
                    container.close()
                    return False
                if frame.pts is None:
                    continue
                try:
                    idx = pts_map.index(frame.pts) if pts_map else count
                except ValueError:
                    continue
                if idx < self.start_frame:
                    if not pts_map:
                        count += 1
                    continue
                if idx > self.end_frame:
                    break
                if (idx - self.start_frame) % self.step == 0:
                    decoded_frames[idx] = frame.to_rgb().to_ndarray()
                    progress.setValue(int(len(decoded_frames) * T / len(needed_frame_indices)))
                    QApplication.processEvents()
                    if len(decoded_frames) == len(needed_frame_indices):
                        break
                if not pts_map:
                    count += 1
            container.close()
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
            # Result needs to be saved first so we can reference its absolute path
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
                    # User cancelled the save dialog
                    return
            else:
                return
                
        # Resolve default PC path
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
            # User wants to save a point cloud
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
                
        # Resolve dimensions
        w = getattr(self, 'video_width', 0)
        h = getattr(self, 'video_height', 0)
        if w <= 0 or h <= 0:
            # Query PyAV fallback
            w, h = get_video_dimensions(self.video_path)
            self.video_width = w
            self.video_height = h
            
        # Generate script code
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
            
            # Copy to clipboard
            clipboard = QApplication.clipboard()
            clipboard.setText(script_code)
            
            # Non-disruptive button styling feedback
            old_style = self.btn_gen_blender.styleSheet()
            old_text = self.btn_gen_blender.text()
            
            self.btn_gen_blender.setText("Copied to Clipboard!")
            self.btn_gen_blender.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 4px;")
            self.lbl_status.setText("Status: Blender import script copied directly to clipboard!")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
            # Restore button after 1.5 seconds
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(1500, lambda: self.btn_gen_blender.setText(old_text))
            QTimer.singleShot(1500, lambda: self.btn_gen_blender.setStyleSheet(old_style))
            
        except Exception as e:
            QMessageBox.critical(self, "Generation Failed", f"Failed to generate Blender script:\n{e}")

    def _on_box_selected(self, x1, y1, x2, y2):
        """Triggered from main window VideoGraphicsView when box selection is completed."""
        if not self.video_path:
            return
            
        current_frame = self.main_window.timeline_slider.value()
        if current_frame != self.start_frame:
            self.lbl_status.setText(f"Status: Jumped to first frame {self.start_frame} to manage points.")
            self.main_window.decoder_thread.seek_frame(self.start_frame)
            return
            
        if self.sam_interact_mode == "sam_box":
            self.sam_box = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
            self.lbl_status.setText(f"SAM: Set Bounding Box Prompt at {self.sam_box}")
            self.update_overlay()

    def _on_sam_mode_changed(self, button):
        """Toggles the appropriate click/drag behaviors depending on SAM radio mode selection."""
        if button == self.radio_manual:
            self.sam_interact_mode = "manual"
            if hasattr(self.main_window, 'video_view') and self.main_window.video_view:
                self.main_window.video_view.set_box_selection_mode(False)
                self.main_window.video_view.setDragMode(QGraphicsView.DragMode.NoDrag)
                self.main_window.video_view.viewport().setCursor(Qt.CursorShape.CrossCursor)
            self.lbl_status.setText("Mode: Manual click to add/remove tracking points.")
        elif button == self.radio_sam_pos:
            self.sam_interact_mode = "sam_pos"
            if hasattr(self.main_window, 'video_view') and self.main_window.video_view:
                self.main_window.video_view.set_box_selection_mode(False)
                self.main_window.video_view.setDragMode(QGraphicsView.DragMode.NoDrag)
                self.main_window.video_view.viewport().setCursor(Qt.CursorShape.CrossCursor)
            self.lbl_status.setText("Mode: Click to add positive/foreground SAM point prompts (+).")
        elif button == self.radio_sam_neg:
            self.sam_interact_mode = "sam_neg"
            if hasattr(self.main_window, 'video_view') and self.main_window.video_view:
                self.main_window.video_view.set_box_selection_mode(False)
                self.main_window.video_view.setDragMode(QGraphicsView.DragMode.NoDrag)
                self.main_window.video_view.viewport().setCursor(Qt.CursorShape.CrossCursor)
            self.lbl_status.setText("Mode: Click to add negative/background SAM point prompts (-).")
        elif button == self.radio_sam_box:
            self.sam_interact_mode = "sam_box"
            if hasattr(self.main_window, 'video_view') and self.main_window.video_view:
                self.main_window.video_view.set_box_selection_mode(True)
            self.lbl_status.setText("Mode: Click and drag to define the SAM bounding box prompt.")

    def _on_reset_sam_prompts(self):
        """Clears current positive/negative points and box prompts for MobileSAM."""
        self.sam_points.clear()
        self.sam_labels.clear()
        self.sam_box = None
        self.lbl_status.setText("SAM: Reset all positive/negative and bounding box prompts.")
        self.update_overlay()

    def _load_mobile_sam(self):
        """Loads MobileSAM model and predictors to CUDA/CPU cleanly."""
        import torch
        from mobile_sam import sam_model_registry, SamPredictor
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam_checkpoint = "mobile_sam.pt"
        model_type = "vit_t"
        
        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam = mobile_sam.to(device=device)
        mobile_sam.eval()
        
        predictor = SamPredictor(mobile_sam)
        return mobile_sam, predictor

    def _unload_mobile_sam(self, mobile_sam, predictor):
        """Cleans up MobileSAM model variables and clears PyTorch CUDA memory cache completely."""
        import torch
        import gc
        del predictor
        del mobile_sam
        torch.cuda.empty_cache()
        gc.collect()

    def _decode_first_frame(self):
        """Decodes the exact frame at self.start_frame as an RGB NumPy array."""
        import av
        if not self.video_path:
            return None
            
        container = av.open(self.video_path, container_options={'ignore_editlist': '1'})
        video_stream = container.streams.video[0]
        video_stream.thread_type = "AUTO"
        
        # Build PTS map
        pts_set = set()
        for f in container.decode(video=0):
            if f.pts is not None:
                pts_set.add(f.pts)
        pts_map = sorted(list(pts_set))
        
        if not pts_map or self.start_frame >= len(pts_map):
            container.close()
            return None
            
        target_pts = pts_map[self.start_frame]
        container.seek(target_pts, stream=video_stream, backward=True)
        
        target_frame = None
        for frame in container.decode(video=0):
            if frame.pts >= target_pts:
                target_frame = frame
                break
                
        img_np = None
        if target_frame:
            img_np = target_frame.to_rgb().to_ndarray()
            
        container.close()
        return img_np

    def _run_sam_segmentation(self):
        """Invokes MobileSAM to segment the object and sample point clusters inside the mask."""
        if not self.video_path:
            self.lbl_status.setText("Status: Error - No video loaded!")
            return
            
        if not self.sam_points and self.sam_box is None:
            self.lbl_status.setText("Status: Error - No SAM point or bounding box prompt provided!")
            return
            
        # Temporarily disable buttons
        self.btn_run_sam.setEnabled(False)
        self.btn_process.setEnabled(False)
        self.lbl_status.setText("SAM: Decoding start frame...")
        self.lbl_status.setStyleSheet("color: #0078D7; font-weight: bold;")
        
        # Process events to show status instantly
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        
        img_np = self._decode_first_frame()
        if img_np is None:
            self.lbl_status.setText("Status: Error - Decoding start frame failed!")
            self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")
            self.btn_run_sam.setEnabled(True)
            self.btn_process.setEnabled(True)
            return
            
        self.lbl_status.setText("SAM: Loading MobileSAM model...")
        QApplication.processEvents()
        
        try:
            mobile_sam, predictor = self._load_mobile_sam()
            
            self.lbl_status.setText("SAM: Predicting segment mask...")
            QApplication.processEvents()
            
            predictor.set_image(img_np)
            
            point_coords = np.array(self.sam_points) if self.sam_points else None
            point_labels = np.array(self.sam_labels) if self.sam_labels else None
            
            # Format box [x_min, y_min, x_max, y_max]
            box = np.array(self.sam_box) if self.sam_box is not None else None
            
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True
            )
            
            # Choose highest scoring mask
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx] # boolean array
            
            # Try to load preprocessed camera geometry and depth parameters for the first frame
            depth_map_3d = None
            K_3d = None
            conf_map_3d = None
            is_3d_active = False
            
            if self.preprocess_npz and os.path.exists(self.preprocess_npz):
                try:
                    self.lbl_status.setText("SAM: Loading preprocessed depth/camera parameters...")
                    QApplication.processEvents()
                    preprocess_data = np.load(self.preprocess_npz, allow_pickle=True)
                    depths = preprocess_data["depths"]
                    intrinsics = preprocess_data["intrinsics"]
                    unc_metrics = preprocess_data["unc_metric"]
                    
                    if len(depths) > 0 and len(intrinsics) > 0 and len(unc_metrics) > 0:
                        depth_map_3d = depths[0]
                        K_3d = intrinsics[0]
                        conf_map_3d = unc_metrics[0]
                        is_3d_active = True
                except Exception as e:
                    print(f"Failed to load preprocess npz for 3D sampling: {e}")
            
            if is_3d_active:
                self.lbl_status.setText("SAM: Sampling points using 3D Farthest Point Sampling...")
            else:
                self.lbl_status.setText("SAM: No 3D data loaded. Sampling points using standard 2D...")
            QApplication.processEvents()
            
            num_pts_to_gen = self.sam_pts_count_spin.value()
            conf_thresh = self.sam_conf_spin.value()
            sampled_pts = generate_points_in_mask(
                mask=best_mask,
                num_points=num_pts_to_gen,
                depth=depth_map_3d,
                K=K_3d,
                conf=conf_map_3d,
                conf_threshold=conf_thresh
            )
            
            if self.sam_clear_existing_cb.isChecked():
                self.points.clear()
                
            for pt in sampled_pts:
                if pt not in self.points:
                    self.points.append(pt)
                    
            self._unload_mobile_sam(mobile_sam, predictor)
            
            self.lbl_pts_count.setText(f"Selected Points: {len(self.points)}")
            if is_3d_active:
                self.lbl_status.setText(f"SAM success: Generated {len(sampled_pts)} points using 3D FPS (VRAM cleared)!")
            else:
                self.lbl_status.setText(f"SAM success: Generated {len(sampled_pts)} points using 2D fallback (VRAM cleared)!")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
            # Reset SAM prompts and switch interaction back to manual standard clicks
            self.sam_points.clear()
            self.sam_labels.clear()
            self.sam_box = None
            self.radio_manual.setChecked(True)
            self._on_sam_mode_changed(self.radio_manual)
            
        except Exception as e:
            self.lbl_status.setText(f"SAM Error: {e}")
            self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")
            if 'mobile_sam' in locals() and 'predictor' in locals():
                self._unload_mobile_sam(mobile_sam, predictor)
                
        self.btn_run_sam.setEnabled(True)
        self.btn_process.setEnabled(True)
        self.update_overlay()
