import os
import gc
import json
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import av
import numpy as np
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QWidget
from PyQt6.QtCore import QThread, pyqtSignal

from gui_tool_base import BaseTool
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

class PreprocessingThread(QThread):
    """Background thread that loads the model, decodes video frames, runs inference, and unloads the model."""
    progress = pyqtSignal(int, int) # (decoded_frames, total_frames_to_decode)
    status = pyqtSignal(str) # Status messages
    finished = pyqtSignal(bool, str) # (success, saved_file_path or error_message)

    def __init__(self, video_path, start_frame, end_frame, step, npz_path, json_path, pts_map=None):
        super().__init__()
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.step = step
        self.npz_path = npz_path
        self.json_path = json_path
        self.pts_map = pts_map or []

    def run(self):
        try:
            self.status.emit("Opening video with PyAV...")
            container = av.open(self.video_path, container_options={'ignore_editlist': '1'})
            video_stream = container.streams.video[0]
            video_stream.thread_type = "AUTO" # Enable multi-threaded decoding
            H_orig, W_orig = video_stream.height, video_stream.width
            
            # Determine number of frames to decode
            total_frames_to_decode = 0
            for i in range(self.start_frame, self.end_frame + 1):
                if (i - self.start_frame) % self.step == 0:
                    total_frames_to_decode += 1
            
            if total_frames_to_decode == 0:
                raise ValueError("No frames found within the active trim range for the selected step size.")
                
            # Perform keyframe seek for fast speed if we have a pts_map
            if self.pts_map and self.start_frame < len(self.pts_map):
                target_pts = self.pts_map[self.start_frame]
                container.seek(target_pts, stream=video_stream, backward=True)
                self.status.emit("Performing fast seek to trim start...")
                
            self.status.emit(f"Decoding and preprocessing {total_frames_to_decode} frames sequentially...")
            frames = []
            count = 0
            for frame in container.decode(video=0):
                if frame.pts is None:
                    continue
                    
                # Match PTS to get exact frame index
                try:
                    idx = self.pts_map.index(frame.pts) if self.pts_map else count
                except ValueError:
                    continue
                    
                if idx < self.start_frame:
                    continue
                if idx > self.end_frame:
                    break
                    
                if (idx - self.start_frame) % self.step == 0:
                    img = frame.to_ndarray(format='rgb24') # Optimized C conversion (extremely fast!)
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
                    img_proc = preprocess_image(img_tensor)
                    frames.append(img_proc)
                    count += 1
                    self.progress.emit(count, total_frames_to_decode)
                    if count >= total_frames_to_decode:
                        break
            
            container.close()
            
            if len(frames) == 0:
                raise ValueError("No frames were successfully decoded in the trim range.")
                
            video_tensor = torch.stack(frames)
            H_proc, W_proc = video_tensor.shape[-2:]
            
            # Determine precision based on hardware capability
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.status.emit(f"Loading VGGT4Track model to GPU (precision: {dtype})...")
            
            vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
            vggt4track_model.eval()
            vggt4track_model = vggt4track_model.to("cuda")
            
            self.status.emit("Running model inference...")
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=dtype):
                        # Add batch dimension and divide by 255.0 to normalize range [0, 1]
                        predictions = vggt4track_model(video_tensor[None].cuda() / 255.0)
                        extrinsic = predictions["poses_pred"]
                        intrinsic = predictions["intrs"]
                        depth_map = predictions["points_map"][..., 2]
                        depth_conf = predictions["unc_metric"]
                        
                        self.status.emit("Post-processing tensors back to original resolution...")
                        # Scale back depth and uncertainty to HD original dimensions
                        depth_tensor_hd = F.interpolate(depth_map[:, None], size=(H_orig, W_orig), mode='bilinear', align_corners=True).squeeze(1)
                        unc_metric_hd = F.interpolate(depth_conf[:, None], size=(H_orig, W_orig), mode='bilinear', align_corners=True).squeeze(1)
                        
                        # Scale intrinsics back to original dimensions
                        intrs_hd = intrinsic.clone()
                        intrs_hd[..., 0, :] *= W_orig / W_proc
                        intrs_hd[..., 1, :] *= H_orig / H_proc
                        
                        # Squeeze out batch dimension safely to maintain sequence length T
                        extrs = extrinsic.squeeze(0).cpu().numpy()
                        intrs = intrs_hd.squeeze(0).cpu().numpy()
                        depth_tensor = depth_tensor_hd.cpu().numpy()
                        unc_metric = unc_metric_hd.cpu().numpy()
            
            self.status.emit("Saving processed data as NPZ...")
            np.savez(self.npz_path,
                     depths=depth_tensor,
                     extrinsics=extrs,
                     intrinsics=intrs,
                     unc_metric=unc_metric)
                     
            self.status.emit("Saving metadata as JSON...")
            metadata = {
                "video_path": os.path.abspath(self.video_path),
                "start_frame": self.start_frame,
                "end_frame": self.end_frame,
                "step": self.step,
                "npz_path": os.path.abspath(self.npz_path)
            }
            with open(self.json_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            self.status.emit("Unloading model and cleaning GPU cache...")
            del vggt4track_model
            torch.cuda.empty_cache()
            gc.collect()
            
            self.finished.emit(True, self.npz_path)
            
        except Exception as e:
            # Clean up model reference and cache even on failures to prevent leak
            if 'vggt4track_model' in locals():
                del vggt4track_model
            torch.cuda.empty_cache()
            gc.collect()
            self.finished.emit(False, str(e))


class PreprocessTool(BaseTool):
    """Tool to configure, run, and load VGGT4Track preprocessing for SpatialTrackerV2."""
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.video_path = ""
        self.thread = None
        self._init_ui()

    def get_name(self):
        return "Video Preprocessing"

    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Info labels
        self.lbl_info = QLabel("<b>Video:</b> No video loaded<br><b>Trim Range:</b> N/A")
        self.lbl_info.setWordWrap(True)
        layout.addWidget(self.lbl_info)
        
        # Step Size Input
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step Size:"))
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 100)
        self.spin_step.setValue(1)
        self.spin_step.valueChanged.connect(self._update_paths_label)
        step_layout.addWidget(self.spin_step)
        layout.addLayout(step_layout)
        
        # Paths display label
        self.lbl_paths = QLabel("<b>Outputs:</b> Auto-named upon Run")
        self.lbl_paths.setWordWrap(True)
        layout.addWidget(self.lbl_paths)
        
        # Run Button
        self.btn_run = QPushButton("Run Preprocessing [run]")
        self.btn_run.setStyleSheet("background-color: #0078D7; color: white; font-weight: bold; padding: 6px;")
        self.btn_run.clicked.connect(self._on_run)
        layout.addWidget(self.btn_run)
        
        # Load Button
        self.btn_load = QPushButton("Load Metadata [load]")
        self.btn_load.setStyleSheet("background-color: #E1E1E1; color: black; font-weight: bold; padding: 6px;")
        self.btn_load.clicked.connect(self._on_load)
        layout.addWidget(self.btn_load)
        
        # Status Label
        self.lbl_status = QLabel("<b>Status:</b> Idle")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #555555;")
        layout.addWidget(self.lbl_status)
        
        layout.addStretch()

    def on_video_loaded(self, metadata):
        self.video_path = getattr(self.main_window, 'current_video_path', '')
        self._update_paths_label()

    def on_frame_changed(self, frame_idx, current_time_sec):
        # Dynamically update the visual trim bounds to reflect slider changes
        self._update_paths_label()

    def showEvent(self, event):
        super().showEvent(event)
        self.video_path = getattr(self.main_window, 'current_video_path', '')
        self._update_paths_label()

    def _get_current_trim_bounds(self):
        start = self.main_window.timeline_slider.minimum()
        end = self.main_window.timeline_slider.maximum()
        return start, end

    def _get_output_paths(self):
        if not self.video_path:
            return None, None
        
        start, end = self._get_current_trim_bounds()
        step = self.spin_step.value()
        
        video_dir = os.path.dirname(self.video_path)
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        
        npz_name = f"{base_name}_trim_{start}_{end}_step_{step}_intermediate.npz"
        json_name = f"{base_name}_trim_{start}_{end}_step_{step}_metadata.json"
        
        npz_path = os.path.join(video_dir, npz_name)
        json_path = os.path.join(video_dir, json_name)
        return npz_path, json_path

    def _update_paths_label(self):
        if not self.video_path:
            self.lbl_info.setText("<b>Video:</b> No video loaded<br><b>Trim Range:</b> N/A")
            self.lbl_paths.setText("<b>Outputs:</b> Auto-named upon Run")
            return
            
        start, end = self._get_current_trim_bounds()
        self.lbl_info.setText(f"<b>Video:</b> {os.path.basename(self.video_path)}<br><b>Trim Range:</b> {start} to {end}")
        
        npz_path, _ = self._get_output_paths()
        if npz_path:
            self.lbl_paths.setText(f"<b>Output NPZ:</b><br>{os.path.basename(npz_path)}")

    def _on_run(self):
        if not self.video_path:
            self.lbl_status.setText("<b>Status:</b> Error - No video loaded!")
            self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")
            return
            
        start, end = self._get_current_trim_bounds()
        step = self.spin_step.value()
        npz_path, json_path = self._get_output_paths()
        
        # Disable UI during background operation
        self.btn_run.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.spin_step.setEnabled(False)
        
        self.lbl_status.setStyleSheet("color: #0078D7; font-weight: bold;")
        self.lbl_status.setText("<b>Status:</b> Preparing thread...")
        
        # Initialize background processing
        pts_map = getattr(self.main_window.decoder_thread, 'pts_map', [])
        self.thread = PreprocessingThread(
            video_path=self.video_path,
            start_frame=start,
            end_frame=end,
            step=step,
            npz_path=npz_path,
            json_path=json_path,
            pts_map=pts_map
        )
        self.thread.progress.connect(self._on_thread_progress)
        self.thread.status.connect(self._on_thread_status)
        self.thread.finished.connect(self._on_thread_finished)
        self.thread.start()

    def _on_thread_progress(self, current, total):
        self.lbl_status.setText(f"<b>Status:</b> Decoding frame {current}/{total}...")

    def _on_thread_status(self, msg):
        self.lbl_status.setText(f"<b>Status:</b> {msg}")

    def _on_thread_finished(self, success, message):
        # Re-enable inputs
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.spin_step.setEnabled(True)
        
        if success:
            self.lbl_status.setText(f"<b>Status:</b> Success!<br>Saved to: {os.path.basename(message)}")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.lbl_status.setText(f"<b>Status:</b> Failed!<br>Error: {message}")
            self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")

    def _on_load(self):
        from PyQt6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Preprocessed Metadata", "", "JSON Files (*.json)"
        )
        if not filename:
            return
            
        try:
            with open(filename, 'r') as f:
                metadata = json.load(f)
                
            video_path = metadata.get("video_path")
            start_frame = metadata.get("start_frame")
            end_frame = metadata.get("end_frame")
            step = metadata.get("step", 1)
            npz_path = metadata.get("npz_path")
            
            if not os.path.exists(video_path):
                # Try relative resolution path based on metadata file's folder
                dir_meta = os.path.dirname(filename)
                rel_video_path = os.path.join(dir_meta, os.path.basename(video_path))
                if os.path.exists(rel_video_path):
                    video_path = rel_video_path
                else:
                    self.lbl_status.setText(f"<b>Status:</b> Error - Video not found at:<br>{video_path}")
                    self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")
                    return
            
            # Load video via main window
            success = self.main_window.load_video(video_path)
            if not success:
                self.lbl_status.setText("<b>Status:</b> Error - Failed to load video.")
                self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")
                return
                
            # Apply trim restriction and sync TrimTool fields
            from gui_tool_trim import TrimTool
            for tool in self.main_window.tools:
                if isinstance(tool, TrimTool):
                    tool.start_frame = start_frame
                    tool.end_frame = end_frame
                    tool._update_labels()
                    self.main_window.apply_timeline_restriction(start_frame, end_frame)
                    break
            
            self.spin_step.setValue(step)
            self._update_paths_label()
            
            self.lbl_status.setText(f"<b>Status:</b> Loaded successfully!<br>NPZ: {os.path.basename(npz_path)}")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
        except Exception as e:
            self.lbl_status.setText(f"<b>Status:</b> Load failed!<br>Error: {str(e)}")
            self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")
