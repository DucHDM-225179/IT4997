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
            
            self.status.emit("Running model inference (chunked/sliding-window)...")
            from models.SpaTrackV2.models.utils import matrix_to_quaternion, quaternion_to_matrix

            chunk_size = 8
            overlap = 4
            num_frames = video_tensor.shape[0]

            final_depths = torch.zeros((num_frames, H_orig, W_orig), device="cpu", dtype=torch.float32)
            final_uncs = torch.zeros((num_frames, H_orig, W_orig), device="cpu", dtype=torch.float32)
            final_intrs = torch.zeros((num_frames, 3, 3), device="cpu", dtype=torch.float32)
            final_extrs = torch.zeros((num_frames, 4, 4), device="cpu", dtype=torch.float32)

            start_idx = 0
            is_first_chunk = True

            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                while start_idx < num_frames:
                    end_idx = min(start_idx + chunk_size, num_frames)
                    if not is_first_chunk:
                        chunk_start = start_idx - overlap
                    else:
                        chunk_start = start_idx
                        
                    chunk_video = video_tensor[chunk_start:end_idx]
                    self.status.emit(f"Running model inference: frame {end_idx}/{num_frames}...")
                    
                    with torch.no_grad():
                        with torch.amp.autocast(device_type="cuda", dtype=dtype):
                            predictions = vggt4track_model(chunk_video[None].cuda() / 255.0)
                            
                            extrinsic = predictions["poses_pred"][0].cuda()  # [W_s, 4, 4]
                            intrinsic = predictions["intrs"][0]       # [W_s, 3, 3]
                            depth_map = predictions["points_map"][..., 2]  # [W_s, H, W]
                            depth_conf = predictions["unc_metric"]         # [W_s, H, W]
                            
                            # Post-processing: Resize back to original resolution
                            H_proc, W_proc = chunk_video.shape[-2:]
                            depth_tensor_hd = F.interpolate(depth_map[:, None], size=(H_orig, W_orig), mode='bilinear', align_corners=True).squeeze(1)
                            unc_metric_hd = F.interpolate(depth_conf[:, None], size=(H_orig, W_orig), mode='bilinear', align_corners=True).squeeze(1)
                            
                            # Scale intrinsics back to original resolution
                            intrs_hd = intrinsic.clone()
                            intrs_hd[..., 0, :] *= W_orig / W_proc
                            intrs_hd[..., 1, :] *= H_orig / H_proc
                            
                    extrinsic = extrinsic.float()
                    intrs_hd = intrs_hd.float()
                    depth_tensor_hd = depth_tensor_hd.float()
                    unc_metric_hd = unc_metric_hd.float()
                    
                    chunk_len = len(chunk_video)
                    
                    if is_first_chunk:
                        final_depths[0:chunk_len] = depth_tensor_hd.cpu()
                        final_uncs[0:chunk_len] = unc_metric_hd.cpu()
                        final_intrs[0:chunk_len] = intrs_hd.cpu()
                        final_extrs[0:chunk_len] = extrinsic.cpu()
                        is_first_chunk = False
                        start_idx += chunk_size
                    else:
                        # 1. Compute scale factor s from overlapping depths
                        D_prev = final_depths[chunk_start : chunk_start + overlap].cuda()
                        D_curr = depth_tensor_hd[:overlap]
                        
                        valid_mask = (D_prev > 0.1) & (D_prev < 100.0) & (D_curr > 0.1) & (D_curr < 100.0)
                        if valid_mask.sum() > 100:
                            X = D_curr[valid_mask]
                            Y = D_prev[valid_mask]
                            sum_xx = (X * X).sum()
                            if sum_xx > 1e-6:
                                s = (X * Y).sum() / sum_xx
                                s = torch.clamp(s, min=0.2, max=5.0)
                            else:
                                s = torch.tensor(1.0, device="cuda")
                        else:
                            s = torch.tensor(1.0, device="cuda")
                        
                        # 2. Scale the current chunk's depth and camera translations
                        depth_tensor_hd = s * depth_tensor_hd
                        extrinsic = extrinsic.clone()
                        extrinsic[:, :3, 3] = s * extrinsic[:, :3, 3]

                        M_trans = []
                        M_quats = []
                        
                        prev_global_extrs = final_extrs[chunk_start : chunk_start + overlap].cuda()
                        
                        for t in range(overlap):
                            G_t = prev_global_extrs[t]
                            C_t = extrinsic[t]
                            M_t = G_t @ torch.inverse(C_t)
                            M_trans.append(M_t[:3, 3])
                            q_t = matrix_to_quaternion(M_t[:3, :3])
                            M_quats.append(q_t)
                            
                        M_trans = torch.stack(M_trans, dim=0).mean(dim=0)
                        M_quats = torch.stack(M_quats, dim=0)
                        M_quats = torch.where(M_quats[:, 0:1] < 0, -M_quats, M_quats)
                        M_quat_avg = M_quats.mean(dim=0)
                        M_quat_avg = M_quat_avg / torch.norm(M_quat_avg).clamp(min=1e-8)
                        
                        M_rot = quaternion_to_matrix(M_quat_avg)
                        
                        M = torch.eye(4, device="cuda")
                        M[:3, :3] = M_rot
                        M[:3, 3] = M_trans
                        
                        aligned_extrinsic = M @ extrinsic
                        blend_weights = torch.linspace(0.0, 1.0, steps=overlap, device="cuda")
                        
                        blended_extrs = []
                        for t in range(overlap):
                            w = blend_weights[t]
                            g_prev = prev_global_extrs[t]
                            g_curr = aligned_extrinsic[t]
                            
                            t_blend = (1.0 - w) * g_prev[:3, 3] + w * g_curr[:3, 3]
                            
                            q_prev = matrix_to_quaternion(g_prev[:3, :3])
                            q_curr = matrix_to_quaternion(g_curr[:3, :3])
                            
                            if torch.dot(q_prev, q_curr) < 0:
                                q_curr = -q_curr
                                
                            q_blend = (1.0 - w) * q_prev + w * q_curr
                            q_blend = q_blend / torch.norm(q_blend).clamp(min=1e-8)
                            r_blend = quaternion_to_matrix(q_blend)
                            
                            g_blend = torch.eye(4, device="cuda")
                            g_blend[:3, :3] = r_blend
                            g_blend[:3, 3] = t_blend
                            blended_extrs.append(g_blend)
                            
                        blended_extrs = torch.stack(blended_extrs, dim=0)
                        
                        blend_w_3d = blend_weights.view(overlap, 1, 1).cpu()
                        prev_depths = final_depths[chunk_start : chunk_start + overlap]
                        prev_uncs = final_uncs[chunk_start : chunk_start + overlap]
                        prev_intrs = final_intrs[chunk_start : chunk_start + overlap]
                        
                        blended_depths = (1.0 - blend_w_3d) * prev_depths + blend_w_3d * depth_tensor_hd[:overlap].cpu()
                        blended_uncs = (1.0 - blend_w_3d) * prev_uncs + blend_w_3d * unc_metric_hd[:overlap].cpu()
                        blended_intrs = (1.0 - blend_weights.view(overlap, 1, 1).cpu()) * prev_intrs + blend_weights.view(overlap, 1, 1).cpu() * intrs_hd[:overlap].cpu()
                        
                        final_depths[chunk_start : chunk_start + overlap] = blended_depths
                        final_uncs[chunk_start : chunk_start + overlap] = blended_uncs
                        final_intrs[chunk_start : chunk_start + overlap] = blended_intrs
                        final_extrs[chunk_start : chunk_start + overlap] = blended_extrs.cpu()
                        
                        new_len = chunk_len - overlap
                        if new_len > 0:
                            final_depths[start_idx : start_idx + new_len] = depth_tensor_hd[overlap:].cpu()
                            final_uncs[start_idx : start_idx + new_len] = unc_metric_hd[overlap:].cpu()
                            final_intrs[start_idx : start_idx + new_len] = intrs_hd[overlap:].cpu()
                            final_extrs[start_idx : start_idx + new_len] = aligned_extrinsic[overlap:].cpu()
                            
                        start_idx += (chunk_size - overlap)

                    # Clear intermediate variables and clean CUDA cache to prevent VRAM accumulation
                    del predictions, extrinsic, intrinsic, depth_map, depth_conf
                    del depth_tensor_hd, unc_metric_hd, intrs_hd
                    if 'prev_global_extrs' in locals():
                        del prev_global_extrs, M_trans, M_quats, M_quat_avg, M_rot, M, aligned_extrinsic, blend_weights, blended_extrs, blend_w_3d, prev_depths, prev_uncs, prev_intrs, blended_depths, blended_uncs, blended_intrs
                    torch.cuda.empty_cache()

            extrs = final_extrs.numpy()
            intrs = final_intrs.numpy()
            depth_tensor = final_depths.numpy()
            unc_metric = final_uncs.numpy()
            
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
