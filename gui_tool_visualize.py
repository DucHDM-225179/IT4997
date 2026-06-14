import os
import json
import numpy as np
import cv2
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt6.QtGui import QImage
from gui_tool_base import BaseTool

class VisualizeVideoTool(BaseTool):
    """Tool to visualize preprocessed/processed depth maps with a jet colormap in the viewport."""
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        
        # Internal state
        self.depths = None
        self.unc_metric = None
        self.start_frame = 0
        self.end_frame = 0
        self.step = 1
        self.loaded_npz_path = ""
        self.video_path = ""
        
        self._init_ui()
        
    def get_name(self):
        return "Visualize Depth Map"
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("<b>Visualize Video (Depth Map)</b>")
        title.setStyleSheet("font-size: 14px; color: #2196F3; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Status Card Widget
        status_card = QWidget()
        status_card.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333; border-radius: 6px; padding: 10px;")
        status_layout = QVBoxLayout(status_card)
        
        # Status Label
        self.lbl_status = QLabel("Visualization Data: Not Loaded")
        self.lbl_status.setStyleSheet("color: #FF5252; font-weight: bold; font-size: 12px;")
        self.lbl_status.setWordWrap(True)
        status_layout.addWidget(self.lbl_status)
        
        # Details Label
        self.lbl_details = QLabel("No active preprocessed depth maps found. Please run preprocessing or tracking first, or load a session.")
        self.lbl_details.setStyleSheet("color: #AAAAAA; font-size: 11px;")
        self.lbl_details.setWordWrap(True)
        status_layout.addWidget(self.lbl_details)
        
        layout.addWidget(status_card)
        
        # Load Button
        self.btn_load = QPushButton("Load NPZ / Metadata JSON")
        self.btn_load.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 8px 12px;
                font-size: 12px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        self.btn_load.clicked.connect(self._on_manual_load)
        layout.addWidget(self.btn_load)
        
        layout.addStretch()
        
    def showEvent(self, event):
        super().showEvent(event)
        self.video_path = getattr(self.main_window, 'current_video_path', '')
        self.auto_detect_data()
        self.refresh_display()
        
    def hideEvent(self, event):
        super().hideEvent(event)
        # Restore the original RGB video frame when switching away
        if self.main_window.timeline_slider.isEnabled():
            frame_idx = self.main_window.timeline_slider.value()
            self.main_window.decoder_thread.seek_frame(frame_idx)
            
    def on_video_loaded(self, metadata):
        # Reset state on new video load
        self.depths = None
        self.unc_metric = None
        self.start_frame = 0
        self.end_frame = 0
        self.step = 1
        self.loaded_npz_path = ""
        self.video_path = metadata.get("video_path", "")
        
        # Try to auto-detect
        self.auto_detect_data()
        self.update_ui_state()
        
    def on_frame_changed(self, frame_idx, current_time_sec):
        # When active and playing, overwrite the viewport image with the depth map
        if self.depths is not None:
            if self.start_frame <= frame_idx <= self.end_frame:
                offset = frame_idx - self.start_frame
                idx_in_depths = int(round(offset / self.step))
                if 0 <= idx_in_depths < len(self.depths):
                    qimage_depth = self.get_depth_qimage(idx_in_depths)
                    if qimage_depth:
                        self.main_window.video_view.set_image(qimage_depth)
                        
    def refresh_display(self):
        if self.depths is not None and self.main_window.timeline_slider.isEnabled():
            frame_idx = self.main_window.timeline_slider.value()
            self.on_frame_changed(frame_idx, 0.0)
            
    def auto_detect_data(self):
        # 1. Search ProcessVideoTool for preprocess_npz
        from gui_tool_process import ProcessVideoTool
        process_tool = None
        for tool in self.main_window.tools:
            if isinstance(tool, ProcessVideoTool):
                process_tool = tool
                break
                
        if process_tool and process_tool.preprocess_npz and os.path.exists(process_tool.preprocess_npz):
            success = self.load_visualization_data(process_tool.preprocess_npz, process_tool.preprocess_json)
            if success:
                self.update_ui_state()
                return
                
        # 2. Search PreprocessTool output paths
        from gui_tool_preprocess import PreprocessTool
        preprocess_tool = None
        for tool in self.main_window.tools:
            if isinstance(tool, PreprocessTool):
                preprocess_tool = tool
                break
                
        if preprocess_tool:
            npz_path, json_path = preprocess_tool._get_output_paths()
            if npz_path and os.path.exists(npz_path):
                success = self.load_visualization_data(npz_path, json_path)
                if success:
                    self.update_ui_state()
                    return
                    
    def load_visualization_data(self, npz_path, json_path=None):
        if not npz_path or not os.path.exists(npz_path):
            return False
            
        # Try to resolve companion json path if not provided
        if not json_path:
            if npz_path.endswith("_intermediate.npz"):
                json_path = npz_path.replace("_intermediate.npz", "_metadata.json")
            elif npz_path.endswith("_result.npz"):
                json_path = npz_path.replace("_result.npz", "_result_metadata.json")
            else:
                json_path = os.path.splitext(npz_path)[0] + ".json"
                
        start_frame = 0
        end_frame = 0
        step = 1
        
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                start_frame = meta.get("start_frame", 0)
                end_frame = meta.get("end_frame", 0)
                step = meta.get("step", 1)
                
                # If this is result metadata, redirect to the preprocessed intermediate NPZ
                if "preprocess_npz" in meta:
                    prep_npz = meta["preprocess_npz"]
                    if os.path.exists(prep_npz):
                        npz_path = prep_npz
            except Exception as e:
                print(f"Error loading companion json: {e}")
                
        try:
            data = np.load(npz_path, allow_pickle=True)
            if "depths" not in data:
                return False
                
            self.depths = data["depths"]
            self.unc_metric = data.get("unc_metric", None)
            self.start_frame = start_frame
            self.end_frame = end_frame
            self.step = step
            self.loaded_npz_path = npz_path
            return True
        except Exception as e:
            print(f"Error loading npz file: {e}")
            return False
            
    def update_ui_state(self):
        if self.depths is not None:
            self.lbl_status.setText("Visualization Data: Loaded")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 12px;")
            
            basename = os.path.basename(self.loaded_npz_path)
            details = (
                f"<b>File:</b> {basename}<br>"
                f"<b>Frames:</b> {len(self.depths)}<br>"
                f"<b>Resolution:</b> {self.depths.shape[2]}x{self.depths.shape[1]}<br>"
                f"<b>Trim Bounds:</b> {self.start_frame} to {self.end_frame} (step {self.step})"
            )
            self.lbl_details.setText(details)
            self.lbl_details.setStyleSheet("color: #E0E0E0; font-size: 11px;")
        else:
            self.lbl_status.setText("Visualization Data: Not Loaded")
            self.lbl_status.setStyleSheet("color: #FF5252; font-weight: bold; font-size: 12px;")
            self.lbl_details.setText("No active preprocessed depth maps found. Please run preprocessing or tracking first, or load a session.")
            self.lbl_details.setStyleSheet("color: #AAAAAA; font-size: 11px;")
            
    def get_depth_qimage(self, idx):
        try:
            frame = self.depths[idx].copy()
            
            # Apply uncertainty masking if available
            if self.unc_metric is not None:
                try:
                    unc_frame = self.unc_metric[idx]
                    if unc_frame.dtype == bool:
                        mask = unc_frame
                    else:
                        mask = unc_frame < 0.5
                    frame[mask] = 0.0
                except Exception:
                    pass
                    
            # Define invalid pixels
            invalid_mask = frame <= 0.01
            
            # Robust normalization (percentile-based)
            valid_depths = frame[~invalid_mask]
            if len(valid_depths) > 0:
                d_min = np.percentile(valid_depths, 2)
                d_max = np.percentile(valid_depths, 98)
                if d_max == d_min:
                    d_max += 1e-6
            else:
                d_min, d_max = 0.0, 1.0
                
            frame_clipped = np.clip(frame, d_min, d_max)
            frame_norm = ((frame_clipped - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
            
            # Default to colormap JET, don't provide option to change color
            frame_colored = cv2.applyColorMap(frame_norm, cv2.COLORMAP_JET)
            frame_colored[invalid_mask] = 0
            
            frame_rgb = cv2.cvtColor(frame_colored, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qimage = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            return qimage.copy()
        except Exception as e:
            print(f"Error rendering depth frame: {e}")
            return None
            
    def _on_manual_load(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Preprocessed NPZ or JSON Metadata", "", "Supported Files (*_intermediate.npz *_metadata.json *.npz *.json)"
        )
        if not filename:
            return
            
        npz_path = None
        json_path = None
        
        if filename.endswith(".json"):
            from gui_preprocess_loader import load_preprocess_metadata
            try:
                meta, filename = load_preprocess_metadata(self.main_window, self, filename)
                if meta:
                    npz_path = meta.get("npz_path") or meta.get("preprocess_npz")
                    json_path = filename
            except Exception as e:
                self.lbl_status.setText("Load Failed!")
                self.lbl_status.setStyleSheet("color: #FF5252; font-weight: bold;")
                self.lbl_details.setText(f"Error loading metadata: {e}")
                self.lbl_details.setStyleSheet("color: #AAAAAA; font-size: 11px;")
                return
        else:
            npz_path = filename
            
        if npz_path and os.path.exists(npz_path):
            success = self.load_visualization_data(npz_path, json_path)
            if success:
                self.update_ui_state()
                self.refresh_display()
            else:
                self.lbl_status.setText("Load Failed!")
                self.lbl_status.setStyleSheet("color: #FF5252; font-weight: bold;")
                self.lbl_details.setText("The selected file is not a valid preprocessed NPZ depth map dataset.")
                self.lbl_details.setStyleSheet("color: #AAAAAA; font-size: 11px;")
