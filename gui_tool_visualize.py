import os
import cv2
import numpy as np
import math
from PyQt6.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QWidget, QFileDialog, QComboBox, QCheckBox)
from PyQt6.QtGui import QImage
from PyQt6.QtCore import Qt

from gui_tool_base import BaseTool

class VisualizeVideoTool(BaseTool):
    """Tool to overlay Depth Map and VGGT4Track intermediate outputs onto the video frame."""
    
    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self.depths = None
        self.unc_metric = None
        self.start_frame = 0
        self.end_frame = 0
        self.step = 1
        self.loaded_npz_path = ""
        self.loaded_json_path = ""
        self.video_path = ""
        
        self._init_ui()
        
        # Sync initial state if already loaded
        self.video_path = self.session.get("video_path", "")
        self.auto_detect_data()
        self.update_ui_state()

    def get_name(self):
        return "Visualize Preprocessed Data"

    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Status Section
        lbl_status_title = QLabel("<b>Status</b>")
        layout.addWidget(lbl_status_title)
        
        self.lbl_status = QLabel("Preprocess Data: [Not Loaded]")
        self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")
        layout.addWidget(self.lbl_status)
        
        self.lbl_details = QLabel("No preprocessed NPZ loaded. Auto-detecting output folders or load manually.")
        self.lbl_details.setStyleSheet("color: #AAAAAA; font-size: 11px;")
        self.lbl_details.setWordWrap(True)
        layout.addWidget(self.lbl_details)
        
        # Controls Section
        layout.addWidget(QLabel("----------------------------------------"))
        lbl_ctrl_title = QLabel("<b>Controls</b>")
        layout.addWidget(lbl_ctrl_title)
        
        # Colormap Select
        cmap_layout = QHBoxLayout()
        cmap_layout.addWidget(QLabel("Colormap:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItem("Magma", cv2.COLORMAP_MAGMA)
        self.combo_cmap.addItem("Inferno", cv2.COLORMAP_INFERNO)
        self.combo_cmap.addItem("Plasma", cv2.COLORMAP_PLASMA)
        self.combo_cmap.addItem("Viridis", cv2.COLORMAP_VIRIDIS)
        self.combo_cmap.addItem("Jet", cv2.COLORMAP_JET)
        self.combo_cmap.addItem("Rainbow", cv2.COLORMAP_RAINBOW)
        self.combo_cmap.currentIndexChanged.connect(self.refresh_display)
        cmap_layout.addWidget(self.combo_cmap)
        layout.addLayout(cmap_layout)
        
        # Uncertainty Mask Checkbox
        self.cb_unc_mask = QCheckBox("Mask High Uncertainty (>0.5)")
        self.cb_unc_mask.setChecked(True)
        self.cb_unc_mask.stateChanged.connect(self.refresh_display)
        layout.addWidget(self.cb_unc_mask)
        
        # Manual Load Button
        self.btn_load = QPushButton("Load Preprocess Manually [load]")
        self.btn_load.setStyleSheet("background-color: #E1E1E1; font-weight: bold; padding: 5px;")
        self.btn_load.clicked.connect(self._on_manual_load)
        layout.addWidget(self.btn_load)
        
        layout.addStretch()

    def _on_session_changed(self, key, value):
        if key == "video_metadata":
            if value:
                self.depths = None
                self.unc_metric = None
                self.start_frame = 0
                self.end_frame = 0
                self.step = 1
                self.loaded_npz_path = ""
                self.loaded_json_path = ""
                self.video_path = value.get("video_path", "")
                self.auto_detect_data()
                self.update_ui_state()
        elif key == "video_path":
            self.video_path = value or ""
            self.auto_detect_data()
        elif key == "current_frame":
            if value is not None and self.isVisible():
                self.refresh_display()
        elif key in ("preprocess_npz", "preprocess_json"):
            self.auto_detect_data()

    def showEvent(self, event):
        super().showEvent(event)
        self.video_path = self.session.get('video_path', '')
        self.auto_detect_data()
        self.refresh_display()
        
    def hideEvent(self, event):
        super().hideEvent(event)
        # Restore the original RGB video frame when switching away by clearing override
        self.session.set("override_display_image", None)

    def refresh_display(self):
        if self.depths is not None and self.session.get("video_metadata") is not None:
            frame_idx = self.session.get("current_frame", 0)
            self._update_override_display(frame_idx)

    def _update_override_display(self, frame_idx):
        if self.depths is not None:
            if self.start_frame <= frame_idx <= self.end_frame:
                offset = frame_idx - self.start_frame
                idx_in_depths = int(round(offset / self.step))
                if 0 <= idx_in_depths < len(self.depths):
                    qimage_depth = self.get_depth_qimage(idx_in_depths)
                    if qimage_depth:
                        self.session.set("override_display_image", qimage_depth)
                        return
        self.session.set("override_display_image", None)

    def auto_detect_data(self):
        npz_path = self.session.get("preprocess_npz")
        json_path = self.session.get("preprocess_json")
        if npz_path and os.path.exists(npz_path):
            success = self.load_visualization_data(npz_path, json_path)
            if success:
                self.update_ui_state()
                return

    def load_visualization_data(self, npz_path, json_path=None):
        if self.loaded_npz_path == npz_path:
            return True
            
        try:
            data = np.load(npz_path, allow_pickle=True)
            if "depths" not in data.files:
                return False
                
            self.depths = data["depths"]
            self.unc_metric = data["unc_metric"] if "unc_metric" in data.files else None
            self.loaded_npz_path = npz_path
            
            # Read metadata JSON if provided
            if json_path and os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                self.start_frame = meta.get("start_frame", 0)
                self.end_frame = meta.get("end_frame", 0)
                self.step = meta.get("step", 1)
            else:
                # Default fallback
                self.start_frame = 0
                self.step = 1
                self.end_frame = len(self.depths) - 1
                
            return True
        except Exception:
            return False

    def update_ui_state(self):
        if self.depths is not None:
            self.lbl_status.setText("Preprocess Data: [Loaded]")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            filename = os.path.basename(self.loaded_npz_path)
            details = f"File: {filename}\nRange: {self.start_frame} to {self.end_frame} (step {self.step})\nFrames: {len(self.depths)}"
            if self.unc_metric is not None:
                details += "\nUncertainty metrics available."
            self.lbl_details.setText(details)
            self.lbl_details.setStyleSheet("color: #4CAF50; font-size: 11px;")
        else:
            self.lbl_status.setText("Preprocess Data: [Not Loaded]")
            self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")
            self.lbl_details.setText("No preprocessed NPZ loaded. Auto-detecting output folders or load manually.")
            self.lbl_details.setStyleSheet("color: #AAAAAA; font-size: 11px;")

    def get_depth_qimage(self, idx):
        try:
            frame = self.depths[idx].copy()
            
            # Apply uncertainty masking if available
            if self.unc_metric is not None and self.cb_unc_mask.isChecked():
                try:
                    unc_frame = self.unc_metric[idx]
                    if unc_frame.dtype == bool:
                        mask = unc_frame
                    else:
                        mask = unc_frame < 0.5
                    frame[mask] = 0.0
                except Exception:
                    pass
                    
            # Resize depth frame to match the original video dimensions on the fly
            meta = self.session.get("video_metadata")
            w_target, h_target = 0, 0
            if meta:
                w_target, h_target = meta.get("width", 0), meta.get("height", 0)
            if w_target > 0 and h_target > 0:
                frame = cv2.resize(frame, (w_target, h_target), interpolation=cv2.INTER_NEAREST)
                    
            # Define invalid pixels
            invalid_mask = frame <= 0.01
            
            # Robust normalization (percentile-based)
            valid_depths = frame[~invalid_mask]
            if len(valid_depths) > 0:
                d_min = np.percentile(valid_depths, 2)
                d_max = np.percentile(valid_depths, 98)
                if d_max == d_min:
                    d_max += 1e-6
                frame = np.clip(frame, d_min, d_max)
                frame = (frame - d_min) / (d_max - d_min)
            else:
                frame = np.zeros_like(frame)
                
            frame = (frame * 255.0).astype(np.uint8)
            
            # Apply Colormap
            colormap_idx = self.combo_cmap.currentData()
            colored = cv2.applyColorMap(frame, colormap_idx)
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            
            # Paint invalid pixels black
            colored[invalid_mask] = [0, 0, 0]
            
            h, w, ch = colored.shape
            bytes_per_line = ch * w
            return QImage(colored.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
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
                meta, filename = load_preprocess_metadata(self.session, self, filename)
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
                # Update session paths so other tools sync automatically
                self.session.set("preprocess_npz", npz_path)
                if json_path:
                    self.session.set("preprocess_json", json_path)
                    
                self.update_ui_state()
                self.refresh_display()
            else:
                self.lbl_status.setText("Load Failed!")
                self.lbl_status.setStyleSheet("color: #FF5252; font-weight: bold;")
                self.lbl_details.setText("The selected file is not a valid preprocessed NPZ depth map dataset.")
                self.lbl_details.setStyleSheet("color: #AAAAAA; font-size: 11px;")
