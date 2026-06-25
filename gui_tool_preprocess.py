import os
import json
import numpy as np
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QWidget, QComboBox
from PyQt6.QtCore import pyqtSignal

from gui_tool_base import BaseTool
from gui_tool_preprocess_logic import PreprocessingThread

class PreprocessTool(BaseTool):
    """Tool to configure, run, and load VGGT4Track preprocessing for SpatialTrackerV2."""

    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self.video_path = ""
        self.pts_map = []
        self.thread = None
        self._init_ui()
        
        # Sync initial state if video metadata is already loaded
        meta = self.session.get("video_metadata")
        if meta:
            self.video_path = meta.get("video_path", "")
            self.pts_map = meta.get("pts_map", [])
        else:
            self.video_path = self.session.get("video_path", "")
            
        self._update_paths_label()

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
        
        # Chunk Size Input (hidden as stitching has been removed)
        chunk_layout = QHBoxLayout()
        self.lbl_chunk = QLabel("Chunk Size:")
        chunk_layout.addWidget(self.lbl_chunk)
        self.spin_chunk = QSpinBox()
        self.spin_chunk.setRange(4, 128)
        self.spin_chunk.setValue(24)
        self.spin_chunk.valueChanged.connect(self._on_chunk_changed)
        chunk_layout.addWidget(self.spin_chunk)
        layout.addLayout(chunk_layout)
        self.lbl_chunk.hide()
        self.spin_chunk.hide()
        
        # Overlap Input (hidden as stitching has been removed)
        overlap_layout = QHBoxLayout()
        self.lbl_overlap = QLabel("Overlap Size:")
        overlap_layout.addWidget(self.lbl_overlap)
        self.spin_overlap = QSpinBox()
        self.spin_overlap.setRange(2, 64)
        self.spin_overlap.setValue(12)
        self.spin_overlap.valueChanged.connect(self._on_overlap_changed)
        overlap_layout.addWidget(self.spin_overlap)
        layout.addLayout(overlap_layout)
        self.lbl_overlap.hide()
        self.spin_overlap.hide()
        
        # Inference Size Input
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Inference Size:"))
        self.combo_size = QComboBox()
        self.combo_size.addItem("518 (High VRAM)", 518)
        self.combo_size.addItem("336 (Medium VRAM)", 336)
        self.combo_size.addItem("252 (Low VRAM)", 252)
        self.combo_size.addItem("168 (Ultra Low VRAM)", 168)
        self.combo_size.addItem("126 (Minimum VRAM)", 126)
        self.combo_size.setCurrentIndex(1) # Default to 336
        size_layout.addWidget(self.combo_size)
        layout.addLayout(size_layout)
        
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

    def _on_session_changed(self, key, value):
        if key == "video_metadata":
            if value:
                self.video_path = value.get("video_path", "")
                self.pts_map = value.get("pts_map", [])
                self._update_paths_label()
        elif key == "video_path":
            self.video_path = value or ""
            self._update_paths_label()
        elif key == "trim_range":
            self._update_paths_label()

    def _on_chunk_changed(self, val):
        if self.spin_overlap.value() >= val:
            self.spin_overlap.setValue(max(2, val // 2))
        self.spin_overlap.setMaximum(max(2, val - 1))
        
    def _on_overlap_changed(self, val):
        if self.spin_chunk.value() <= val:
            self.spin_chunk.setValue(val + 1)

    def showEvent(self, event):
        super().showEvent(event)
        self._update_paths_label()

    def _get_current_trim_bounds(self):
        trim_range = self.session.get("trim_range")
        if trim_range:
            return trim_range
        meta = self.session.get("video_metadata")
        if meta:
            return 0, max(0, meta.get("total_frames", 0) - 1)
        return 0, 0

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
        self.spin_chunk.setEnabled(False)
        self.spin_overlap.setEnabled(False)
        self.combo_size.setEnabled(False)
        
        self.lbl_status.setStyleSheet("color: #0078D7; font-weight: bold;")
        self.lbl_status.setText("<b>Status:</b> Preparing thread...")
        
        # Fetch decoder from session state and construct synchronous decode closure
        decoder = self.session.get("decoder")
        if not decoder:
            self.lbl_status.setText("<b>Status:</b> Error - Decoder not initialized in session!")
            self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")
            
            # Re-enable inputs
            self.btn_run.setEnabled(True)
            self.btn_load.setEnabled(True)
            self.spin_step.setEnabled(True)
            self.spin_chunk.setEnabled(True)
            self.spin_overlap.setEnabled(True)
            self.combo_size.setEnabled(True)
            return

        def decode_fn(s, e, st):
            return decoder.decode_current_video_frames(s, e, st)

        chunk_size = self.spin_chunk.value()
        overlap = self.spin_overlap.value()
        target_size = self.combo_size.currentData()
        self.thread = PreprocessingThread(
            video_path=self.video_path,
            start_frame=start,
            end_frame=end,
            step=step,
            npz_path=npz_path,
            json_path=json_path,
            chunk_size=chunk_size,
            overlap=overlap,
            target_size=target_size,
            decode_fn=decode_fn
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
        self.spin_chunk.setEnabled(True)
        self.spin_overlap.setEnabled(True)
        self.combo_size.setEnabled(True)
        
        if success:
            self.session.set("preprocess_npz", self.thread.npz_path)
            self.session.set("preprocess_json", self.thread.json_path)
            self.lbl_status.setText(f"<b>Status:</b> Success!<br>Saved to: {os.path.basename(message)}")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.lbl_status.setText(f"<b>Status:</b> Failed!<br>Error: {message}")
            self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")

    def _on_load(self):
        from gui_preprocess_loader import load_preprocess_metadata
        try:
            meta, filename = load_preprocess_metadata(self.session, self)
            if not meta:
                return
                
            step = meta.get("step", 1)
            npz_path = self.session.get("preprocess_npz")
            
            self.video_path = self.session.get("video_path", "")
            self.spin_step.setValue(step)
            self._update_paths_label()
            
            self.lbl_status.setText(f"<b>Status:</b> Loaded successfully!<br>NPZ: {os.path.basename(npz_path)}")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        except Exception as e:
            self.lbl_status.setText(f"<b>Status:</b> Load failed!<br>Error: {str(e)}")
            self.lbl_status.setStyleSheet("color: #F44336; font-weight: bold;")
