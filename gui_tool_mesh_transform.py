# gui_tool_mesh_transform.py
import os
import numpy as np
import trimesh

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSlider, 
                             QPushButton, QLabel, QFileDialog, QLineEdit, 
                             QGroupBox, QGridLayout, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

class MeshTransformWidget(QWidget):
    """Unified widget to load a 3D OBJ and control its scale, offset, and rotation.
    
    Shares mesh selection, scale, and euler rotations across all widget instances via
    class-level shared state, while keeping XYZ offsets (translation) local to each instance.
    """
    
    # Class-level shared state across all instances
    _shared_state = {
        "obj_path": "",
        "mesh": None,
        "face_colors": [],
        "scale_slider_val": 100,  # default value (maps to 0.1)
        "yaw_slider_val": 0,
        "pitch_slider_val": 0,
        "roll_slider_val": 0,
    }
    
    # Signals
    changed = pyqtSignal()
    mesh_loaded = pyqtSignal(bool, str)  # (success, filename_or_error)

    def __init__(self, parent=None, group_title="Transform Adjustments"):
        super().__init__(parent)
        self.group_title = group_title
        
        # Local state (not shared, since offsets have different meanings per tool)
        self.offset_x_slider_val = 0
        self.offset_y_slider_val = 0
        self.offset_z_slider_val = 0
        
        self._init_ui()
        self.sync_from_shared()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. OBJ Loading Section
        grp_load = QGroupBox("1. Load 3D Mesh")
        load_layout = QHBoxLayout(grp_load)
        
        self.txt_obj_path = QLineEdit()
        self.txt_obj_path.setReadOnly(True)
        self.txt_obj_path.setPlaceholderText("Select an .obj file...")
        load_layout.addWidget(self.txt_obj_path)
        
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self._on_browse_obj)
        load_layout.addWidget(btn_browse)
        
        layout.addWidget(grp_load)
        
        # 2. Controls Section
        self.grp_controls = QGroupBox(f"2. {self.group_title}")
        ctrl_grid = QGridLayout(self.grp_controls)
        
        # Scale
        ctrl_grid.addWidget(QLabel("Scale:"), 0, 0)
        self.sld_scale = QSlider(Qt.Orientation.Horizontal)
        self.sld_scale.setRange(1, 2000) # 0.001 to 2.0
        self.sld_scale.setValue(100) # default 0.1
        self.sld_scale.valueChanged.connect(self._on_slider_changed)
        ctrl_grid.addWidget(self.sld_scale, 0, 1)
        self.lbl_scale = QLabel("0.100")
        ctrl_grid.addWidget(self.lbl_scale, 0, 2)
        
        # Offset X
        ctrl_grid.addWidget(QLabel("Offset X:"), 1, 0)
        self.sld_offset_x = QSlider(Qt.Orientation.Horizontal)
        self.sld_offset_x.setRange(-2000, 2000) # -2.0 to 2.0
        self.sld_offset_x.setValue(0)
        self.sld_offset_x.valueChanged.connect(self._on_slider_changed)
        ctrl_grid.addWidget(self.sld_offset_x, 1, 1)
        self.lbl_offset_x = QLabel("0.000")
        ctrl_grid.addWidget(self.lbl_offset_x, 1, 2)
        
        # Offset Y
        ctrl_grid.addWidget(QLabel("Offset Y:"), 2, 0)
        self.sld_offset_y = QSlider(Qt.Orientation.Horizontal)
        self.sld_offset_y.setRange(-2000, 2000)
        self.sld_offset_y.setValue(0)
        self.sld_offset_y.valueChanged.connect(self._on_slider_changed)
        ctrl_grid.addWidget(self.sld_offset_y, 2, 1)
        self.lbl_offset_y = QLabel("0.000")
        ctrl_grid.addWidget(self.lbl_offset_y, 2, 2)
        
        # Offset Z
        ctrl_grid.addWidget(QLabel("Offset Z:"), 3, 0)
        self.sld_offset_z = QSlider(Qt.Orientation.Horizontal)
        self.sld_offset_z.setRange(-2000, 2000)
        self.sld_offset_z.setValue(0)
        self.sld_offset_z.valueChanged.connect(self._on_slider_changed)
        ctrl_grid.addWidget(self.sld_offset_z, 3, 1)
        self.lbl_offset_z = QLabel("0.000")
        ctrl_grid.addWidget(self.lbl_offset_z, 3, 2)
        
        # Yaw (Rotation Z)
        ctrl_grid.addWidget(QLabel("Yaw (Z-rot):"), 4, 0)
        self.sld_yaw = QSlider(Qt.Orientation.Horizontal)
        self.sld_yaw.setRange(-180, 180)
        self.sld_yaw.setValue(0)
        self.sld_yaw.valueChanged.connect(self._on_slider_changed)
        ctrl_grid.addWidget(self.sld_yaw, 4, 1)
        self.lbl_yaw = QLabel("0°")
        ctrl_grid.addWidget(self.lbl_yaw, 4, 2)
        
        # Pitch (Rotation Y)
        ctrl_grid.addWidget(QLabel("Pitch (Y-rot):"), 5, 0)
        self.sld_pitch = QSlider(Qt.Orientation.Horizontal)
        self.sld_pitch.setRange(-180, 180)
        self.sld_pitch.setValue(0)
        self.sld_pitch.valueChanged.connect(self._on_slider_changed)
        ctrl_grid.addWidget(self.sld_pitch, 5, 1)
        self.lbl_pitch = QLabel("0°")
        ctrl_grid.addWidget(self.lbl_pitch, 5, 2)
        
        # Roll (Rotation X)
        ctrl_grid.addWidget(QLabel("Roll (X-rot):"), 6, 0)
        self.sld_roll = QSlider(Qt.Orientation.Horizontal)
        self.sld_roll.setRange(-180, 180)
        self.sld_roll.setValue(0)
        self.sld_roll.valueChanged.connect(self._on_slider_changed)
        ctrl_grid.addWidget(self.sld_roll, 6, 1)
        self.lbl_roll = QLabel("0°")
        ctrl_grid.addWidget(self.lbl_roll, 6, 2)
        
        layout.addWidget(self.grp_controls)
        
        # 3. Snap Button
        self.btn_snap = QPushButton("Snap Center")
        self.btn_snap.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 6px;")
        self.btn_snap.clicked.connect(self._on_snap_clicked)
        layout.addWidget(self.btn_snap)
        
        # Checkbox to toggle points
        self.chk_show_points = QCheckBox("Show tracked points & trails")
        self.chk_show_points.setChecked(True)
        self.chk_show_points.stateChanged.connect(self._on_checkbox_changed)
        layout.addWidget(self.chk_show_points)

    def set_snap_button_text(self, text):
        self.btn_snap.setText(text)

    def sync_from_shared(self):
        """Restores all shared parameters from the class-level state to update this UI instance."""
        # Temporarily block signals to avoid triggering updates during loading
        self.blockSignals(True)
        self.sld_scale.blockSignals(True)
        self.sld_yaw.blockSignals(True)
        self.sld_pitch.blockSignals(True)
        self.sld_roll.blockSignals(True)
        
        self.txt_obj_path.setText(self._shared_state["obj_path"])
        self.sld_scale.setValue(self._shared_state["scale_slider_val"])
        self.sld_yaw.setValue(self._shared_state["yaw_slider_val"])
        self.sld_pitch.setValue(self._shared_state["pitch_slider_val"])
        self.sld_roll.setValue(self._shared_state["roll_slider_val"])
        
        self.blockSignals(False)
        self.sld_scale.blockSignals(False)
        self.sld_yaw.blockSignals(False)
        self.sld_pitch.blockSignals(False)
        self.sld_roll.blockSignals(False)
        
        self._update_labels()

    def load_default_mesh_if_exist(self):
        """Loads teapot.obj by default on initialization if no mesh path is set yet."""
        if not self._shared_state["obj_path"]:
            default_obj = "teapot.obj"
            if os.path.exists(default_obj):
                self.load_obj_mesh(os.path.abspath(default_obj))

    def load_obj_mesh(self, file_path):
        if not os.path.exists(file_path):
            self.mesh_loaded.emit(False, "File path does not exist.")
            return False
            
        try:
            mesh = trimesh.load(file_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
                
            # Center the mesh vertices
            centroid = mesh.centroid
            mesh.vertices -= centroid
            
            # Colorize faces based on distance to the new center
            face_centers = mesh.vertices[mesh.faces].mean(axis=1)
            face_dists = np.linalg.norm(face_centers, axis=1)
            
            min_d = face_dists.min() if len(face_dists) > 0 else 0.0
            max_d = face_dists.max() if len(face_dists) > 0 else 1.0
            
            face_colors = []
            for d in face_dists:
                face_colors.append(self.get_color_from_distance(d, min_d, max_d))
                
            # Update shared class-level state
            self._shared_state["obj_path"] = file_path
            self._shared_state["mesh"] = mesh
            self._shared_state["face_colors"] = face_colors
            
            # Update UI on current instance
            self.txt_obj_path.setText(file_path)
            
            # Emit success
            self.mesh_loaded.emit(True, os.path.basename(file_path))
            self.changed.emit()
            return True
        except Exception as e:
            print(f"Error loading OBJ: {e}")
            self.mesh_loaded.emit(False, str(e))
            return False

    def get_color_from_distance(self, d, min_d, max_d):
        if max_d == min_d:
            t = 0.5
        else:
            t = (d - min_d) / (max_d - min_d)
        t = np.clip(t, 0.0, 1.0)
        
        # Premium gradient: Deep violet-blue (hue 240) to vibrant pink-red (hue 0/360)
        hue = int(240 - 240 * t)
        return QColor.fromHsv(hue, 200, 240)

    def _on_slider_changed(self):
        # Update shared state
        self._shared_state["scale_slider_val"] = self.sld_scale.value()
        self._shared_state["yaw_slider_val"] = self.sld_yaw.value()
        self._shared_state["pitch_slider_val"] = self.sld_pitch.value()
        self._shared_state["roll_slider_val"] = self.sld_roll.value()
        
        # Update local state
        self.offset_x_slider_val = self.sld_offset_x.value()
        self.offset_y_slider_val = self.sld_offset_y.value()
        self.offset_z_slider_val = self.sld_offset_z.value()
        
        self._update_labels()
        self.changed.emit()

    def _on_checkbox_changed(self, state):
        self.changed.emit()

    def _on_snap_clicked(self):
        self.sld_offset_x.setValue(0)
        self.sld_offset_y.setValue(0)
        self.sld_offset_z.setValue(0)
        self._on_slider_changed()

    def _on_browse_obj(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open OBJ File", "", "OBJ Files (*.obj)"
        )
        if filename:
            self.load_obj_mesh(filename)

    def _update_labels(self):
        scale = self.sld_scale.value() / 1000.0
        self.lbl_scale.setText(f"{scale:.3f}")
        
        ox = self.sld_offset_x.value() / 1000.0
        self.lbl_offset_x.setText(f"{ox:.3f}")
        
        oy = self.sld_offset_y.value() / 1000.0
        self.lbl_offset_y.setText(f"{oy:.3f}")
        
        oz = self.sld_offset_z.value() / 1000.0
        self.lbl_offset_z.setText(f"{oz:.3f}")
        
        self.lbl_yaw.setText(f"{self.sld_yaw.value()}°")
        self.lbl_pitch.setText(f"{self.sld_pitch.value()}°")
        self.lbl_roll.setText(f"{self.sld_roll.value()}°")

    # Accessors for Rendering/Simulation
    def get_mesh(self):
        return self._shared_state["mesh"]
        
    def get_face_colors(self):
        return self._shared_state["face_colors"]
        
    def get_obj_path(self):
        return self._shared_state["obj_path"]

    def get_scale(self):
        return self.sld_scale.value() / 1000.0

    def get_offset(self):
        return np.array([
            self.sld_offset_x.value() / 1000.0,
            self.sld_offset_y.value() / 1000.0,
            self.sld_offset_z.value() / 1000.0
        ])
        
    def get_euler_rotation_matrix(self):
        yaw = np.radians(self.sld_yaw.value())
        pitch = np.radians(self.sld_pitch.value())
        roll = np.radians(self.sld_roll.value())
        
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

    def get_show_points(self):
        return self.chk_show_points.isChecked()
