import os
import json
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from gui_backend import get_video_dimensions

def gui_tool_load_video(session, parent_widget, video_path, json_path=None):
    """
    Universal video loader that resolves absolute or relative video paths.
    If the video file is not found, prompts the user to locate it.
    If a new video is loaded, sets the video_path key in the session.
    """
    if not video_path:
        QMessageBox.critical(parent_widget, "Error", "No video path specified.")
        return None

    resolved_path = os.path.abspath(video_path)
    
    if not os.path.exists(resolved_path):
        # Try resolving relative to the JSON file directory
        if json_path:
            json_dir = os.path.dirname(os.path.abspath(json_path))
            rel_path = os.path.join(json_dir, os.path.basename(video_path))
            if os.path.exists(rel_path):
                resolved_path = os.path.abspath(rel_path)
        
        # If still not found, prompt the user to browse for the video file
        if not os.path.exists(resolved_path):
            browse_dir = os.path.dirname(json_path) if json_path else ""
            selected_path, _ = QFileDialog.getOpenFileName(
                parent_widget,
                f"Locate Video File ({os.path.basename(video_path)})",
                browse_dir,
                "Video Files (*.mp4 *.avi *.mkv *.mov)"
            )
            if selected_path:
                resolved_path = os.path.abspath(selected_path)
            else:
                QMessageBox.critical(
                    parent_widget,
                    "Error",
                    f"Associated video file '{os.path.basename(video_path)}' could not be found."
                )
                return None

    # Verify the video can be opened using backend dimensions utility
    try:
        w, h = get_video_dimensions(resolved_path)
    except Exception as e:
        QMessageBox.critical(
            parent_widget,
            "Error",
            f"Failed to open video file: {resolved_path}\nError: {e}"
        )
        return None

    # Load video into session state if it's different from the currently loaded video
    current_video = session.get('video_path', '')
    if not current_video or os.path.abspath(current_video) != resolved_path:
        session.set('video_path', resolved_path)

    return resolved_path

def load_preprocess_metadata(session, parent_widget, filename=None):
    """
    Universal loader for preprocessing metadata JSON.
    """
    if not filename:
        filename, _ = QFileDialog.getOpenFileName(
            parent_widget,
            "Load Preprocess Metadata",
            "",
            "JSON Files (*_metadata.json *.json)"
        )
        if not filename:
            return None, None

    filename = os.path.abspath(filename)
    try:
        with open(filename, 'r') as f:
            meta = json.load(f)
    except Exception as e:
        QMessageBox.critical(parent_widget, "Error", f"Failed to parse JSON file:\n{e}")
        return None, None

    # Resolve video path
    video_path = meta.get("video_path")
    resolved_video_path = gui_tool_load_video(session, parent_widget, video_path, filename)
    if not resolved_video_path:
        return None, None
    meta["video_path"] = resolved_video_path

    # Resolve NPZ path
    npz_path = meta.get("npz_path") or meta.get("preprocess_npz")
    if npz_path:
        resolved_npz = os.path.abspath(npz_path)
        if not os.path.exists(resolved_npz):
            # Try relative to the JSON metadata file
            rel_npz = os.path.join(os.path.dirname(filename), os.path.basename(npz_path))
            if os.path.exists(rel_npz):
                resolved_npz = os.path.abspath(rel_npz)
        
        # Update NPZ path in metadata dict
        if "npz_path" in meta:
            meta["npz_path"] = resolved_npz
        if "preprocess_npz" in meta:
            meta["preprocess_npz"] = resolved_npz
            
        if not os.path.exists(resolved_npz):
            QMessageBox.warning(
                parent_widget,
                "Missing Preprocessed Data",
                f"The preprocessed NPZ file was not found at:\n{resolved_npz}\n\n"
                "Please run preprocessing again to generate it."
            )
        else:
            # Sync session reactively
            session.set("preprocess_npz", resolved_npz)
            session.set("preprocess_json", filename)
            session.set("trim_range", (meta.get("start_frame", 0), meta.get("end_frame", 0)))

    return meta, filename

def load_tracking_result(session, parent_widget, filename=None):
    """
    Universal loader for tracking session / result metadata JSON.
    """
    if not filename:
        filename, _ = QFileDialog.getOpenFileName(
            parent_widget,
            "Load Tracking Session",
            "",
            "JSON Files (*_result_metadata.json *.json)"
        )
        if not filename:
            return None, None

    filename = os.path.abspath(filename)
    try:
        with open(filename, 'r') as f:
            meta = json.load(f)
    except Exception as e:
        QMessageBox.critical(parent_widget, "Error", f"Failed to parse JSON file:\n{e}")
        return None, None

    # Resolve video path
    video_path = meta.get("video_path")
    resolved_video_path = gui_tool_load_video(session, parent_widget, video_path, filename)
    if not resolved_video_path:
        return None, None
    meta["video_path"] = resolved_video_path

    # Resolve preprocess and tracking result paths
    preprocess_npz = meta.get("preprocess_npz")
    preprocess_json = meta.get("preprocess_json")
    result_npz = meta.get("result_npz")
    json_dir = os.path.dirname(filename)

    # 1. Resolve preprocess NPZ
    if preprocess_npz:
        resolved_prep_npz = os.path.abspath(preprocess_npz)
        if not os.path.exists(resolved_prep_npz):
            rel_prep_npz = os.path.join(json_dir, os.path.basename(preprocess_npz))
            if os.path.exists(rel_prep_npz):
                resolved_prep_npz = os.path.abspath(rel_prep_npz)
        meta["preprocess_npz"] = resolved_prep_npz

    # 2. Resolve preprocess JSON
    if preprocess_json:
        resolved_prep_json = os.path.abspath(preprocess_json)
        if not os.path.exists(resolved_prep_json):
            rel_prep_json = os.path.join(json_dir, os.path.basename(preprocess_json))
            if os.path.exists(rel_prep_json):
                resolved_prep_json = os.path.abspath(rel_prep_json)
        meta["preprocess_json"] = resolved_prep_json

    # 3. Resolve result NPZ
    if result_npz:
        resolved_res_npz = os.path.abspath(result_npz)
        if not os.path.exists(resolved_res_npz):
            rel_res_npz = os.path.join(json_dir, os.path.basename(result_npz))
            if os.path.exists(rel_res_npz):
                resolved_res_npz = os.path.abspath(rel_res_npz)
        meta["result_npz"] = resolved_res_npz

    # Check for missing files and notify
    missing = []
    if not meta.get("preprocess_npz") or not os.path.exists(meta["preprocess_npz"]):
        missing.append("Preprocessed NPZ file")
    if not meta.get("preprocess_json") or not os.path.exists(meta["preprocess_json"]):
        missing.append("Preprocessed metadata JSON file")
    if not meta.get("result_npz") or not os.path.exists(meta["result_npz"]):
        missing.append("Tracking results NPZ file")

    if missing:
        QMessageBox.warning(
            parent_widget,
            "Incomplete Session Data",
            "The following associated data files were not found and might need update/regeneration:\n" +
            "\n".join(f"- {item}" for item in missing)
        )
    
    # Sync session reactively
    if meta.get("preprocess_npz") and os.path.exists(meta["preprocess_npz"]):
        session.set("preprocess_npz", meta["preprocess_npz"])
    if meta.get("preprocess_json") and os.path.exists(meta["preprocess_json"]):
        session.set("preprocess_json", meta["preprocess_json"])
    if meta.get("result_npz") and os.path.exists(meta["result_npz"]):
        session.set("tracking_result_npz", meta["result_npz"])
    
    session.set("trim_range", (meta.get("start_frame", 0), meta.get("end_frame", 0)))

    return meta, filename
