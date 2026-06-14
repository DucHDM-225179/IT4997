import os
import json
from PyQt6.QtWidgets import QFileDialog
from gui_tool_trim import TrimTool

def load_preprocess_metadata(main_window, parent_widget, filename=None):
    """
    Loads preprocessed metadata, updates the video in main_window,
    restricts the timeline slider, and updates TrimTool bounds.
    If filename is not provided, prompts the user via a file dialog.
    
    Returns:
        tuple: (metadata_dict, metadata_filename) if successful, (None, None) if cancelled.
    """
    if not filename:
        filename, _ = QFileDialog.getOpenFileName(
            parent_widget, "Load Preprocessed Metadata", "", "JSON Files (*.json *_metadata.json)"
        )
    if not filename:
        return None, None
        
    try:
        with open(filename, 'r') as f:
            meta = json.load(f)
            
        video_path = meta.get("video_path")
        npz_path = meta.get("npz_path") or meta.get("preprocess_npz")
        start = meta.get("start_frame", 0)
        end = meta.get("end_frame", 0)
        step = meta.get("step", 1)
        
        if not video_path:
            raise ValueError("No video_path found in metadata JSON.")
            
        if not os.path.exists(video_path):
            # Try relative path resolution based on metadata file's folder
            meta_dir = os.path.dirname(filename)
            rel_vid = os.path.join(meta_dir, os.path.basename(video_path))
            if os.path.exists(rel_vid):
                video_path = rel_vid
            else:
                raise FileNotFoundError(f"Associated video not found at: {video_path}")
                
        # Load video via main window if not already loaded
        current_vid = getattr(main_window, 'current_video_path', '')
        if not current_vid or os.path.abspath(current_vid) != os.path.abspath(video_path):
            success = main_window.load_video(video_path)
            if not success:
                raise RuntimeError("Failed to load associated video.")
                
        # Apply trim restrictions to timeline
        main_window.apply_timeline_restriction(start, end)
        
        # Sync TrimTool labels
        for tool in main_window.tools:
            if isinstance(tool, TrimTool):
                tool.start_frame = start
                tool.end_frame = end
                tool._update_labels()
                break
                
        return meta, filename
    except Exception as e:
        raise e
