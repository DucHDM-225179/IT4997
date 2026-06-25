import os
import gc
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from models.SpaTrackV2.models.predictor import Predictor
from models.monoD.depth_anything_v2.util.transform import Resize


def generate_blender_script(npz_path, pc_npz_path, video_path, width, height, stride, start_frame):
    """
    Reads the blender_loading_script_template.py file, substitutes placeholder strings
    with resolved configuration parameters, and returns the customized Blender import script.
    """
    npz_norm = os.path.abspath(npz_path).replace("\\", "/")
    pc_npz_norm = os.path.abspath(pc_npz_path).replace("\\", "/") if pc_npz_path else ""
    video_norm = os.path.abspath(video_path).replace("\\", "/")
    
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blender_loading_script_template.py")
    
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Blender script template file not found at: {template_path}")
        
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            template_code = f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read Blender script template file: {e}")

    code = (template_code
            .replace('"TO_BE_REPLACED_NPZ_PATH"', f'"{npz_norm}"')
            .replace('"TO_BE_REPLACED_PC_NPZ_PATH"', f'"{pc_npz_norm}"' if pc_npz_norm else '""')
            .replace('"TO_BE_REPLACED_VIDEO_PATH"', f'"{video_norm}"')
            .replace('TO_BE_REPLACED_ORIGINAL_RES', f'({width}, {height})')
            .replace('TO_BE_REPLACED_FRAME_STRIDE', str(stride))
            .replace('TO_BE_REPLACED_START_FRAME', str(start_frame)))
            
    return code


class TrackingThread(QThread):
    """Background thread that runs the SpatialTrackerV2 tracking Predictor model on CUDA."""
    progress = pyqtSignal(str) # Status messages
    finished = pyqtSignal(bool, dict, str) # (success, results_dict, error_msg)

    def __init__(self, start_frame, end_frame, step, npz_preprocess_path, points, vo_points=120, fixed_cam=False, model_name="Yuxihenry/SpatialTrackerV2-Online", s_wind=30, overlap=10, decode_fn=None):
        super().__init__()
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.step = step
        self.npz_preprocess_path = npz_preprocess_path
        self.points = points  # List of (x, y) in original HD resolution
        self.vo_points = vo_points
        self.fixed_cam = fixed_cam
        self.model_name = model_name
        self.s_wind = s_wind
        self.overlap = overlap
        self.decode_fn = decode_fn

    def run(self):
        try:
            if not self.points:
                raise ValueError("No tracking points selected. Please click on the first frame of the trim to add points.")
            if not self.decode_fn:
                raise ValueError("No video decoder function provided.")

            self.progress.emit("Starting video decoding for tracking sequence...")
            
            generator = self.decode_fn(self.start_frame, self.end_frame, self.step)
            frames_gen = list(generator)
            if not frames_gen:
                raise ValueError("No video frames were successfully decoded for the tracking trim range.")

            first_frame = frames_gen[0][1]
            H_orig, W_orig = first_frame.shape[0], first_frame.shape[1]

            total_frames_to_decode = len(frames_gen)
            self.progress.emit(f"Resizing {total_frames_to_decode} frames for tracking...")
            
            resizer = Resize(
                width=336, height=336, resize_target=False, keep_aspect_ratio=True,
                ensure_multiple_of=14, resize_method='lower_bound',
            )
            new_W, new_H = resizer.get_size(W_orig, H_orig)

            frames = []
            for idx, img in frames_gen:
                img_resized = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_CUBIC)
                frames.append(img_resized)

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
            unc_metric_in = preprocess_data["unc_metric"]

            self.progress.emit("Interpolating depth and camera parameters to matched resolution...")
            depth_tensor = F.interpolate(torch.from_numpy(depth_in)[:, None], 
                                         size=video_tensor.shape[2:], 
                                         mode='bilinear', align_corners=True).squeeze(1).numpy()
            unc_metric = F.interpolate(torch.from_numpy(unc_metric_in)[:, None].float(), 
                                        size=video_tensor.shape[2:], 
                                        mode='bilinear', align_corners=True).squeeze(1).numpy() > 0.5

            scale_w = video_tensor.shape[3] / depth_in.shape[2]
            scale_h = video_tensor.shape[2] / depth_in.shape[1]
            intrs_in[:, 0, :] *= scale_w
            intrs_in[:, 1, :] *= scale_h

            self.progress.emit("Preparing tracking query coordinates...")
            scale_w_query = video_tensor.shape[3] / W_orig
            scale_h_query = video_tensor.shape[2] / H_orig
            query_xyt = np.zeros((len(self.points), 3), dtype=np.float32)
            for idx, (x_hd, y_hd) in enumerate(self.points):
                x_proc = x_hd * scale_w_query
                y_proc = y_hd * scale_h_query
                query_xyt[idx, 0] = 0.0
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
            track2d_pred[..., 0] *= W_orig / video_tensor.shape[3]
            track2d_pred[..., 1] *= H_orig / video_tensor.shape[2]
            intrs[:, 0, :] *= W_orig / video_tensor.shape[3]
            intrs[:, 1, :] *= H_orig / video_tensor.shape[2]

            results = {}
            results["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
            results["tracks_2d"] = track2d_pred.cpu().numpy()
            results["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
            results["intrinsics"] = intrs.cpu().numpy()
            results["visibs"] = vis_pred.cpu().numpy()
            results["confs"] = conf_pred.cpu().numpy()
            
            refined_depth = point_map[:, 2, ...].cpu().numpy()
            conf_depth_np = conf_depth.cpu().numpy()
            results["depths"] = np.where(conf_depth_np > 0.5, refined_depth, depth_tensor)
            results["unc_metric"] = np.where(conf_depth_np > 0.5, conf_depth_np, unc_metric)

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
