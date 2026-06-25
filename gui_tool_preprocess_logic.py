import os
import gc
import json
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image


class PreprocessingThread(QThread):
    """Background thread that loads the model, decodes video frames, runs inference, and unloads the model."""
    progress = pyqtSignal(int, int) # (decoded_frames, total_frames_to_decode)
    status = pyqtSignal(str) # Status messages
    finished = pyqtSignal(bool, str) # (success, saved_file_path or error_message)

    def __init__(self, video_path, start_frame, end_frame, step, npz_path, json_path, chunk_size=24, overlap=12, target_size=336, decode_fn=None):
        super().__init__()
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.step = step
        self.npz_path = npz_path
        self.json_path = json_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.target_size = target_size
        self.decode_fn = decode_fn

    def run(self):
        try:
            if not self.decode_fn:
                raise ValueError("No video decoder function provided.")

            self.status.emit("Starting video decoding for preprocessing sequence...")
            
            # Fetch decoding generator from caller
            generator = self.decode_fn(self.start_frame, self.end_frame, self.step)
            frames_gen = list(generator)
            if not frames_gen:
                raise ValueError("No frames were successfully decoded in the trim range.")

            total_frames_to_decode = len(frames_gen)
            self.status.emit(f"Preprocessing {total_frames_to_decode} frames sequentially...")
            
            frames = []
            count = 0
            for idx, img in frames_gen:
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
                img_proc = preprocess_image(img_tensor, target_size=self.target_size)
                frames.append(img_proc)
                count += 1
                self.progress.emit(count, total_frames_to_decode)
                
            video_tensor = torch.stack(frames)
            H_proc, W_proc = video_tensor.shape[-2:]
            
            # Determine precision based on hardware capability
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.status.emit(f"Loading VGGT4Track model to GPU (precision: {dtype})...")
            
            vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
            vggt4track_model.eval()
            vggt4track_model = vggt4track_model.to(device="cuda", dtype=dtype)
            
            self.status.emit("Running model inference in a single pass...")
            num_frames = video_tensor.shape[0]

            H_proc, W_proc = video_tensor.shape[-2:]
            final_depths = torch.zeros((num_frames, H_proc, W_proc), device="cpu", dtype=torch.float32)
            final_uncs = torch.zeros((num_frames, H_proc, W_proc), device="cpu", dtype=torch.float32)
            final_intrs = torch.zeros((num_frames, 3, 3), device="cpu", dtype=torch.float32)
            final_extrs = torch.zeros((num_frames, 4, 4), device="cpu", dtype=torch.float32)

            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=dtype):
                        predictions = vggt4track_model(video_tensor[None].cuda() / 255.0)

            # Post-processing: copy to host in chunks
            self.status.emit("Post-processing predictions in chunks...")
            post_chunk_size = 8
            
            for start_t in range(0, num_frames, post_chunk_size):
                end_t = min(start_t + post_chunk_size, num_frames)
                
                # Fetch GPU slices for intrinsics and extrinsics
                extrinsic_chunk = predictions["poses_pred"][0, start_t:end_t].float()
                intrinsic_chunk = predictions["intrs"][0, start_t:end_t].float()
                
                # Fetch depth and confidence slices (float)
                depth_map_chunk = predictions["points_map"][start_t:end_t, ..., 2].float()  # [chunk_len, H_proc, W_proc]
                depth_conf_chunk = predictions["unc_metric"][start_t:end_t].float()         # [chunk_len, H_proc, W_proc]
                
                # Store in CPU buffers without HD upsampling or scaling
                final_depths[start_t:end_t] = depth_map_chunk.cpu()
                final_uncs[start_t:end_t] = depth_conf_chunk.cpu()
                final_intrs[start_t:end_t] = intrinsic_chunk.cpu()
                final_extrs[start_t:end_t] = extrinsic_chunk.cpu()

            # Clean up predictions and empty CUDA cache
            del predictions
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
            import traceback
            traceback.print_exc()
            # Clean up model reference and cache even on failures to prevent leak
            if 'vggt4track_model' in locals():
                del vggt4track_model
            torch.cuda.empty_cache()
            gc.collect()
            self.finished.emit(False, str(e))
