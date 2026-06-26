import os
import gc
import glob
import time
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.predictor import Predictor

def compute_aspect_preserving_size(w_orig, h_orig, target_max_dim):
    if w_orig >= h_orig:
        new_w = target_max_dim
        new_h = round(h_orig * (target_max_dim / w_orig) / 14) * 14
    else:
        new_h = target_max_dim
        new_w = round(w_orig * (target_max_dim / h_orig) / 14) * 14
    return new_w, new_h

def get_grid_queries(width, height, cols=8, rows=6):
    xs = np.linspace(width // (cols + 1), width - width // (cols + 1), cols)
    ys = np.linspace(height // (rows + 1), height - height // (rows + 1), rows)
    xv, yv = np.meshgrid(xs, ys)
    points = np.stack([xv.flatten(), yv.flatten()], axis=-1)
    
    query_xyt = np.zeros((len(points), 3), dtype=np.float32)
    query_xyt[:, 0] = 0.0  # Start frame index is 0
    query_xyt[:, 1] = points[:, 0]
    query_xyt[:, 2] = points[:, 1]
    return query_xyt

def compute_3d_metrics(coords_gt, vis_gt, coords_pred, vis_pred, thresholds=[1.0, 2.0, 4.0, 8.0, 16.0]):
    v_gt = (vis_gt > 0.5).astype(np.float32)
    v_pred = (vis_pred > 0.5).astype(np.float32)
    
    # Euclidean distance in centimeters (coordinates expected in meters)
    dist = np.linalg.norm(coords_pred - coords_gt, axis=-1) * 100.0
    
    oa = np.mean(v_gt == v_pred)
    
    ap_list = []
    aj_list = []
    T, N = v_gt.shape
    
    for delta in thresholds:
        # True Positives: visible in both and within threshold distance
        tp = (v_gt == 1) & (v_pred == 1) & (dist <= delta)
        
        sum_visible_gt = np.sum(v_gt == 1)
        if sum_visible_gt > 0:
            ap = np.sum(tp) / sum_visible_gt
        else:
            ap = 1.0
        ap_list.append(ap)
        
        aj_points = []
        for i in range(N):
            tp_i = np.sum((v_gt[:, i] == 1) & (v_pred[:, i] == 1) & (dist[:, i] <= delta))
            denom_i = np.sum((v_gt[:, i] == 1) | (v_pred[:, i] == 1))
            if denom_i > 0:
                aj_points.append(tp_i / denom_i)
            else:
                aj_points.append(1.0)
        aj_list.append(np.mean(aj_points))
        
    return {
        "occlusion_accuracy": float(oa),
        "average_position_accuracy": float(np.mean(ap_list)),
        "average_jaccard": float(np.mean(aj_list))
    }

def run_benchmark():
    # 1. Gather all mp4 videos and check frame count
    video_dir = "examples"
    all_video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    
    if not all_video_paths:
        print(f"Error: No video files found in '{video_dir}/'")
        return

    lengths = [50, 100, 200]
    max_length_needed = max(lengths)
    
    print("Scanning videos for frame count...")
    video_paths = []
    for vp in all_video_paths:
        cap = cv2.VideoCapture(vp)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Keep video if it has at least the maximum benchmark length
        if total_frames >= max_length_needed:
            video_paths.append(vp)
            if len(video_paths) == 10:
                break
                
    if len(video_paths) < 10:
        print(f"Warning: Only found {len(video_paths)} videos with at least {max_length_needed} frames. Using all of them.")
        # If we couldn't find enough, fill up with any other videos to make up 10
        for vp in all_video_paths:
            if vp not in video_paths:
                video_paths.append(vp)
                if len(video_paths) == 10:
                    break

    print(f"Selected {len(video_paths)} videos to benchmark:")
    for vp in video_paths:
        print(f" - {vp}")

    test_resolutions = [252, 336, 518]
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Results nested dictionary: length -> resolution -> list of metric dicts
    eval_results = {L: {R: [] for R in test_resolutions} for L in lengths}

    for video_idx, video_path in enumerate(video_paths):
        print(f"\n[{video_idx + 1}/10] Processing video: {video_path}")
        
        # Read all frames of video
        cap = cv2.VideoCapture(video_path)
        raw_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_frames.append(frame_rgb)
        cap.release()
        
        total_frames = len(raw_frames)
        if total_frames == 0:
            print(f"Skipping empty or unreadable video: {video_path}")
            continue

        H_orig, W_orig = raw_frames[0].shape[0], raw_frames[0].shape[1]
        W_gt, H_gt = compute_aspect_preserving_size(W_orig, H_orig, 518)
        query_xyt = get_grid_queries(W_gt, H_gt, cols=8, rows=6)

        # Scale frames to GT resolution (518 max-dim)
        frames_gt = [cv2.resize(f, (W_gt, H_gt), interpolation=cv2.INTER_CUBIC) for f in raw_frames]
        video_np_gt = np.stack(frames_gt)
        video_tensor_gt = torch.from_numpy(video_np_gt).permute(0, 3, 1, 2).float() # (T, 3, H_gt, W_gt)

        for L in lengths:
            actual_len = min(L, total_frames)
            print(f"  Length: {actual_len} frames")
            
            # Slice video tensors to length
            video_tensor_gt_L = video_tensor_gt[:actual_len]
            
            # -------------------------------------------------------------
            # GROUND TRUTH LABELS GENERATION (Preprocess at 518, Refiner at 518)
            # -------------------------------------------------------------
            
            # Preprocess Phase GT
            preprocess_frames_gt = []
            for t in range(actual_len):
                img_proc = preprocess_image(video_tensor_gt_L[t], target_size=518)
                preprocess_frames_gt.append(img_proc)
            vggt_input_gt = torch.stack(preprocess_frames_gt).to(device=device, dtype=dtype)
            
            # Clean VRAM & load VGGT4Track
            gc.collect()
            torch.cuda.empty_cache()
            
            vggt_model = VGGT4Track.from_pretrained(
                "Yuxihenry/SpatialTrackerV2_Front", offload_block=True, enable_chunking=True
            ).eval().to(device=device, dtype=dtype)
            
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=dtype):
                        predictions_gt = vggt_model(vggt_input_gt[None] / 255.0)
            
            depth_gt = predictions_gt["points_map"][..., 2].float().cpu().numpy()
            unc_metric_gt = (predictions_gt["unc_metric"].float().cpu().numpy() > 0.5)
            intrs_gt = predictions_gt["intrs"][0].float().cpu().numpy()
            extrs_gt = predictions_gt["poses_pred"][0].float().cpu().numpy()
            extrs_gt_c2w = np.linalg.inv(extrs_gt).astype(np.float32)
            
            # Unload VGGT4Track
            del predictions_gt, vggt_model
            gc.collect()
            torch.cuda.empty_cache()
            
            # Refinement Phase GT
            predictor = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
            predictor.spatrack.track_num = 120
            predictor.S_wind = 30
            predictor.overlap = 10
            predictor.eval().to(device)
            if hasattr(predictor.spatrack, "base_model") and predictor.spatrack.base_model is not None:
                predictor.spatrack.base_model.to("cpu")
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    (
                        c2w_traj_gt, intrs_ref_gt, _, _,
                        track3d_gt, _, vis_gt, _, _
                    ) = predictor.forward(
                        video_tensor_gt_L, depth=depth_gt,
                        intrs=intrs_gt, extrs=extrs_gt_c2w,
                        queries=query_xyt, fps=1, full_point=False, iters_track=8,
                        query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric_gt,
                        support_frame=actual_len-1, replace_ratio=0.2
                    )
            
            coords_gt_label = (torch.einsum("tij,tnj->tni", c2w_traj_gt[:,:3,:3], track3d_gt[:,:,:3].cpu()) + c2w_traj_gt[:,:3,3][:,None,:]).numpy()
            vis_gt_label = vis_gt.cpu().numpy()
            
            # Unload Predictor
            del predictor
            gc.collect()
            torch.cuda.empty_cache()

            # -------------------------------------------------------------
            # TEST RESOLUTIONS RUN
            # -------------------------------------------------------------
            for R in test_resolutions:
                # -- Phase 1: VGGT4Track Preprocessing at resolution R --
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                W_R, H_R = compute_aspect_preserving_size(W_orig, H_orig, R)
                preprocess_frames_R = []
                for t in range(actual_len):
                    img_proc = preprocess_image(video_tensor_gt_L[t], target_size=R)
                    preprocess_frames_R.append(img_proc)
                vggt_input_R = torch.stack(preprocess_frames_R).to(device=device, dtype=dtype)
                
                vggt_model = VGGT4Track.from_pretrained(
                    "Yuxihenry/SpatialTrackerV2_Front", offload_block=True, enable_chunking=True
                ).eval().to(device=device, dtype=dtype)
                
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    with torch.no_grad():
                        with torch.amp.autocast(device_type="cuda", dtype=dtype):
                            predictions_R = vggt_model(vggt_input_R[None] / 255.0)
                
                depth_R = predictions_R["points_map"][..., 2].float().cpu() # Move to host CPU immediately
                unc_metric_R = predictions_R["unc_metric"].float().cpu()
                intrs_R = predictions_R["intrs"][0].float().cpu().numpy()
                extrs_R = predictions_R["poses_pred"][0].float().cpu().numpy()
                extrs_R_c2w = np.linalg.inv(extrs_R).astype(np.float32)
                
                # Unload VGGT4Track & get peak memory for Preprocess Phase
                del predictions_R, vggt_model
                gc.collect()
                torch.cuda.empty_cache()
                peak_vggt = torch.cuda.max_memory_allocated() / (1024 ** 2)
                
                # -- Phase 2: Predictor Refinement at resolution 518 --
                torch.cuda.reset_peak_memory_stats()
                
                # Interpolate preprocessed outputs back to ground truth resolution
                depth_interp = F.interpolate(
                    depth_R.unsqueeze(0).unsqueeze(1), size=(H_gt, W_gt), mode='bilinear', align_corners=True
                ).squeeze(0).squeeze(0).numpy()
                
                unc_interp = (F.interpolate(
                    unc_metric_R.unsqueeze(0).unsqueeze(1), size=(H_gt, W_gt), mode='bilinear', align_corners=True
                ).squeeze(0).squeeze(0).numpy() > 0.5)
                
                # Scale intrinsics
                scale_w = W_gt / W_R
                scale_h = H_gt / H_R
                intrs_scaled = intrs_R.copy()
                intrs_scaled[:, 0, :] *= scale_w
                intrs_scaled[:, 1, :] *= scale_h
                
                predictor = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
                predictor.spatrack.track_num = 120
                predictor.S_wind = 30
                predictor.overlap = 10
                predictor.eval().to(device)
                if hasattr(predictor.spatrack, "base_model") and predictor.spatrack.base_model is not None:
                    predictor.spatrack.base_model.to("cpu")
                torch.cuda.empty_cache()
                
                # Run Refiner
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=dtype):
                        (
                            c2w_traj_pred, _, _, _,
                            track3d_pred, _, vis_pred, _, _
                        ) = predictor.forward(
                            video_tensor_gt_L, depth=depth_interp,
                            intrs=intrs_scaled, extrs=extrs_R_c2w,
                            queries=query_xyt, fps=1, full_point=False, iters_track=8,
                            query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_interp,
                            support_frame=actual_len-1, replace_ratio=0.2
                        )
                
                coords_pred_label = (torch.einsum("tij,tnj->tni", c2w_traj_pred[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj_pred[:,:3,3][:,None,:]).numpy()
                vis_pred_label = vis_pred.cpu().numpy()
                
                # Unload Predictor & get peak memory for Refinement Phase
                del predictor
                gc.collect()
                torch.cuda.empty_cache()
                peak_refiner = torch.cuda.max_memory_allocated() / (1024 ** 2)
                
                # Overall peak is the maximum of the two independent passes
                total_peak_mem = max(peak_vggt, peak_refiner)

                # Calculate metrics
                metrics = compute_3d_metrics(coords_gt_label, vis_gt_label, coords_pred_label, vis_pred_label)
                metrics["peak_vggt"] = peak_vggt
                metrics["peak_refiner"] = peak_refiner
                metrics["peak_total"] = total_peak_mem
                eval_results[L][R].append(metrics)
                
                print(f"    Res {R}x{R} -> AJ-3D: {metrics['average_jaccard']:.4f} | AP-3D: {metrics['average_position_accuracy']:.4f} | OA: {metrics['occlusion_accuracy']:.4f} | VGGT Peak: {peak_vggt:.2f} MB | Refiner Peak: {peak_refiner:.2f} MB | Max Peak VRAM: {total_peak_mem:.2f} MB")

    # Aggregate and average metrics over all processed videos
    print("\n" + "=" * 100)
    print("RESOLUTION BENCHMARK COMPLETED")
    print("=" * 100)
    
    # Save Report
    report_path = "benchmark_resolution_results.md"
    with open(report_path, "w") as f:
        f.write("# VGGT4Track Preprocessing Resolution Benchmark Report\n\n")
        f.write("- **Methodology**: Ground truth computed using Preprocessing at 518x518. Lower resolutions preprocessed, interpolated back, and run in the track refiner at 518x518.\n")
        f.write("- **Execution Protocol**: Run in 2 passes sequentially (VGGT Preprocessing phase first, then unloaded, followed by Predictor tracking phase, and then unloaded). Peak VRAM represents the maximum memory utilized in either phase.\n")
        f.write("- **Metrics averaged across 10 videos** using 48 query points (8x6 grid).\n\n")
        
        for L in lengths:
            f.write(f"## Sequence Length: {L} frames\n\n")
            f.write("| Preprocess Resolution | AJ-3D | AP-3D | OA-3D | VGGT Peak VRAM | Refiner Peak VRAM | Combined Max Peak |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
            
            print(f"\nSequence Length: {L} frames")
            print("-" * 120)
            print(f"{'Resolution':<12} | {'AJ-3D':<10} | {'AP-3D':<10} | {'OA-3D':<10} | {'VGGT Peak VRAM':<18} | {'Refiner Peak':<18} | {'Max Peak (MB)':<15}")
            print("-" * 120)
            
            for R in test_resolutions:
                metrics_list = eval_results[L][R]
                if not metrics_list:
                    continue
                mean_aj = np.mean([m["average_jaccard"] for m in metrics_list])
                mean_ap = np.mean([m["average_position_accuracy"] for m in metrics_list])
                mean_oa = np.mean([m["occlusion_accuracy"] for m in metrics_list])
                mean_vggt_mem = np.mean([m["peak_vggt"] for m in metrics_list])
                mean_refiner_mem = np.mean([m["peak_refiner"] for m in metrics_list])
                mean_peak_mem = np.mean([m["peak_total"] for m in metrics_list])
                
                row_str = f"| {R}x{R} | {mean_aj:.4f} | {mean_ap:.4f} | {mean_oa:.4f} | {mean_vggt_mem:.2f} MB | {mean_refiner_mem:.2f} MB | {mean_peak_mem:.2f} MB |\n"
                f.write(row_str)
                
                print(f"{f'{R}x{R}':<12} | {mean_aj:<10.4f} | {mean_ap:<10.4f} | {mean_oa:<10.4f} | {f'{mean_vggt_mem:.2f} MB':<18} | {f'{mean_refiner_mem:.2f} MB':<18} | {mean_peak_mem:<15.2f}")
            f.write("\n")
            
    print(f"\nFull report written to {report_path}")

if __name__ == "__main__":
    run_benchmark()
