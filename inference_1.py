import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import argparse
import av
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from rich import print

def read_video_av(vid_path, step, target_size=336):
    container = av.open(vid_path)
    video_stream = container.streams.video[0]
    H_orig, W_orig = video_stream.height, video_stream.width
    
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i % step == 0:
            img = frame.to_rgb().to_ndarray()
            # Convert to tensor [C, H, W]
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            # Preprocess using model's native function to save memory early
            img_proc = preprocess_image(img_tensor, target_size=target_size)
            frames.append(img_proc)
            if len(frames) >= 40:
                break
            
    container.close()
    video_tensor = torch.stack(frames) 
    return video_tensor, (H_orig, W_orig)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="examples")
    parser.add_argument("--video_name", type=str, default="drifting")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--out_name", type=str, default=None, help="Name of the intermediate npz file")
    parser.add_argument("--chunk_size", type=int, default=24, help="Chunk size for sliding window")
    parser.add_argument("--overlap", type=int, default=12, help="Overlap size for sliding window")
    parser.add_argument("--target_size", type=int, default=336, help="Target spatial size for inference")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    vid_dir = os.path.join(args.data_dir, f"{args.video_name}.mp4")
    
    if not os.path.exists(vid_dir):
        print(f"[bold red]Error:[/bold red] Video file {vid_dir} not found.")
        exit(1)
        
    # Optimization: Determine best precision
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using mixed precision: [bold cyan]{dtype}[/bold cyan]")
    
    # Load preprocessed video
    video_tensor, (H_orig, W_orig) = read_video_av(vid_dir, args.fps, target_size=args.target_size)
    H_proc, W_proc = video_tensor.shape[-2:]

    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        print(f"Loading VGGT4Track model...")
        vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
        vggt4track_model.eval()
        vggt4track_model = vggt4track_model.to("cuda")

        from models.SpaTrackV2.models.utils import matrix_to_quaternion, quaternion_to_matrix

        chunk_size = args.chunk_size
        overlap = args.overlap
        num_frames = video_tensor.shape[0]

        final_depths = torch.zeros((num_frames, H_orig, W_orig), device="cpu", dtype=torch.float32)
        final_uncs = torch.zeros((num_frames, H_orig, W_orig), device="cpu", dtype=torch.float32)
        final_intrs = torch.zeros((num_frames, 3, 3), device="cpu", dtype=torch.float32)
        final_extrs = torch.zeros((num_frames, 4, 4), device="cpu", dtype=torch.float32)

        start_idx = 0
        is_first_chunk = True

        print(f"Running chunked inference on {num_frames} frames (chunk_size={chunk_size}, overlap={overlap})...")
        while start_idx < num_frames:
            end_idx = min(start_idx + chunk_size, num_frames)
            if not is_first_chunk:
                chunk_start = start_idx - overlap
            else:
                chunk_start = start_idx
                
            chunk_video = video_tensor[chunk_start:end_idx]
            
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
                ratios = None
                D_prev = final_depths[chunk_start : chunk_start + overlap].cuda()
                D_curr = depth_tensor_hd[:overlap]
                valid_mask = (D_prev > 0.1) & (D_prev < 100.0) & (D_curr > 0.1) & (D_curr < 100.0)
                
                prev_global_extrs = final_extrs[chunk_start : chunk_start + overlap].cuda()
                prev_uncs = final_uncs[chunk_start : chunk_start + overlap].cuda()
                curr_uncs = unc_metric_hd[:overlap].cuda()
                
                # Compute average confidence per frame in the overlap region
                conf_prev = []
                conf_curr = []
                for t in range(overlap):
                    mask_t = valid_mask[t]
                    if mask_t.sum() > 10:
                        cp = prev_uncs[t][mask_t].mean()
                        cc = curr_uncs[t][mask_t].mean()
                    else:
                        cp = prev_uncs[t].mean()
                        cc = curr_uncs[t].mean()
                    cp = torch.clamp(cp, min=1e-4)
                    cc = torch.clamp(cc, min=1e-4)
                    conf_prev.append(cp)
                    conf_curr.append(cc)
                    
                conf_prev = torch.stack(conf_prev)
                conf_curr = torch.stack(conf_curr)
                
                # Joint weight for alignment: weight_t = conf_prev_t * conf_curr_t
                align_weights = conf_prev * conf_curr
                align_weights = align_weights / align_weights.sum().clamp(min=1e-6)

                # --- Sim(3) Alignment of Camera Trajectories ---
                # 1. Compute relative rotation average R_M
                M_quats = []
                for t in range(overlap):
                    G_t = prev_global_extrs[t]
                    C_t = extrinsic[t]
                    R_M_t = torch.matmul(G_t[:3, :3].t(), C_t[:3, :3])
                    q_t = matrix_to_quaternion(R_M_t)
                    M_quats.append(q_t)
                    
                M_quats = torch.stack(M_quats, dim=0) # [overlap, 4]
                for t in range(1, overlap):
                    if torch.dot(M_quats[0], M_quats[t]) < 0:
                        M_quats[t] = -M_quats[t]
                        
                w_unsqueezed = align_weights.view(overlap, 1)
                avg_quat = torch.sum(M_quats * w_unsqueezed, dim=0)
                avg_quat = avg_quat / torch.norm(avg_quat).clamp(min=1e-8)
                R_M = quaternion_to_matrix(avg_quat) # [3, 3]

                # 2. Extract camera centers
                c_global = []
                c_local = []
                for t in range(overlap):
                    G_t = prev_global_extrs[t]
                    C_t = extrinsic[t]
                    c_g = -torch.matmul(G_t[:3, :3].t(), G_t[:3, 3])
                    c_l = -torch.matmul(C_t[:3, :3].t(), C_t[:3, 3])
                    c_global.append(c_g)
                    c_local.append(c_l)
                    
                c_global = torch.stack(c_global, dim=0) # [overlap, 3]
                c_local = torch.stack(c_local, dim=0)   # [overlap, 3]
                
                # 3. Rotate local camera centers
                c_rot = torch.matmul(c_local, R_M.t()) # [overlap, 3]
                
                # 4. Compute centroids
                mean_global = c_global.mean(dim=0)
                mean_rot = c_rot.mean(dim=0)
                
                # 5. Centered coordinates
                d_global = c_global - mean_global
                d_rot = c_rot - mean_rot
                
                # 6. Compute robust scale factor s from overlapping depth maps (millions of pixels, highly stable)
                if valid_mask.sum() > 100:
                    ratios = D_prev[valid_mask] / D_curr[valid_mask]
                    s = torch.median(ratios)
                    s = torch.clamp(s, min=0.2, max=5.0)
                else:
                    s = torch.tensor(1.0, device="cuda")
                        
                # 7. Solve for optimal translation t_M
                t_M = mean_global - s * mean_rot
                
                # 8. Scale current chunk's depth maps
                depth_tensor_hd = s * depth_tensor_hd
                
                # 9. Apply similarity transform to all extrinsics in the chunk
                aligned_extrinsic = extrinsic.clone()
                for t in range(chunk_len):
                    R_l = extrinsic[t, :3, :3]
                    t_l = extrinsic[t, :3, 3]
                    
                    R_aligned = torch.matmul(R_l, R_M.t())
                    t_aligned = s * t_l - torch.matmul(R_aligned, t_M)
                    
                    aligned_extrinsic[t, :3, :3] = R_aligned
                    aligned_extrinsic[t, :3, 3] = t_aligned
                
                blend_weights = torch.linspace(0.0, 1.0, steps=overlap, device="cuda")
                envelope = torch.sin(np.pi * blend_weights)
                
                # Relative confidence of current chunk: r_t = cc / (cp + cc)
                r_t = conf_curr / (conf_prev + conf_curr)
                
                # Modulate: w_t = w_linear + alpha * envelope * (r_t - 0.5)
                alpha = 0.5
                guided_weights = torch.clamp(blend_weights + alpha * envelope * (r_t - 0.5), min=0.0, max=1.0)
                
                blended_extrs = []
                for t in range(overlap):
                    w = guided_weights[t]
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
                
                blend_w_3d = guided_weights.view(overlap, 1, 1).cpu()
                prev_depths = final_depths[chunk_start : chunk_start + overlap]
                prev_uncs = final_uncs[chunk_start : chunk_start + overlap]
                prev_intrs = final_intrs[chunk_start : chunk_start + overlap]
                
                blended_depths = (1.0 - blend_w_3d) * prev_depths + blend_w_3d * depth_tensor_hd[:overlap].cpu()
                blended_uncs = (1.0 - blend_w_3d) * prev_uncs + blend_w_3d * unc_metric_hd[:overlap].cpu()
                blended_intrs = (1.0 - guided_weights.view(overlap, 1, 1).cpu()) * prev_intrs + guided_weights.view(overlap, 1, 1).cpu() * intrs_hd[:overlap].cpu()
                
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
                D_prev = D_curr = valid_mask = ratios = s = None
                prev_global_extrs = prev_uncs = curr_uncs = None
                conf_prev = conf_curr = align_weights = None
                M_trans = M_quats = avg_trans = avg_quat = M_rot = M_avg = None
                aligned_extrinsic = blend_weights = envelope = guided_weights = None
                blended_extrs = blend_w_3d = prev_depths = prev_intrs = None
                blended_depths = blended_uncs = blended_intrs = None
            torch.cuda.empty_cache()

        depth_tensor = final_depths.numpy()
        extrs = final_extrs.numpy()
        intrs = final_intrs.numpy()
        unc_metric = final_uncs.numpy()
            
    # Save intermediate data
    out_name = args.out_name if args.out_name else f"{args.video_name}_intermediate.npz"
    out_path = os.path.join(args.data_dir, out_name)
    
    np.savez(out_path, 
             depths=depth_tensor,
             extrinsics=extrs,
             intrinsics=intrs,
             unc_metric=unc_metric)
    
    print(f"Intermediate data saved to [bold green]{out_path}[/bold green] (Resolution: {W_orig}x{H_orig})")
