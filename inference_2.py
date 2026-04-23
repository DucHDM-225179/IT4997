import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import argparse
import cv2
import av
from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.utils.visualizer import Visualizer
from models.SpaTrackV2.models.utils import get_points_on_a_grid
from models.monoD.depth_anything_v2.util.transform import Resize
from rich import print

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_mode", type=str, default="online")
    parser.add_argument("--data_dir", type=str, default="examples")
    parser.add_argument("--input_npz", type=str, required=True, help="Path to intermediate npz file from inference_1.py")
    parser.add_argument("--video_name", type=str, default="drifting", help="Used for mask and output naming")
    parser.add_argument("--fps", type=int, default=1, help="Should match fps used in inference_1.py")
    parser.add_argument("--grid_size", type=int, default=30)
    parser.add_argument("--vo_points", type=int, default=120)
    return parser.parse_args()

def read_video_av(vid_path, step, target_size=518):
    container = av.open(vid_path)
    video_stream = container.streams.video[0]
    H_orig, W_orig = video_stream.height, video_stream.width
    
    # Use the model's preferred resizing logic (from normalize_rgb / ProcVid)
    resizer = Resize(
        width=target_size,
        height=target_size,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
    )
    new_W, new_H = resizer.get_size(W_orig, H_orig)
    
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i % step == 0:
            img = frame.to_rgb().to_ndarray()
            # Resize per frame to save memory
            img_resized = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_CUBIC)
            frames.append(img_resized)
            if len(frames) >= 40:
                break
            
    container.close()
    video_np = np.stack(frames) 
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float()
    return video_tensor, (H_orig, W_orig)

if __name__ == "__main__":
    args = parse_args()
    out_dir = os.path.join(args.data_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    
    vid_path = os.path.join(args.data_dir, f"{args.video_name}.mp4")
    mask_path = os.path.join(args.data_dir, f"{args.video_name}.png")
    
    if not os.path.exists(vid_path):
        print(f"[bold red]Error:[/bold red] Video file {vid_path} not found.")
        exit(1)
        
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using mixed precision: [bold cyan]{dtype}[/bold cyan]")

    # Load and preprocess video to save memory
    print(f"Loading and resizing video from {vid_path} at {args.fps} fps...")
    video_tensor, (H_orig, W_orig) = read_video_av(vid_path, args.fps)
    
    # Load intermediate data
    print(f"Loading intermediate data from {args.input_npz}...")
    data = np.load(args.input_npz, allow_pickle=True)
    depth_in = data["depths"]
    intrs_in = data["intrinsics"]
    extrs_in = data["extrinsics"]
    unc_metric_in = data["unc_metric"]
    
    # Resize intermediate data to match the resized video_tensor to save memory
    print(f"Resizing intermediate data to match video resolution...")
    depth_tensor = F.interpolate(torch.from_numpy(depth_in)[:, None], 
                                 size=video_tensor.shape[2:], 
                                 mode='bilinear', align_corners=True).squeeze(1).numpy()
    unc_metric = F.interpolate(torch.from_numpy(unc_metric_in)[:, None].float(), 
                                size=video_tensor.shape[2:], 
                                mode='bilinear', align_corners=True).squeeze(1).numpy() > 0.5
    
    # Intrinsics are already correct if they were saved in HD and Predictor handles rescaling
    # but since we are passing resized video, we should pass scaled intrinsics.
    # Predictor.forward will scale them again if video is resized, so we pass them matching video_tensor.
    scale_w = video_tensor.shape[3] / W_orig
    scale_h = video_tensor.shape[2] / H_orig
    intrs_in[:, 0, :] *= scale_w
    intrs_in[:, 1, :] *= scale_h

    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (video_tensor.shape[3], video_tensor.shape[2]))
        mask = mask.sum(axis=-1) > 0
    else:
        mask = np.ones((video_tensor.shape[2], video_tensor.shape[3]), dtype=bool)

    # Load Predictor
    print(f"Loading Predictor model ({args.track_mode} mode)...")
    if args.track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")

    model.spatrack.track_num = args.vo_points
    model.eval()
    model.to("cuda")
    
    if hasattr(model.spatrack, "base_model") and model.spatrack.base_model is not None:
        model.spatrack.base_model.to("cpu")
        torch.cuda.empty_cache()

    viser = Visualizer(save_dir=out_dir, grayscale=True, 
                     fps=10, pad_value=0, tracks_leave_trace=5)
    
    frame_H, frame_W = video_tensor.shape[2:]
    grid_pts = get_points_on_a_grid(args.grid_size, (frame_H, frame_W), device="cpu")
    
    if os.path.exists(mask_path):
        grid_pts_int = grid_pts[0].long()
        mask_values = mask[grid_pts_int[...,1], grid_pts_int[...,0]]
        grid_pts = grid_pts[:, mask_values]
    
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()

    print(f"Running Predictor...")
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            (
                c2w_traj, intrs, point_map, conf_depth,
                track3d_pred, track2d_pred, vis_pred, conf_pred, video
            ) = model.forward(video_tensor, depth=depth_tensor,
                                intrs=intrs_in, extrs=extrs_in, 
                                queries=query_xyt,
                                fps=1, full_point=False, iters_track=4,
                                query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
                                support_frame=len(video_tensor)-1, replace_ratio=0.2) 
        
        # Post-process for saving: resize back to original resolution as requested
        print(f"Post-processing results back to original resolution {W_orig}x{H_orig}...")
        
        video_hd = F.interpolate(video, size=(H_orig, W_orig), mode='bilinear', align_corners=True)
        video_tensor_hd = F.interpolate(video_tensor, size=(H_orig, W_orig), mode='bilinear', align_corners=True)
        point_map_hd = F.interpolate(point_map, size=(H_orig, W_orig), mode='bilinear', align_corners=True)
        conf_depth_hd = F.interpolate(conf_depth[:, None], size=(H_orig, W_orig), mode='bilinear', align_corners=True).squeeze(1)
        
        # Scale 2D tracks and intrinsics back to HD
        track2d_pred[..., 0] *= W_orig / video_tensor.shape[3]
        track2d_pred[..., 1] *= H_orig / video_tensor.shape[2]
        intrs[:, 0, :] *= W_orig / video_tensor.shape[3]
        intrs[:, 1, :] *= H_orig / video_tensor.shape[2]

        # Save results
        final_results = {}
        final_results["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
        final_results["tracks_2d"] = track2d_pred.cpu().numpy()
        final_results["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
        final_results["intrinsics"] = intrs.cpu().numpy()
        depth_save = point_map_hd[:,2,...]
        depth_save[conf_depth_hd<0.5] = 0
        final_results["depths"] = depth_save.cpu().numpy()
        final_results["visibs"] = vis_pred.cpu().numpy()
        final_results["unc_metric"] = conf_depth_hd.cpu().numpy()
        
        out_path = os.path.join(out_dir, f'{args.video_name}_result.npz')
        np.savez(out_path, **final_results)

        print(f"Results saved to [bold green]{out_path}[/bold green]")
