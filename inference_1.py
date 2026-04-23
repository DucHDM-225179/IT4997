import os
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import av
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from rich import print

def read_video_av(vid_path, step):
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
            # This resizes/crops to 518x518 by default
            img_proc = preprocess_image(img_tensor)
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
    video_tensor, (H_orig, W_orig) = read_video_av(vid_dir, args.fps)
    H_proc, W_proc = video_tensor.shape[-2:]
    
    print(f"Loading VGGT4Track model...")
    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    vggt4track_model.eval()
    vggt4track_model = vggt4track_model.to("cuda")

    print(f"Running inference on {video_tensor.shape[0]} frames...")
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            predictions = vggt4track_model(video_tensor[None].cuda() / 255.0)
            extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
            depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]
    
            # Post-processing: Resize back to original resolution as the model does internally
            # Correcting dimensions for F.interpolate: [S, 1, H, W] to ensure 2D interpolation
            depth_tensor_hd = F.interpolate(depth_map[:, None], size=(H_orig, W_orig), mode='bilinear', align_corners=True).squeeze(1)
            unc_metric_hd = F.interpolate(depth_conf[:, None], size=(H_orig, W_orig), mode='bilinear', align_corners=True).squeeze(1)
            
            # Scale intrinsics back to original resolution
            intrs_hd = intrinsic.clone()
            intrs_hd[..., 0, :] *= W_orig / W_proc
            intrs_hd[..., 1, :] *= H_orig / H_proc

            # Extract to CPU
            depth_tensor = depth_tensor_hd.cpu().numpy()
            extrs = extrinsic.squeeze().cpu().numpy()
            intrs = intrs_hd.squeeze().cpu().numpy()
            unc_metric = unc_metric_hd.cpu().numpy()
            
    # Save intermediate data
    out_name = args.out_name if args.out_name else f"{args.video_name}_intermediate.npz"
    out_path = os.path.join(args.data_dir, out_name)
    
    np.savez(out_path, 
             depths=depth_tensor,
             extrinsics=extrs,
             intrinsics=intrs,
             unc_metric=unc_metric)
    
    print(f"Intermediate data saved to [bold green]{out_path}[/bold green] (Resolution: {W_orig}x{H_orig})")
