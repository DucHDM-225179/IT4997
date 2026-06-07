import os
import argparse
import numpy as np
import cv2
from rich import print

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize depth maps from preprocessed NPZ files as a video.")
    parser.add_argument("--input_npz", type=str, required=True, help="Path to the intermediate npz file.")
    parser.add_argument("--output_video", type=str, default=None, help="Path to save the output video file (e.g. depth.mp4).")
    parser.add_argument("--fps", type=int, default=15, help="Frame rate of the output video.")
    parser.add_argument("--colormap", type=str, default="jet", choices=["jet", "viridis", "plasma", "inferno"], help="Colormap to use.")
    parser.add_argument("--use_unc", action="store_true", help="Mask out depth values where uncertainty confidence is low (< 0.5).")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.input_npz):
        print(f"[bold red]Error:[/bold red] Input file {args.input_npz} not found.")
        exit(1)
        
    print(f"Loading intermediate data from [bold cyan]{args.input_npz}[/bold cyan]...")
    data = np.load(args.input_npz, allow_pickle=True)
    
    if "depths" not in data:
        print("[bold red]Error:[/bold red] 'depths' key not found in NPZ file.")
        exit(1)
        
    depths = data["depths"]  # Shape: [T, H, W]
    T, H, W = depths.shape
    print(f"Loaded depth maps with sequence length: [bold green]{T}[/bold green], resolution: [bold green]{W}x{H}[/bold green]")
    
    # Check for uncertainty metric if requested
    unc_mask = None
    if args.use_unc:
        if "unc_metric" in data:
            # Mask out depths where uncertainty is low (< 0.5)
            unc_mask = data["unc_metric"] < 0.5
            print("[bold yellow]Info:[/bold yellow] Masking depth map using uncertainty/confidence threshold.")
        else:
            print("[bold yellow]Warning:[/bold yellow] 'unc_metric' not found in NPZ. Skipping uncertainty masking.")
            
    # Resolve output path
    if args.output_video is None:
        base_name = os.path.splitext(os.path.basename(args.input_npz))[0]
        args.output_video = os.path.join(os.path.dirname(args.input_npz) or ".", f"{base_name}_depth.mp4")
        
    # Set colormap
    colormap_map = {
        "jet": cv2.COLORMAP_JET,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "inferno": cv2.COLORMAP_INFERNO
    }
    cv2_colormap = colormap_map[args.colormap]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, args.fps, (W, H))
    
    print(f"Processing and saving video to [bold green]{args.output_video}[/bold green]...")
    
    for t in range(T):
        frame = depths[t].copy()
        
        # Apply uncertainty masking if enabled
        if unc_mask is not None:
            frame[unc_mask[t]] = 0.0
            
        # Define invalid pixels
        invalid_mask = frame <= 0.01
        
        # Get non-zero depth values for robust normalization (percentile-based)
        valid_depths = frame[~invalid_mask]
        if len(valid_depths) > 0:
            d_min = np.percentile(valid_depths, 2)
            d_max = np.percentile(valid_depths, 98)
            if d_max == d_min:
                d_max += 1e-6
        else:
            d_min, d_max = 0.0, 1.0
            
        # Clip and scale to [0, 255]
        frame_clipped = np.clip(frame, d_min, d_max)
        frame_norm = ((frame_clipped - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
        
        # Apply colormap
        frame_colored = cv2.applyColorMap(frame_norm, cv2_colormap)
        
        # Color invalid pixels as black
        frame_colored[invalid_mask] = 0
        
        out.write(frame_colored)
        
    out.release()
    print(f"[bold green]Success![/bold green] Depth visualization saved to {args.output_video}")

if __name__ == "__main__":
    main()
