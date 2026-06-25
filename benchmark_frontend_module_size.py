import torch
import torch.nn as nn
import sys

# Ensure models/SpaTrackV2, models/SpaTrackV2/models/vggt4track, and workspace are in python path
import os


from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track

def get_param_stats(module):
    if module is None:
        return 0, 0.0
    num_params = sum(p.numel() for p in module.parameters())
    # Size in MB (assuming float32: 4 bytes per param, but let's count actual parameter bytes if possible, or standard float32 size)
    # Let's calculate based on actual element sizes
    total_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    size_mb = total_bytes / (1024 * 1024)
    return num_params, size_mb

def main():
    print("Loading VGGT4Track from pretrained weights (Yuxihenry/SpatialTrackerV2_Front) to CPU...")
    # This will load and fetch pretrained weights using PyTorchModelHubMixin
    model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    model.eval()

    print("\n" + "="*50)
    print("VGGT4Track Module Parameters & Size Benchmarks")
    print("="*50)

    # 1. Camera Head
    cam_params, cam_size = get_param_stats(model.camera_head)
    print(f"Camera Head (camera_head):")
    print(f"  - Parameters: {cam_params:,}")
    print(f"  - Size: {cam_size:.2f} MB")

    # 2. Depth Head
    depth_params, depth_size = get_param_stats(model.depth_head)
    print(f"\nDepth Head (depth_head):")
    print(f"  - Parameters: {depth_params:,}")
    print(f"  - Size: {depth_size:.2f} MB")

    # 3. Track Head (in VGGT or VGGT4Track if present)
    track_head = getattr(model, 'track_head', None)
    if track_head is not None:
        track_params, track_size = get_param_stats(track_head)
        print(f"\nTrack Head (track_head):")
        print(f"  - Parameters: {track_params:,}")
        print(f"  - Size: {track_size:.2f} MB")
    else:
        print("\nTrack Head (track_head): Not present in VGGT4Track.")

    # 4. Point Head (in VGGT or VGGT4Track if present)
    point_head = getattr(model, 'point_head', None)
    if point_head is not None:
        point_params, point_size = get_param_stats(point_head)
        print(f"\nPoint Head (point_head):")
        print(f"  - Parameters: {point_params:,}")
        print(f"  - Size: {point_size:.2f} MB")
    else:
        print("\nPoint Head (point_head): Not present in VGGT4Track.")

    # 5. Aggregator
    agg = model.aggregator
    agg_params, agg_size = get_param_stats(agg)
    print(f"\nAggregator (aggregator) Total:")
    print(f"  - Parameters: {agg_params:,}")
    print(f"  - Size: {agg_size:.2f} MB")

    # DinoV2 inside aggregator (patch_embed)
    pe_params, pe_size = get_param_stats(agg.patch_embed)
    print(f"  -> Vision Transformer / DinoV2 (patch_embed):")
    print(f"     - Parameters: {pe_params:,}")
    print(f"     - Size: {pe_size:.2f} MB")

    # Frame Blocks
    fb_params, fb_size = get_param_stats(agg.frame_blocks)
    print(f"  -> Frame Blocks (frame_blocks):")
    print(f"     - Blocks Count: {len(agg.frame_blocks)}")
    print(f"     - Parameters (Total): {fb_params:,}")
    print(f"     - Size (Total): {fb_size:.2f} MB")
    if len(agg.frame_blocks) > 0:
        single_fb_params, single_fb_size = get_param_stats(agg.frame_blocks[0])
        print(f"     - Per-block Parameters: {single_fb_params:,}")
        print(f"     - Per-block Size: {single_fb_size:.2f} MB")

    # Global Blocks
    gb_params, gb_size = get_param_stats(agg.global_blocks)
    print(f"  -> Global Blocks (global_blocks):")
    print(f"     - Blocks Count: {len(agg.global_blocks)}")
    print(f"     - Parameters (Total): {gb_params:,}")
    print(f"     - Size (Total): {gb_size:.2f} MB")
    if len(agg.global_blocks) > 0:
        single_gb_params, single_gb_size = get_param_stats(agg.global_blocks[0])
        print(f"     - Per-block Parameters: {single_gb_params:,}")
        print(f"     - Per-block Size: {single_gb_size:.2f} MB")

    print("\n" + "="*50)
    print("Total model statistics:")
    total_params, total_size = get_param_stats(model)
    print(f"  - Total Parameters: {total_params:,}")
    print(f"  - Total Size: {total_size:.2f} MB")
    print("="*50)

if __name__ == "__main__":
    main()
