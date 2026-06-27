import torch
import torch.nn as nn
import sys
import os

from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track

def get_param_stats(module):
    if module is None:
        return 0, 0.0
    num_params = sum(p.numel() for p in module.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    size_mb = total_bytes / (1024 * 1024)
    return num_params, size_mb

def get_module_stats_str(name, module):
    params, size = get_param_stats(module)
    return params, size, f"| {name} | {params:,} | {size:.2f} MB |"

def main():
    print("Loading VGGT4Track from pretrained weights (Yuxihenry/SpatialTrackerV2_Front) to CPU...")
    model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    model.eval()

    report_path = "benchmark_frontend_module_size_results.md"
    print(f"Writing frontend module size benchmark report to {report_path}...")

    with open(report_path, "w") as f:
        f.write("# VGGT4Track Module Parameters & Size Benchmarks\n\n")
        f.write("- **Model Reference**: `Yuxihenry/SpatialTrackerV2_Front`\n")
        f.write("- **Data Type Precision**: `float32` (4 bytes per parameter)\n\n")

        # 1. Component Breakdown
        f.write("## Component Breakdown\n\n")
        f.write("| Module Component | Parameters | Size (FP32) |\n")
        f.write("| --- | --- | --- |\n")

        # Camera Head
        cam_params, cam_size, line = get_module_stats_str("Camera Head (camera_head)", model.camera_head)
        f.write(line + "\n")

        # Depth Head
        depth_params, depth_size, line = get_module_stats_str("Depth Head (depth_head)", model.depth_head)
        f.write(line + "\n")

        # Track Head
        track_head = getattr(model, 'track_head', None)
        if track_head is not None:
            _, _, line = get_module_stats_str("Track Head (track_head)", track_head)
            f.write(line + "\n")

        # Point Head
        point_head = getattr(model, 'point_head', None)
        if point_head is not None:
            _, _, line = get_module_stats_str("Point Head (point_head)", point_head)
            f.write(line + "\n")

        # Aggregator Total
        agg = model.aggregator
        agg_params, agg_size, line = get_module_stats_str("Aggregator (aggregator) Total", agg)
        f.write(line + "\n")

        # DinoV2 patch embed
        pe_params, pe_size, line = get_module_stats_str("  -> Vision Transformer / DinoV2 (patch_embed)", agg.patch_embed)
        f.write(line + "\n")

        # Frame Blocks
        fb_params, fb_size, line = get_module_stats_str("  -> Frame Blocks (frame_blocks)", agg.frame_blocks)
        f.write(line + "\n")

        # Global Blocks
        gb_params, gb_size, line = get_module_stats_str("  -> Global Blocks (global_blocks)", agg.global_blocks)
        f.write(line + "\n")

        f.write("\n")

        # 2. Block Detailed View
        f.write("## Block Detailed View\n\n")
        f.write("| Block Type | Count | Per-Block Params | Per-Block Size (FP32) |\n")
        f.write("| --- | --- | --- | --- |\n")
        
        if len(agg.frame_blocks) > 0:
            single_fb_params, single_fb_size = get_param_stats(agg.frame_blocks[0])
            f.write(f"| Frame Blocks | {len(agg.frame_blocks)} | {single_fb_params:,} | {single_fb_size:.2f} MB |\n")
        if len(agg.global_blocks) > 0:
            single_gb_params, single_gb_size = get_param_stats(agg.global_blocks[0])
            f.write(f"| Global Blocks | {len(agg.global_blocks)} | {single_gb_params:,} | {single_gb_size:.2f} MB |\n")
        
        f.write("\n")

        # 3. Model Total Summary
        f.write("## Model Summary\n\n")
        total_params, total_size = get_param_stats(model)
        f.write(f"- **Total Model Parameters**: {total_params:,}\n")
        f.write(f"- **Total Model Size (FP32)**: {total_size:.2f} MB\n")

    print(f"Frontend module report generated successfully.")

if __name__ == "__main__":
    main()
