import os
import gc
import time
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track

def run_benchmark():
    # Experimental dimensions
    resolutions = [252, 336, 518]
    frame_lengths = [50, 100, 200]
    offload_options = [False, True]  # False = Disabled, True = Enabled

    # Device & precision setup matching gui_tool_preprocess_logic.py
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print("=" * 80)
    print("STARTING VGGT4TRACK MEMORY & RUNTIME BENCHMARK")
    print(f"Device: {device} | Precision: {dtype}")
    print("=" * 80)

    results = []

    for size in resolutions:
        for length in frame_lengths:
            for offload in offload_options:
                offload_str = "Enabled" if offload else "Disabled"
                print(f"\nRunning: Size {size}x{size} | Frames {length} | Offload Block: {offload_str}")
                
                # Cleanup before run
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                initial_mem = torch.cuda.memory_allocated() / (1024 ** 2)
                
                try:
                    # 1. Load Model with specified offload/chunking config
                    start_time = time.perf_counter()
                    
                    # Passing offload_block and enable_chunking to from_pretrained
                    model = VGGT4Track.from_pretrained(
                        "Yuxihenry/SpatialTrackerV2_Front", 
                        offload_block=offload,
                        enable_chunking=True
                    )
                    model.eval()
                    model = model.to(device=device, dtype=dtype)
                    
                    # 2. Create Dummy Video Tensor (B, T, C, H, W)
                    # Note: VGGT4Track expects range [0, 1] after division in the frontend logic.
                    # In gui_tool_preprocess_logic.py: predictions = vggt4track_model(video_tensor[None].cuda() / 255.0)
                    # Input resolution here is (size, size)
                    dummy_video = torch.rand(1, length, 3, size, size, device=device, dtype=dtype)
                    
                    torch.cuda.synchronize()
                    
                    # Reset memory statistics right before inference to measure actual peak during execution
                    torch.cuda.reset_peak_memory_stats()
                    inference_start = time.perf_counter()
                    
                    # 3. Perform Inference
                    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                        with torch.no_grad():
                            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                                _ = model(dummy_video)
                                
                    torch.cuda.synchronize()
                    inference_end = time.perf_counter()
                    total_time = inference_end - start_time
                    inference_time = inference_end - inference_start
                    
                    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    
                    print(f"  Inference Time: {inference_time:.3f}s (Total: {total_time:.3f}s)")
                    print(f"  Peak GPU Memory: {peak_mem:.2f} MB")
                    
                    results.append({
                        "size": f"{size}x{size}",
                        "frames": length,
                        "offload_block": offload_str,
                        "status": "Success",
                        "inference_time": f"{inference_time:.3f}s",
                        "total_time": f"{total_time:.3f}s",
                        "peak_mem": f"{peak_mem:.2f} MB"
                    })
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  OOM encountered!")
                        results.append({
                            "size": f"{size}x{size}",
                            "frames": length,
                            "offload_block": offload_str,
                            "status": "OOM",
                            "inference_time": "N/A",
                            "total_time": "N/A",
                            "peak_mem": "N/A"
                        })
                    else:
                        print(f"  Error: {e}")
                        results.append({
                            "size": f"{size}x{size}",
                            "frames": length,
                            "offload_block": offload_str,
                            "status": f"Error: {type(e).__name__}",
                            "inference_time": "N/A",
                            "total_time": "N/A",
                            "peak_mem": "N/A"
                        })
                finally:
                    # Clean up to release VRAM
                    if 'model' in locals():
                        del model
                    if 'dummy_video' in locals():
                        del dummy_video
                    gc.collect()
                    torch.cuda.empty_cache()
                    
    # Display Results in a Markdown Table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    headers = ["Resolution", "Frames", "Offload Block", "Status", "Inference Time", "Peak VRAM"]
    row_format = "{:<12} | {:<8} | {:<15} | {:<10} | {:<15} | {:<12}"
    print(row_format.format(*headers))
    print("-" * 80)
    for r in results:
        print(row_format.format(
            r["size"], r["frames"], r["offload_block"], r["status"], r["inference_time"], r["peak_mem"]
        ))
        
    # Also save report to markdown file
    report_path = "benchmark_memory_results.md"
    with open(report_path, "w") as f:
        f.write("# VGGT4Track Memory Benchmark Report\n\n")
        f.write(f"- **Device**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"- **Precision**: {dtype}\n\n")
        f.write("| Resolution | Frames | Offload Block | Status | Inference Time | Peak VRAM |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for r in results:
            f.write(f"| {r['size']} | {r['frames']} | {r['offload_block']} | {r['status']} | {r['inference_time']} | {r['peak_mem']} |\n")
            
    print(f"\nReport written to {report_path}")

if __name__ == "__main__":
    run_benchmark()
