import torch
import torch.nn as nn
import sys
import os

from models.SpaTrackV2.models.predictor import Predictor

def get_param_stats(module_or_param):
    if module_or_param is None:
        return 0, 0.0
    if isinstance(module_or_param, nn.Parameter):
        num_params = module_or_param.numel()
        total_bytes = num_params * module_or_param.element_size()
    else:
        num_params = sum(p.numel() for p in module_or_param.parameters())
        total_bytes = sum(p.numel() * p.element_size() for p in module_or_param.parameters())
    size_mb = total_bytes / (1024 * 1024)
    return num_params, size_mb

def get_module_stats_str(name, module_or_param):
    params, size = get_param_stats(module_or_param)
    return params, size, f"| {name} | {params:,} | {size:.2f} MB |"

def main():
    print("Loading Predictor from pretrained weights (Yuxihenry/SpatialTrackerV2-Online) to CPU...")
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    model.eval()

    track3d = model.spatrack.Track3D

    report_path = "benchmark_backend_module_size_results.md"
    print(f"Writing backend module size benchmark report to {report_path}...")

    with open(report_path, "w") as f:
        f.write("# SpatialTrackerV2 Backend Module Size Benchmark (TrackRefiner3D)\n\n")
        f.write("- **Model Reference**: `Yuxihenry/SpatialTrackerV2-Online`\n")
        f.write("- **Data Type Precision**: `float32` (4 bytes per parameter)\n\n")

        # 1. 2D Image Feature Extractor
        f.write("## 1. 2D Image Feature Extractor\n\n")
        f.write("| Module | Parameters | Size (FP32) |\n")
        f.write("| --- | --- | --- |\n")
        fnet_p, fnet_s, line = get_module_stats_str("2D Feature Extractor (fnet)", track3d.fnet)
        f.write(line + "\n\n")

        # 2. 3D Point Embedding & Correlation Modules
        f.write("## 2. 3D Point Embedding & Correlation Modules\n\n")
        f.write("| Module | Parameters | Size (FP32) |\n")
        f.write("| --- | --- | --- |\n")
        
        p_rel_pos_mlp, s_rel_pos_mlp, line = get_module_stats_str("rel_pos_mlp", track3d.rel_pos_mlp)
        f.write(line + "\n")
        p_rel_pos_glob_mlp, s_rel_pos_glob_mlp, line = get_module_stats_str("rel_pos_glob_mlp", track3d.rel_pos_glob_mlp)
        f.write(line + "\n")
        p_corr_xyz_mlp, s_corr_xyz_mlp, line = get_module_stats_str("corr_xyz_mlp", track3d.corr_xyz_mlp)
        f.write(line + "\n")
        p_corr_transformer, s_corr_transformer, line = get_module_stats_str("corr_transformer", track3d.corr_transformer)
        f.write(line + "\n")
        p_corr_depth_mlp, s_corr_depth_mlp, line = get_module_stats_str("corr_depth_mlp", track3d.corr_depth_mlp)
        f.write(line + "\n")
        p_xyz_mlp, s_xyz_mlp, line = get_module_stats_str("xyz_mlp", track3d.xyz_mlp)
        f.write(line + "\n")
        p_proj_xyz_embed, s_proj_xyz_embed, line = get_module_stats_str("proj_xyz_embed", track3d.proj_xyz_embed)
        f.write(line + "\n")

        embed_params = (p_rel_pos_mlp + p_rel_pos_glob_mlp + p_corr_xyz_mlp + 
                        p_corr_transformer + p_corr_depth_mlp + p_xyz_mlp + p_proj_xyz_embed)
        embed_size = (s_rel_pos_mlp + s_rel_pos_glob_mlp + s_corr_xyz_mlp + 
                      s_corr_transformer + s_corr_depth_mlp + s_xyz_mlp + s_proj_xyz_embed)
        f.write(f"| **Total 3D Point Embedding & Correlation** | **{embed_params:,}** | **{embed_size:.2f} MB** |\n\n")

        # 3. 2D Correlation / Token Input Processing
        f.write("## 3. 2D Correlation / Token Processing\n\n")
        f.write("| Module | Parameters | Size (FP32) |\n")
        f.write("| --- | --- | --- |\n")
        corr_mlp_p, corr_mlp_s, line = get_module_stats_str("corr_mlp", track3d.corr_mlp)
        f.write(line + "\n\n")

        # 4. Big Transformer Module (Attention Blob)
        f.write("## 4. Big Transformer Module (Attention Blob)\n\n")
        f.write("| Module | Parameters | Size (FP32) |\n")
        f.write("| --- | --- | --- |\n")
        
        uf = track3d.updateformer
        p_uf_in, s_uf_in, line = get_module_stats_str("Base input_transform", uf.input_transform)
        f.write(line + "\n")
        p_uf_virt, s_uf_virt, line = get_module_stats_str("Base virtual_tracks parameter", uf.virual_tracks if hasattr(uf, 'virual_tracks') else None)
        f.write(line + "\n")
        p_uf_time, s_uf_time, line = get_module_stats_str("Base time_blocks", uf.time_blocks)
        f.write(line + "\n")
        p_uf_space_virt, s_uf_space_virt, line = get_module_stats_str("Base space_virtual_blocks", uf.space_virtual_blocks if hasattr(uf, 'space_virtual_blocks') else None)
        f.write(line + "\n")
        p_uf_s_p2v, s_uf_s_p2v, line = get_module_stats_str("Base space_point2virtual_blocks", uf.space_point2virtual_blocks if hasattr(uf, 'space_point2virtual_blocks') else None)
        f.write(line + "\n")
        p_uf_s_v2p, s_uf_s_v2p, line = get_module_stats_str("Base space_virtual2point_blocks", uf.space_virtual2point_blocks if hasattr(uf, 'space_virtual2point_blocks') else None)
        f.write(line + "\n")

        uf3d = track3d.updateformer3D
        p_uf3d_switch_t, s_uf3d_switch_t, line = get_module_stats_str("3D switcher_tokens parameter", uf3d.switcher_tokens if hasattr(uf3d, 'switcher_tokens') else None)
        f.write(line + "\n")
        p_uf3d_space_sw, s_uf3d_space_sw, line = get_module_stats_str("3D space_switcher_blocks", uf3d.space_switcher_blocks)
        f.write(line + "\n")
        p_uf3d_t3d2sw, s_uf3d_t3d2sw, line = get_module_stats_str("3D space_track3d2switcher_blocks", uf3d.space_track3d2switcher_blocks)
        f.write(line + "\n")
        p_uf3d_sw2t3d, s_uf3d_sw2t3d, line = get_module_stats_str("3D space_switcher2track3d_blocks", uf3d.space_switcher2track3d_blocks)
        f.write(line + "\n")
        p_uf3d_v2sw, s_uf3d_v2sw, line = get_module_stats_str("3D space_virtual2switcher_blocks", uf3d.space_virtual2switcher_blocks)
        f.write(line + "\n")
        p_uf3d_sw2v, s_uf3d_sw2v, line = get_module_stats_str("3D space_switcher2virtual_blocks", uf3d.space_switcher2virtual_blocks)
        f.write(line + "\n")
        p_uf3d_time_new, s_uf3d_time_new, line = get_module_stats_str("3D time_blocks_new", uf3d.time_blocks_new)
        f.write(line + "\n")
        p_uf3d_ss_cross, s_uf3d_ss_cross, line = get_module_stats_str("3D scale_shift_cross_attn", uf3d.scale_shift_cross_attn)
        f.write(line + "\n")
        p_uf3d_ss_self, s_uf3d_ss_self, line = get_module_stats_str("3D scale_shift_self_attn", uf3d.scale_shift_self_attn)
        f.write(line + "\n")
        p_uf3d_dr_cross, s_uf3d_dr_cross, line = get_module_stats_str("3D dense_res_cross_attn", uf3d.dense_res_cross_attn)
        f.write(line + "\n")
        p_uf3d_dr_self, s_uf3d_dr_self, line = get_module_stats_str("3D dense_res_self_attn", uf3d.dense_res_self_attn)
        f.write(line + "\n")

        blob_params = (p_uf_in + p_uf_virt + p_uf_time + p_uf_space_virt + p_uf_s_p2v + p_uf_s_v2p +
                       p_uf3d_switch_t + p_uf3d_space_sw + p_uf3d_t3d2sw + p_uf3d_sw2t3d + p_uf3d_v2sw + p_uf3d_sw2v +
                       p_uf3d_time_new + p_uf3d_ss_cross + p_uf3d_ss_self + p_uf3d_dr_cross + p_uf3d_dr_self)
        blob_size = (s_uf_in + s_uf_virt + s_uf_time + s_uf_space_virt + s_uf_s_p2v + s_uf_s_v2p +
                     s_uf3d_switch_t + s_uf3d_space_sw + s_uf3d_t3d2sw + s_uf3d_sw2t3d + s_uf3d_v2sw + s_uf3d_sw2v +
                     s_uf3d_time_new + s_uf3d_ss_cross + s_uf3d_ss_self + s_uf3d_dr_cross + s_uf3d_dr_self)
        f.write(f"| **Total Transformer Attention Blob** | **{blob_params:,}** | **{blob_size:.2f} MB** |\n\n")

        # 5. Decoder Heads
        f.write("## 5. Decoder Heads\n\n")
        f.write("| Module | Parameters | Size (FP32) |\n")
        f.write("| --- | --- | --- |\n")
        
        p_uf_flow, s_uf_flow, line = get_module_stats_str("Base flow_head", uf.flow_head)
        f.write(line + "\n")
        p_uf_vis, s_uf_vis, line = get_module_stats_str("Base vis_conf_head", uf.vis_conf_head if hasattr(uf, 'vis_conf_head') else None)
        f.write(line + "\n")
        p_uf3d_pt, s_uf3d_pt, line = get_module_stats_str("3D point_head", getattr(uf3d, 'point_head', None))
        f.write(line + "\n")
        p_uf3d_dp, s_uf3d_dp, line = get_module_stats_str("3D depth_head", getattr(uf3d, 'depth_head', None))
        f.write(line + "\n")
        p_uf3d_pro, s_uf3d_pro, line = get_module_stats_str("3D pro_analysis_w_head", uf3d.pro_analysis_w_head)
        f.write(line + "\n")
        p_uf3d_vis, s_uf3d_vis, line = get_module_stats_str("3D vis_conf_head", uf3d.vis_conf_head)
        f.write(line + "\n")
        p_uf3d_res, s_uf3d_res, line = get_module_stats_str("3D residual_head", uf3d.residual_head)
        f.write(line + "\n")
        p_uf3d_ss_dec, s_uf3d_ss_dec, line = get_module_stats_str("3D scale_shift_dec", uf3d.scale_shift_dec)
        f.write(line + "\n")
        p_uf3d_dr_dec, s_uf3d_dr_dec, line = get_module_stats_str("3D dense_res_dec", uf3d.dense_res_dec)
        f.write(line + "\n")

        dec_params = (p_uf_flow + p_uf_vis + p_uf3d_pt + p_uf3d_dp + p_uf3d_pro + p_uf3d_vis + p_uf3d_res + p_uf3d_ss_dec + p_uf3d_dr_dec)
        dec_size = (s_uf_flow + s_uf_vis + s_uf3d_pt + s_uf3d_dp + s_uf3d_pro + s_uf3d_vis + s_uf3d_res + s_uf3d_ss_dec + s_uf3d_dr_dec)
        f.write(f"| **Total Decoder Heads** | **{dec_params:,}** | **{dec_size:.2f} MB** |\n\n")

        # 6. Verification & Summary
        f.write("## 6. Verification & Summary\n\n")
        
        total_spatrack_p, total_spatrack_s = get_param_stats(model.spatrack)
        total_track3d_p, total_track3d_s = get_param_stats(track3d)
        
        summed_p = fnet_p + embed_params + corr_mlp_p + blob_params + dec_params
        summed_s = fnet_s + embed_size + corr_mlp_s + blob_size + dec_size
        
        f.write(f"- **Total Predictor Parameters**: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"- **Total SpaTrack2 Parameters**: {total_spatrack_p:,} ({total_spatrack_s:.2f} MB)\n")
        f.write(f"- **Total TrackRefiner3D (Track3D) Parameters**: {total_track3d_p:,} ({total_track3d_s:.2f} MB)\n")
        f.write(f"- **Sum of Grouped Parameters**: {summed_p:,} ({summed_s:.2f} MB)\n")
        f.write(f"- **Difference**: {total_track3d_p - summed_p:,} parameters\n")

    print(f"Backend module report generated successfully.")

if __name__ == "__main__":
    main()
