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

def print_module_stats(name, module_or_param, indent=2):
    params, size = get_param_stats(module_or_param)
    sp = " " * indent
    print(f"{sp}- {name}:")
    print(f"{sp}  Parameters: {params:,}")
    print(f"{sp}  Size: {size:.2f} MB")
    return params, size

def main():
    print("Loading Predictor from pretrained weights (Yuxihenry/SpatialTrackerV2-Online) to CPU...")
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    model.eval()

    track3d = model.spatrack.Track3D

    print("\n" + "="*60)
    print("SpatialTrackerV2 Backend Module Size Benchmark (TrackRefiner3D)")
    print("="*60)

    # 1. 2D Image Feature Extractor (fnet)
    print("\n1. 2D Image Feature Extractor:")
    fnet_p, fnet_s = print_module_stats("2D Feature Extractor (fnet)", track3d.fnet)

    # 2. 3D Point Embedding & Correlation Modules
    print("\n2. 3D Point Embedding & Correlation Modules:")
    p_rel_pos_mlp, s_rel_pos_mlp = print_module_stats("rel_pos_mlp", track3d.rel_pos_mlp)
    p_rel_pos_glob_mlp, s_rel_pos_glob_mlp = print_module_stats("rel_pos_glob_mlp", track3d.rel_pos_glob_mlp)
    p_corr_xyz_mlp, s_corr_xyz_mlp = print_module_stats("corr_xyz_mlp", track3d.corr_xyz_mlp)
    p_corr_transformer, s_corr_transformer = print_module_stats("corr_transformer", track3d.corr_transformer)
    p_corr_depth_mlp, s_corr_depth_mlp = print_module_stats("corr_depth_mlp", track3d.corr_depth_mlp)
    p_xyz_mlp, s_xyz_mlp = print_module_stats("xyz_mlp", track3d.xyz_mlp)
    p_proj_xyz_embed, s_proj_xyz_embed = print_module_stats("proj_xyz_embed", track3d.proj_xyz_embed)
    
    embed_params = (p_rel_pos_mlp + p_rel_pos_glob_mlp + p_corr_xyz_mlp + 
                    p_corr_transformer + p_corr_depth_mlp + p_xyz_mlp + p_proj_xyz_embed)
    embed_size = (s_rel_pos_mlp + s_rel_pos_glob_mlp + s_corr_xyz_mlp + 
                  s_corr_transformer + s_corr_depth_mlp + s_xyz_mlp + s_proj_xyz_embed)
    print(f"  -> Total 3D Point Embedding & Correlation: {embed_params:,} parameters ({embed_size:.2f} MB)")

    # 3. 2D Correlation / Token Input Processing
    print("\n3. 2D Correlation / Token Processing:")
    corr_mlp_p, corr_mlp_s = print_module_stats("corr_mlp", track3d.corr_mlp)

    # 4. Big Transformer Module (Attention Blob)
    print("\n4. Big Transformer Module (Attention Blob):")
    # Base UpdateFormer components (excluding heads)
    uf = track3d.updateformer
    p_uf_in, s_uf_in = print_module_stats("Base input_transform", uf.input_transform, indent=4)
    p_uf_virt, s_uf_virt = print_module_stats("Base virtual_tracks parameter", uf.virual_tracks if hasattr(uf, 'virual_tracks') else None, indent=4)
    p_uf_time, s_uf_time = print_module_stats("Base time_blocks", uf.time_blocks, indent=4)
    p_uf_space_virt, s_uf_space_virt = print_module_stats("Base space_virtual_blocks", uf.space_virtual_blocks if hasattr(uf, 'space_virtual_blocks') else None, indent=4)
    p_uf_s_p2v, s_uf_s_p2v = print_module_stats("Base space_point2virtual_blocks", uf.space_point2virtual_blocks if hasattr(uf, 'space_point2virtual_blocks') else None, indent=4)
    p_uf_s_v2p, s_uf_s_v2p = print_module_stats("Base space_virtual2point_blocks", uf.space_virtual2point_blocks if hasattr(uf, 'space_virtual2point_blocks') else None, indent=4)

    uf3d = track3d.updateformer3D
    p_uf3d_switch_t, s_uf3d_switch_t = print_module_stats("3D switcher_tokens parameter", uf3d.switcher_tokens if hasattr(uf3d, 'switcher_tokens') else None, indent=4)
    p_uf3d_space_sw, s_uf3d_space_sw = print_module_stats("3D space_switcher_blocks", uf3d.space_switcher_blocks, indent=4)
    p_uf3d_t3d2sw, s_uf3d_t3d2sw = print_module_stats("3D space_track3d2switcher_blocks", uf3d.space_track3d2switcher_blocks, indent=4)
    p_uf3d_sw2t3d, s_uf3d_sw2t3d = print_module_stats("3D space_switcher2track3d_blocks", uf3d.space_switcher2track3d_blocks, indent=4)
    p_uf3d_v2sw, s_uf3d_v2sw = print_module_stats("3D space_virtual2switcher_blocks", uf3d.space_virtual2switcher_blocks, indent=4)
    p_uf3d_sw2v, s_uf3d_sw2v = print_module_stats("3D space_switcher2virtual_blocks", uf3d.space_switcher2virtual_blocks, indent=4)
    p_uf3d_time_new, s_uf3d_time_new = print_module_stats("3D time_blocks_new", uf3d.time_blocks_new, indent=4)
    p_uf3d_ss_cross, s_uf3d_ss_cross = print_module_stats("3D scale_shift_cross_attn", uf3d.scale_shift_cross_attn, indent=4)
    p_uf3d_ss_self, s_uf3d_ss_self = print_module_stats("3D scale_shift_self_attn", uf3d.scale_shift_self_attn, indent=4)
    p_uf3d_dr_cross, s_uf3d_dr_cross = print_module_stats("3D dense_res_cross_attn", uf3d.dense_res_cross_attn, indent=4)
    p_uf3d_dr_self, s_uf3d_dr_self = print_module_stats("3D dense_res_self_attn", uf3d.dense_res_self_attn, indent=4)

    blob_params = (p_uf_in + p_uf_virt + p_uf_time + p_uf_space_virt + p_uf_s_p2v + p_uf_s_v2p +
                   p_uf3d_switch_t + p_uf3d_space_sw + p_uf3d_t3d2sw + p_uf3d_sw2t3d + p_uf3d_v2sw + p_uf3d_sw2v +
                   p_uf3d_time_new + p_uf3d_ss_cross + p_uf3d_ss_self + p_uf3d_dr_cross + p_uf3d_dr_self)
    blob_size = (s_uf_in + s_uf_virt + s_uf_time + s_uf_space_virt + s_uf_s_p2v + s_uf_s_v2p +
                 s_uf3d_switch_t + s_uf3d_space_sw + s_uf3d_t3d2sw + s_uf3d_sw2t3d + s_uf3d_v2sw + s_uf3d_sw2v +
                 s_uf3d_time_new + s_uf3d_ss_cross + s_uf3d_ss_self + s_uf3d_dr_cross + s_uf3d_dr_self)
    print(f"  -> Total Transformer Attention Blob: {blob_params:,} parameters ({blob_size:.2f} MB)")

    # 5. Decoder Heads
    print("\n5. Decoder Heads:")
    p_uf_flow, s_uf_flow = print_module_stats("Base flow_head", uf.flow_head, indent=4)
    p_uf_vis, s_uf_vis = print_module_stats("Base vis_conf_head", uf.vis_conf_head if hasattr(uf, 'vis_conf_head') else None, indent=4)
    
    p_uf3d_pt, s_uf3d_pt = print_module_stats("3D point_head", getattr(uf3d, 'point_head', None), indent=4)
    p_uf3d_dp, s_uf3d_dp = print_module_stats("3D depth_head", getattr(uf3d, 'depth_head', None), indent=4)
    p_uf3d_pro, s_uf3d_pro = print_module_stats("3D pro_analysis_w_head", uf3d.pro_analysis_w_head, indent=4)
    p_uf3d_vis, s_uf3d_vis = print_module_stats("3D vis_conf_head", uf3d.vis_conf_head, indent=4)
    p_uf3d_res, s_uf3d_res = print_module_stats("3D residual_head", uf3d.residual_head, indent=4)
    p_uf3d_ss_dec, s_uf3d_ss_dec = print_module_stats("3D scale_shift_dec", uf3d.scale_shift_dec, indent=4)
    p_uf3d_dr_dec, s_uf3d_dr_dec = print_module_stats("3D dense_res_dec", uf3d.dense_res_dec, indent=4)

    dec_params = (p_uf_flow + p_uf_vis + p_uf3d_pt + p_uf3d_dp + p_uf3d_pro + p_uf3d_vis + p_uf3d_res + p_uf3d_ss_dec + p_uf3d_dr_dec)
    dec_size = (s_uf_flow + s_uf_vis + s_uf3d_pt + s_uf3d_dp + s_uf3d_pro + s_uf3d_vis + s_uf3d_res + s_uf3d_ss_dec + s_uf3d_dr_dec)
    print(f"  -> Total Decoder Heads: {dec_params:,} parameters ({dec_size:.2f} MB)")

    # 6. Verification of Sum
    print("\n" + "="*60)
    print("Verification & Summary")
    print("="*60)
    
    total_spatrack_p, total_spatrack_s = get_param_stats(model.spatrack)
    total_track3d_p, total_track3d_s = get_param_stats(track3d)
    
    summed_p = fnet_p + embed_params + corr_mlp_p + blob_params + dec_params
    summed_s = fnet_s + embed_size + corr_mlp_s + blob_size + dec_size
    
    print(f"Total Predictor Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Total SpaTrack2 Parameters: {total_spatrack_p:,} ({total_spatrack_s:.2f} MB)")
    print(f"Total TrackRefiner3D (Track3D) Parameters: {total_track3d_p:,} ({total_track3d_s:.2f} MB)")
    print(f"Sum of Grouped Parameters: {summed_p:,} ({summed_s:.2f} MB)")
    print(f"Difference: {total_track3d_p - summed_p:,} parameters")
    print("="*60)

if __name__ == "__main__":
    main()
