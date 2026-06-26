# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from models.SpaTrackV2.models.vggt4track.models.aggregator import Aggregator
from models.SpaTrackV2.models.vggt4track.heads.camera_head import CameraHead
from models.SpaTrackV2.models.vggt4track.heads.dpt_head import DPTHead
from models.SpaTrackV2.models.vggt4track.heads.track_head import TrackHead
from models.SpaTrackV2.models.vggt4track.utils.loss import compute_loss
from models.SpaTrackV2.models.vggt4track.utils.pose_enc import pose_encoding_to_extri_intri
from models.SpaTrackV2.models.tracker3D.spatrack_modules.utils import depth_to_points_colmap, get_nth_visible_time_index
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from einops import rearrange
import torch.nn.functional as F

class VGGT4Track(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, offload_block=True, enable_chunking=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            offload_blocks=offload_block, enable_chunking=enable_chunking)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="sigmoid")

    def forward(
        self,
        images: torch.Tensor,
        annots = {},
        fx_prev = None,
        fy_prev = None,
        **kwargs):
        """
        Forward pass of the VGGT4Track model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """

        # If without batch dimension, add it
        B, T, C, H, W = images.shape
        images_proc = preprocess_image(images.view(B*T, C, H, W).clone(), target_size=W)
        images_proc = rearrange(images_proc, '(b t) c h w -> b t c h w', b=B, t=T)
        _, _, _, H_proc, W_proc = images_proc.shape

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        
        # Optimization: Request only necessary layers from aggregator
        if hasattr(self.depth_head, "intermediate_layer_idx"):
            intermediate_layers = self.depth_head.intermediate_layer_idx
        else:
            intermediate_layers = None

        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.aggregator(images_proc, intermediate_layers=intermediate_layers)

        # Create a zero-allocation CPU placeholder tensor for DPTHead shape metadata
        dummy_images = torch.empty(B, T, 3, H_proc, W_proc, device="cpu")
        
        # Discard the actual video tensors from memory immediately
        del images, images_proc
        torch.cuda.empty_cache()

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                # Ensure camera_head is on the correct GPU device
                camera_head_device = next(self.camera_head.parameters()).device
                if camera_head_device != dummy_images.device: # Wait, dummy_images is on cpu, we want it on GPU
                    # We can use predictions["intrs"].device or target device which is "cuda"
                    self.camera_head.to("cuda")

                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

                # Offload camera_head to CPU after inference to save VRAM
                if not self.training:
                    self.camera_head.to('cpu')
                    torch.cuda.empty_cache()

            if self.depth_head is not None:
                # Ensure depth_head is on the correct GPU device
                self.depth_head.to("cuda")

                # Temporarily update indices to match the filtered list
                original_idx = self.depth_head.intermediate_layer_idx
                if intermediate_layers is not None:
                    self.depth_head.intermediate_layer_idx = list(range(len(intermediate_layers)))
                
                # Optimization: Process depth in chunks to save memory (chunk size 8)
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=dummy_images, patch_start_idx=patch_start_idx,
                    frames_chunk_size=kwargs.get("frames_chunk_size", 8)
                )
                
                # Restore original indices
                self.depth_head.intermediate_layer_idx = original_idx
                
                predictions["depth"] = depth
                predictions["unc_metric"] = depth_conf.view(B*T, H_proc, W_proc)

                # Offload depth_head to CPU after inference to save VRAM
                if not self.training:
                    self.depth_head.to('cpu')
                    torch.cuda.empty_cache()

        # Optimization: Do not keep a full copy of images on GPU if possible
        # predictions["images"] = (images)*255.0
        # If the user really needs it, they can use the input 'images'
        
        # output the camera pose
        predictions["poses_pred"] = torch.eye(4)[None].repeat(T, 1, 1)[None]
        predictions["poses_pred"][:,:,:3,:4], predictions["intrs"] = pose_encoding_to_extri_intri(predictions["pose_enc_list"][-1],
                                                                                                                     (H_proc, W_proc))
        predictions["poses_pred"] = torch.inverse(predictions["poses_pred"])

        if fx_prev is not None:
            scale_x = torch.from_numpy(fx_prev).to(predictions["intrs"].device) / predictions["intrs"][0, :fx_prev.shape[0], 0, 0]
            scale_x = scale_x.mean() * W_proc / W 
            predictions["intrs"][:, :, 0, 0] *= scale_x
        if fy_prev is not None:
            scale_y = torch.from_numpy(fy_prev).to(predictions["intrs"].device) / predictions["intrs"][0, :fy_prev.shape[0], 1, 1]
            scale_y = scale_y.mean() * H_proc / H
            predictions["intrs"][:, :, 1, 1] *= scale_y

        # get the points map
        if not self.training:
            # Chunked post-processing to minimize GPU VRAM while maintaining speed
            post_chunk_size = 8
            points_maps_list = []
            unc_metrics_list = []
            
            for start_t in range(0, T, post_chunk_size):
                end_t = min(start_t + post_chunk_size, T)
                
                # Move a batch of depths to GPU (cast to float32)
                d_chunk = depth[0, start_t:end_t, ..., 0].float().to(predictions["intrs"].device) # [chunk_len, H_proc, W_proc]
                intr_chunk = predictions["intrs"][0, start_t:end_t].float() # [chunk_len, 3, 3]
                
                pt_map_chunk = depth_to_points_colmap(d_chunk, intr_chunk) # [chunk_len, H_proc, W_proc, 3]
                
                pt_map_chunk = F.interpolate(
                    pt_map_chunk.permute(0, 3, 1, 2),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=True
                ).permute(0, 2, 3, 1) # [chunk_len, H, W, 3]
                
                points_maps_list.append(pt_map_chunk.cpu())
                
                unc_chunk = predictions["unc_metric"][start_t:end_t, None].float().to(predictions["intrs"].device) # [chunk_len, 1, H_proc, W_proc]
                unc_chunk = F.interpolate(
                    unc_chunk,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=True
                )[:, 0] # [chunk_len, H, W]
                
                unc_metrics_list.append(unc_chunk.cpu())
                
            predictions["points_map"] = torch.cat(points_maps_list, dim=0) # [T, H, W, 3]
            predictions["unc_metric"] = torch.cat(unc_metrics_list, dim=0) # [T, H, W]
            predictions["depth"] = depth.cpu()
        else:
            points_map = depth_to_points_colmap(depth.view(B*T, H_proc, W_proc), predictions["intrs"].view(B*T, 3, 3))
            predictions["points_map"] = points_map
            #NOTE: resize back
            predictions["points_map"] = F.interpolate(points_map.permute(0,3,1,2),
                                                             size=(H, W), mode='bilinear', align_corners=True).permute(0,2,3,1)

            predictions["unc_metric"] = F.interpolate(predictions["unc_metric"][:,None],
                                                             size=(H, W), mode='bilinear', align_corners=True)[:,0]
        predictions["intrs"][..., :1, :] *= W/W_proc 
        predictions["intrs"][..., 1:2, :] *= H/H_proc 

        if self.training:
            loss = compute_loss(predictions, annots)
            predictions["loss"] = loss
                                                                                   
        return predictions
