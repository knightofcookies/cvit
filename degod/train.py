"""
    Implementation discontinued as I discovered Deepseek-OCR-2 came out this morning while I was having lunch.
"""

import torch
import torch.nn as nn
from deepseek_ocr import build_clip_l, build_sam_fast_vit_b, MlpProjector

class DeepEncoder(nn.Module):
    """
    Adapted from Deepseek-OCR's DeepseekOCRModel class.
    This module performs the Image Encoding -> Feature Fusion -> Projection -> 
    Spatial Arrangement (with special tokens).
    
    Co-authored by Gemini 3 Pro.
    """

    def __init__(self):
        super().__init__()

        # Ensure these build functions are available in your scope
        self.sam_model = build_sam_fast_vit_b()
        self.vision_model = build_clip_l() # Renamed to match original 'vision_model' usage

        # Original uses 1280, your snippet had 1028. Kept dynamic based on your n_embed variable.
        self.n_embed = 1028 

        self.projector = MlpProjector(
            # Fixed typo: input_dum -> input_dim
            {"projector_type": "linear", "input_dim": 2048, "n_embed": self.n_embed}
        )

        # Scale by 1/root(n_embed) to control variance for stability
        embed_std = 1 / torch.sqrt(torch.tensor(self.n_embed, dtype=torch.float32))
        
        self.image_newline = nn.Parameter(torch.randn(self.n_embed) * embed_std)
        
        # Note: Original weights use key 'view_seperator' (spelling error in original repo).
        # If loading pre-trained weights, you might need to rename this or map keys.
        # Fixed typo: nn.Paramete -> nn.Parameter
        self.view_separator = nn.Parameter(torch.randn(self.n_embed) * embed_std)

    def forward(
        self, 
        images: list, 
        images_spatial_crop: list
    ):
        """
        Args:
            images: A list where each item is a tuple/list: (patches_tensor, original_image_tensor).
                    - patches_tensor shape: [P, C, H, W]
                    - original_image_tensor shape: [1, C, H, W]
            images_spatial_crop: A list of [width_crop_num, height_crop_num] for each image.
            
        Returns:
            images_in_batch: A list of 1D tensors. Each tensor is the sequence of 
                             visual embeddings for one image in the batch.
        """
        
        images_in_batch = []
        
        # Iterate over the batch
        for image, crop_shape in zip(images, images_spatial_crop):
            
            # Unpack the tuple created by the data processor
            # image[0] = patches (High Res Crops), image[1] = global view (Low Res)
            patches = image[0]
            image_ori = image[1]

            # Check if we have patches (Crop Mode)
            if torch.sum(patches).item() != 0:
                # --- PROCESS LOCAL PATCHES ---
                # 1. Extract Features
                local_features_1 = self.sam_model(patches)
                local_features_2 = self.vision_model(patches, local_features_1)

                # 2. Fuse Features (taking specific slices as per original code)
                local_features = torch.cat(
                    (
                        local_features_2[:, 1:], 
                        local_features_1.flatten(2).permute(0, 2, 1),
                    ),
                    dim=-1,
                )
                
                # 3. Project
                local_features = self.projector(local_features)

                # --- PROCESS GLOBAL VIEW ---
                global_features_1 = self.sam_model(image_ori)
                global_features_2 = self.vision_model(image_ori, global_features_1)
                
                global_features = torch.cat(
                    (
                        global_features_2[:, 1:],
                        global_features_1.flatten(2).permute(0, 2, 1),
                    ),
                    dim=-1,
                )
                global_features = self.projector(global_features)

                # --- RESHAPE & ADD SPECIAL TOKENS ---
                
                # Calculate dimensions
                _, hw, n_dim = global_features.shape
                h = w = int(hw**0.5)

                _2, hw2, n_dim2 = local_features.shape
                h2 = w2 = int(hw2**0.5)

                width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]

                # Format Global Features
                global_features = global_features.view(h, w, n_dim)
                global_features = torch.cat(
                    [
                        global_features,
                        self.image_newline[None, None, :].expand(h, 1, n_dim),
                    ],
                    dim=1,
                )
                global_features = global_features.view(-1, n_dim)

                # Format Local Features (Re-arrange patches spatially)
                local_features = (
                    local_features.view(
                        height_crop_num, width_crop_num, h2, w2, n_dim2
                    )
                    .permute(0, 2, 1, 3, 4)
                    .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
                )
                
                # Add Newline to Local Features
                local_features = torch.cat(
                    [
                        local_features,
                        self.image_newline[None, None, :].expand(
                            height_crop_num * h2, 1, n_dim2
                        ),
                    ],
                    dim=1,
                )
                local_features = local_features.view(-1, n_dim2)

                # Concatenate Local + Global + View Separator
                global_local_features = torch.cat(
                    [
                        local_features,
                        global_features,
                        self.view_separator[None, :],
                    ],
                    dim=0,
                )

            else:
                # --- NO PATCHES (Low Res / No Crop Mode) ---
                global_features_1 = self.sam_model(image_ori)
                global_features_2 = self.vision_model(image_ori, global_features_1)
                
                global_features = torch.cat(
                    (
                        global_features_2[:, 1:],
                        global_features_1.flatten(2).permute(0, 2, 1),
                    ),
                    dim=-1,
                )
                global_features = self.projector(global_features)

                _, hw, n_dim = global_features.shape
                h = w = int(hw**0.5)

                global_features = global_features.view(h, w, n_dim)

                # Add Newline
                global_features = torch.cat(
                    [
                        global_features,
                        self.image_newline[None, None, :].expand(h, 1, n_dim),
                    ],
                    dim=1,
                )

                global_features = global_features.view(-1, n_dim)

                # Add View Separator
                global_local_features = torch.cat(
                    [global_features, self.view_separator[None, :]], dim=0
                )

            images_in_batch.append(global_local_features)

        # Returns a list of Tensors. 
        # Note: These tensors have different lengths depending on the crop strategy,
        # so they cannot simply be stacked without padding.
        return images_in_batch

class Projector(nn.Module):
    """
        MLP to bridge the Deepseek-OCR decoder's
        expected inputs to GOT-OCR2's decoder.
    """
    def __init__(self, input_dim, output_dim):
        self.mnlp = nn.Sequential([
            nn.Linear(),
            nn.GeLU(),
            nn.Linear(),
            nn.LayerNorm()
        ])
        
    def forward(self, x):
        return self.mnlp(x)
        
class GOTOCR2Decoder(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass

class FusionModel(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass
