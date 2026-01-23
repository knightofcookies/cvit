from typing import Dict

import torch
import torch.nn as nn

from degod.deepseek_ocr import MlpProjector, build_clip_l, build_sam_fast_vit_b


class DeepEncoder(nn.Module):
    """
    Adapted from Deepseek-OCR's DeepseekOCRModel class,
    eliminating the Deepseek Language Model dependency.
    """

    def __init__(self):
        super().__init__()

        self.sam_model = build_sam_fast_vit_b()
        self.clip_model = build_clip_l()

        n_embed = 1028

        # TODO Do we use this and/or our custom projector?
        self.projector = MlpProjector(
            {"projector_type": "linear", "input_dum": 2048, "n_embed": n_embed}
        )

        # scale by 1/root(n_embed) to control variance for stability
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
        self.view_separator = nn.Paramete(torch.randn(n_embed) * embed_std)

    def forward(self):
        pass
