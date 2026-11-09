from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor
from segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom


class CorrespondenceEncoder(nn.Module):
    """Add coordinate, image-identity, and correspondence-identity embeddings.

    The two views use different image IDs and share the pair ID of each match.
    Coordinates are pixel positions in the corresponding input image.
    """

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],  # (H_tokens, W_tokens)
        input_image_size: Tuple[int, int],  # default fallback
        num_images: int = 2,
        max_pairs: int = 256,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        # Token coordinates are pixel / patch size, the convention Click3R was
        # trained with. The trainer can change this offset (--prompt_token_offset).
        self.token_offset = 0.0
        self.register_buffer(
            "default_input_image_size",
            torch.tensor(input_image_size, dtype=torch.int64),
            persistent=False,
        )
        self.coordinate_encoding = PositionEmbeddingRandom(embed_dim // 2)
        self.image_embedding = nn.Embedding(num_images, embed_dim)
        self.pair_embedding = nn.Embedding(max_pairs, embed_dim)

        # cached dense grid PE
        Ht, Wt = image_embedding_size
        with torch.no_grad():
            dense = self.coordinate_encoding.forward((Ht, Wt))  # (C,H,W)
        self.register_buffer("dense_image_pe", dense, persistent=False)

    @torch.no_grad()
    def set_input_image_size(self, size_hw: Union[Tuple[int, int], Tensor]) -> None:
        if isinstance(size_hw, tuple):
            self.default_input_image_size[:] = torch.tensor(
                size_hw, dtype=torch.int64, device=self.default_input_image_size.device
            )

    def get_dense_image_pe(self) -> Tensor:
        return self.dense_image_pe  # (C,H,W)

    def _normalize_pixel_coordinates(
        self, coords_xy_pix: Tensor, img_sizes_b2: Tensor
    ) -> Tensor:
        # Map pixel coordinates to [0,1] on the scale of the dense token PE, which
        # places token i at i / (N_tokens - 1). The token coordinate is
        # pixel / patch size minus token_offset.
        B, P, _ = coords_xy_pix.shape
        Ht, Wt = self.image_embedding_size
        H_img = img_sizes_b2[:, 0].float().clamp(min=1).view(B, 1)
        W_img = img_sizes_b2[:, 1].float().clamp(min=1).view(B, 1)
        patch_h = (H_img / float(Ht)).view(B, 1)
        patch_w = (W_img / float(Wt)).view(B, 1)
        x_pix = coords_xy_pix[..., 0].float()
        y_pix = coords_xy_pix[..., 1].float()
        x_tok = x_pix / patch_w - self.token_offset
        y_tok = y_pix / patch_h - self.token_offset
        x01 = (x_tok / max(Wt - 1, 1)).clamp(0, 1)
        y01 = (y_tok / max(Ht - 1, 1)).clamp(0, 1)
        return torch.stack([x01, y01], dim=-1)  # (B,P,2) (x,y)

    def _encode_coordinates(self, coords01_xy: Tensor) -> Tensor:
        if hasattr(self.coordinate_encoding, "_pe_encoding"):
            return self.coordinate_encoding._pe_encoding(coords01_xy)  # (B,P,C)
        # fallback via grid_sample
        B, P, _ = coords01_xy.shape
        C, Ht, Wt = self.dense_image_pe.shape
        dense = (
            self.dense_image_pe.unsqueeze(0)
            .expand(B, -1, -1, -1)
            .to(coords01_xy.device)
        )
        grid = (coords01_xy * 2.0 - 1.0).view(B, 1, P, 2)
        pe = torch.nn.functional.grid_sample(
            dense, grid, mode="bilinear", align_corners=True
        )
        return pe.squeeze(2).transpose(1, 2).contiguous()  # (B,P,C)

    def encode_points(
        self,
        coords_xy_pixels: Tensor,  # (B,Pvalid,2)
        image_ids: Tensor,  # (B,Pvalid)
        pair_ids: Tensor,  # (B,Pvalid) in [0..max_pairs-1], padded rows too
        image_sizes: Optional[Tensor] = None,  # (B,2) (H,W)
    ) -> Tensor:
        B, P, two = coords_xy_pixels.shape
        assert two == 2, "coords must be (x,y)"

        if image_sizes is None:
            img_sizes_b2 = (
                self.default_input_image_size.view(1, 2)
                .expand(B, -1)
                .to(coords_xy_pixels.device)
            )
        else:
            img_sizes_b2 = image_sizes.to(coords_xy_pixels.device)

        coords01_xy = self._normalize_pixel_coordinates(coords_xy_pixels, img_sizes_b2)
        pe = self._encode_coordinates(coords01_xy)  # (B,P,C)
        img_emb = self.image_embedding(image_ids)  # (B,P,C)
        pair_emb = self.pair_embedding(pair_ids)  # (B,P,C)

        return pe + img_emb + pair_emb  # (B,P,C)

    def image_position_encoding(self, token_positions: Tensor) -> Tensor:
        B, N, _ = token_positions.shape
        Ht, Wt = self.image_embedding_size
        assert N == Ht * Wt
        y = token_positions[..., 0].float() / max(Ht - 1, 1)
        x = token_positions[..., 1].float() / max(Wt - 1, 1)
        xy01 = torch.stack([x, y], dim=-1)
        pe_bNC = self._encode_coordinates(xy01)
        return pe_bNC.transpose(1, 2).reshape(B, self.embed_dim, Ht, Wt)  # (B,C,H,W)
