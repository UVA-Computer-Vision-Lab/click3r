from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from segment_anything.modeling.transformer import TwoWayTransformer
from click3r.correspondence_encoder import CorrespondenceEncoder


def _linear(module, x, weights):
    """module(x) for an nn.Linear, or F.linear with weights[module] = (weight, bias)."""
    if weights is None:
        return module(x)
    return F.linear(x, *weights[module])


def _attention(attention, q, k, v, key_bias=None, weights=None):
    """SAM's Attention.forward as one fused attention call.

    key_bias: optional (B,1,1,Nk) additive mask, -inf for keys left out of the softmax.
    weights: optional (weight, bias) per projection, as for _linear.
    """
    heads = attention.num_heads
    q = attention._separate_heads(_linear(attention.q_proj, q, weights), heads)
    k = attention._separate_heads(_linear(attention.k_proj, k, weights), heads)
    v = attention._separate_heads(_linear(attention.v_proj, v, weights), heads)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_bias)
    return _linear(attention.out_proj, attention._recombine_heads(out), weights)


class CorrespondenceAdapter(nn.Module):
    """
    Sparse prompting adapter. The prompts of a sample update its image tokens through
    a SAM two-way transformer; padded prompts stay out of attention as keys. Training
    runs only the prompted samples of a batch. Evaluation runs every sample without
    host synchronization and gives samples without valid prompts their input tokens
    back. Rounding depends on the batch shape, so results can differ between batch
    sizes in the last bits.
    Inputs:
      image_tokens: (B,N,C) tokens AFTER DUSt3R decoder_embed, BEFORE dec_blocks
      coords_xy:        (B,P,2) pixel (x,y)
      image_ids:        (B,P)   0 or 1
      pair_ids:         (B,P)   0..max_pairs-1
      image_sizes: (B,2) (H_img,W_img)
      token_positions: (B,N,2) DUSt3R token positions (y,x)
      valid_mask:       (B,P) bool; True rows are valid prompts; if all False for a sample → identity
    """

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],  # (H_tokens, W_tokens)
        input_image_size: Tuple[int, int],
        max_pairs: int = 256,
        image_position_mode: str = "external",  # "external" | "internal" | "zeros"
        transformer_depth: int = 2,
        transformer_heads: int = 8,
        transformer_mlp_dim: int = 2048,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size  # (Ht,Wt)
        self.image_position_mode = image_position_mode

        self.correspondence_encoder = CorrespondenceEncoder(
            embed_dim=embed_dim,
            image_embedding_size=image_embedding_size,
            input_image_size=input_image_size,
            max_pairs=max_pairs,
        )
        self.transformer = TwoWayTransformer(
            depth=transformer_depth,
            embedding_dim=embed_dim,
            num_heads=transformer_heads,
            mlp_dim=transformer_mlp_dim,
        )
        # The Linear layers that run (the final token-to-image attention is skipped)
        # and their weights cast to an autocast dtype, by (dtype, device).
        self._linears = [
            m for m in self.transformer.layers.modules() if isinstance(m, nn.Linear)
        ]
        self._linear_casts = {}

    def _image_position_encoding(self, token_positions: Optional[Tensor]) -> Tensor:
        """
        Returns (B,C,Ht,Wt) positional encodings for the Two-Way image stream.
        """
        Ht, Wt = self.image_embedding_size
        if self.image_position_mode == "zeros":
            assert token_positions is not None, "need B for zeros shape"
            B = token_positions.shape[0]
            return torch.zeros(
                B,
                self.embed_dim,
                Ht,
                Wt,
                device=token_positions.device,
                dtype=token_positions.dtype,
            )
        if self.image_position_mode == "internal":
            # Repeat cached dense PE for batch
            pe = self.correspondence_encoder.get_dense_image_pe().unsqueeze(
                0
            )  # (1,C,H,W)
            return pe.to(next(self.parameters()).device).expand(
                token_positions.shape[0], -1, -1, -1
            )
        # default: external from DUSt3R pos
        assert token_positions is not None, "external image_pe requires token_positions"
        return self.correspondence_encoder.image_position_encoding(
            token_positions
        )  # (B,C,H,W)

    def _cast_linear_weights(self, device_type):
        """(weight, bias) of each Linear layer in the autocast dtype.

        Autocast would cast every weight and bias at its own call; one multi-tensor
        copy per adapter call casts them all with the same rounding. Returns None,
        so the layers cast for themselves, with autograd or without autocast.
        """
        if torch.is_grad_enabled() or not torch.is_autocast_enabled(device_type):
            return None
        tensors = [t for m in self._linears for t in (m.weight, m.bias)]
        key = (torch.get_autocast_dtype(device_type), tensors[0].device)
        if key not in self._linear_casts:
            # Normal tensors, not inference tensors, so that calls under no_grad and
            # under inference_mode can both write them.
            with torch.inference_mode(False):
                self._linear_casts[key] = [
                    torch.empty_like(t, dtype=key[0]) for t in tensors
                ]
        casts = self._linear_casts[key]
        torch._foreach_copy_(casts, tensors)
        return {m: casts[2 * i : 2 * i + 2] for i, m in enumerate(self._linears)}

    def _update_image_tokens_batched(
        self,
        image_tokens: Tensor,  # (B,N,C)
        image_position_encoding: Tensor,  # (B,C,H,W)
        prompt_tokens: Tensor,  # (B,P,C)
        attend: Tensor,  # (B,P) bool; False rows are padding
    ) -> Tensor:
        """TwoWayTransformer.forward, keeping padded prompts out of attention as keys.

        Padded rows are still updated as queries, but nothing reads them. The final
        token-to-image attention only updates the prompts, so it is skipped.
        """
        keys = image_tokens
        key_pe = image_position_encoding.flatten(2).permute(0, 2, 1)
        queries = prompt_tokens
        query_pe = prompt_tokens
        # The additive mask SDPA would build from a bool mask at every call, in the
        # attention dtype, with rows padded to 16 for memory-efficient attention.
        device = attend.device.type
        dtype = (
            torch.get_autocast_dtype(device)
            if torch.is_autocast_enabled(device)
            else prompt_tokens.dtype
        )
        B, P = attend.shape
        key_bias = torch.zeros(
            B, 1, 1, -(-P // 16) * 16, dtype=dtype, device=attend.device
        )
        key_bias = key_bias[..., :P].masked_fill_(
            ~attend[:, None, None, :], float("-inf")
        )
        weights = self._cast_linear_weights(device)
        for layer in self.transformer.layers:
            if layer.skip_first_layer_pe:
                queries = _attention(
                    layer.self_attn, queries, queries, queries, key_bias, weights
                )
            else:
                q = queries + query_pe
                queries = queries + _attention(
                    layer.self_attn, q, q, queries, key_bias, weights
                )
            queries = layer.norm1(queries)
            q = queries + query_pe
            k = keys + key_pe
            queries = layer.norm2(
                queries
                + _attention(layer.cross_attn_token_to_image, q, k, keys, None, weights)
            )
            mlp = layer.mlp
            hidden = mlp.act(_linear(mlp.lin1, queries, weights))
            queries = layer.norm3(queries + _linear(mlp.lin2, hidden, weights))
            q = queries + query_pe
            keys = layer.norm4(
                keys
                + _attention(
                    layer.cross_attn_image_to_token, k, q, queries, key_bias, weights
                )
            )
        return keys

    def forward(
        self,
        image_tokens: Tensor,  # (B,N,C)
        *,
        coords_xy: Tensor,  # (B,P,2) pixels
        image_ids: Tensor,  # (B,P)
        pair_ids: Tensor,  # (B,P)
        image_sizes: Tensor,  # (B,2)
        token_positions: Optional[Tensor] = None,  # (B,N,2) for "external" image_pe
        valid_mask: Optional[Tensor] = None,  # (B,P) bool
    ) -> Tensor:
        B, N, C = image_tokens.shape
        assert C == self.embed_dim
        Ht, Wt = self.image_embedding_size
        assert N == Ht * Wt, f"N={N} must equal H*W={Ht * Wt}"

        image_position_encoding = self._image_position_encoding(
            token_positions
        )  # (B,C,H,W)

        valid = (
            torch.ones(
                B, coords_xy.shape[1], dtype=torch.bool, device=image_tokens.device
            )
            if valid_mask is None
            else valid_mask
        )
        prompted = valid.any(dim=1)

        if self.training:
            if not prompted.any():
                return image_tokens
            prompt_tokens = self.correspondence_encoder.encode_points(
                coords_xy[prompted],
                image_ids[prompted],
                pair_ids[prompted],
                image_sizes=image_sizes[prompted],
            )  # (Bp,P,C)
            updated = self._update_image_tokens_batched(
                image_tokens[prompted],
                image_position_encoding[prompted],
                prompt_tokens,
                valid[prompted],
            )
            # Under autocast the updated tokens are wider than the input ones.
            out = image_tokens.to(updated.dtype, copy=True)
            out[prompted] = updated
            return out

        # Evaluation runs every sample, so it needs no host synchronization. A sample
        # without valid prompts attends to all its rows, which keeps its softmax rows
        # non-empty, and then gets its input tokens back.
        prompt_tokens = self.correspondence_encoder.encode_points(
            coords_xy, image_ids, pair_ids, image_sizes=image_sizes
        )  # (B,P,C)
        updated = self._update_image_tokens_batched(
            image_tokens,
            image_position_encoding,
            prompt_tokens,
            valid | ~prompted[:, None],
        )
        return torch.where(prompted[:, None, None], updated, image_tokens)
