# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# Modified by the Click3R authors: adapted from dust3r/model.py with
# correspondence adapters and prompt handling.
import torch
import os
from packaging import version
from copy import deepcopy
import huggingface_hub

from dust3r.utils.misc import (
    fill_default_args,
    freeze_all_params,
    is_symmetrized,
    interleave,
    transpose_to_landscape,
)
from dust3r.heads import head_factory
from dust3r.patch_embed import get_patch_embed

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa

from click3r.correspondence_adapter import CorrespondenceAdapter
from click3r.correspondence_encoder import CorrespondenceEncoder

inf = float("inf")

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), (
    "Outdated huggingface_hub version, please reinstall requirements.txt"
)


def normalize_checkpoint_keys(state_dict):
    """Map checkpoint keys to the current module names.

    Strips the DDP ("module") and torch.compile ("_orig_mod") prefixes and renames
    the module names that the released checkpoint uses. Only key names change;
    tensors and their order are preserved.
    """
    historical_names = {
        "sam_adapter1": "correspondence_adapter1",
        "sam_adapter2": "correspondence_adapter2",
        "prompt_enc": "correspondence_encoder",
        "two_way": "transformer",
        "pe_coords": "coordinate_encoding",
        "img_id_embed": "image_embedding",
        "pair_id_embed": "pair_embedding",
    }
    normalized = {}
    for name, tensor in state_dict.items():
        parts = [
            part for part in name.split(".") if part not in ("module", "_orig_mod")
        ]
        name = ".".join(historical_names.get(part, part) for part in parts)
        if name in normalized:
            raise ValueError(f"Duplicate checkpoint parameter after renaming: {name}")
        normalized[name] = tensor
    return normalized


def load_model(model_path, device, verbose=True):
    if verbose:
        print("... loading model from", model_path)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model_spec = ckpt["args"].model if "args" in ckpt else ckpt["metadata"]["model"]
    args = model_spec.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if "landscape_only" not in args:
        args = args[:-1] + ", landscape_only=False)"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt["model"], strict=True)
    if verbose:
        print(s)
    return net.to(device)


class Click3R(
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="click3r",
    tags=["image-to-3d"],
):
    """DUSt3R stereo reconstruction with correspondence-conditioned decoder inputs.

    A shared image encoder feeds two decoder streams. Each stream has its own
    correspondence adapter, and both adapters share a correspondence encoder.
    Both output pointmaps use the first camera's coordinate frame.
    """

    def __init__(
        self,
        output_mode="pts3d",
        head_type="linear",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",  # PatchEmbedDust3R or ManyAR_PatchEmbed
        **croco_kwargs,
    ):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(
            output_mode,
            head_type,
            landscape_only,
            depth_mode,
            conf_mode,
            **croco_kwargs,
        )
        self.set_freeze(freeze)

        # Correspondence adapters share one coordinate and identity encoder.
        H, W = self.patch_embed.grid_size
        shared_encoder = CorrespondenceEncoder(
            embed_dim=self.dec_embed_dim,
            image_embedding_size=(H, W),
            input_image_size=self.patch_embed.img_size,
            num_images=2,
            max_pairs=256,
        )

        self.prompts_enabled = True
        self.correspondence_adapter1 = CorrespondenceAdapter(
            embed_dim=self.dec_embed_dim,
            image_embedding_size=(H, W),
            input_image_size=self.patch_embed.img_size,
            max_pairs=256,
            image_position_mode="external",
            transformer_depth=2,
            transformer_heads=8,
            transformer_mlp_dim=2048,
        )
        # Replace each adapter's own encoder with the shared one.
        self.correspondence_adapter1.correspondence_encoder = shared_encoder
        self.correspondence_adapter2 = CorrespondenceAdapter(
            embed_dim=self.dec_embed_dim,
            image_embedding_size=(H, W),
            input_image_size=self.patch_embed.img_size,
            max_pairs=256,
            image_position_mode="external",
            transformer_depth=2,
            transformer_heads=8,
            transformer_mlp_dim=2048,
        )
        self.correspondence_adapter2.correspondence_encoder = shared_encoder

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            try:
                model = super(Click3R, cls).from_pretrained(
                    pretrained_model_name_or_path, **kw
                )
            except TypeError as exc:
                raise RuntimeError(
                    f"tried to load {pretrained_model_name_or_path} from huggingface, but failed"
                ) from exc
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_size = patch_size
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim
        )

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = normalize_checkpoint_keys(ckpt)
        if not any(k.startswith("dec_blocks2") for k in new_ckpt):
            for key, value in list(new_ckpt.items()):
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks2")] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "mask": [self.mask_token],
            "encoder": [self.mask_token, self.patch_embed, self.enc_blocks],
            "dust3r": [
                # Freeze all DUSt3R original parameters, keep Click3R adapters trainable
                self.mask_token,
                self.patch_embed,
                self.enc_blocks,
                self.enc_norm,
                self.decoder_embed,
                self.dec_blocks,
                self.dec_blocks2,
                self.dec_norm,
                self.downstream_head1,
                self.downstream_head2,
            ],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        patch_size,
        img_size,
        **kw,
    ):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, (
            f"{img_size=} must be multiple of {patch_size=}"
        )
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(
            head_type, output_mode, self, has_conf=bool(conf_mode)
        )
        self.downstream_head2 = head_factory(
            head_type, output_mode, self, has_conf=bool(conf_mode)
        )
        # Handle landscape and portrait inputs in the prediction heads.
        self.head1 = transpose_to_landscape(
            self.downstream_head1, activate=landscape_only
        )
        self.head2 = transpose_to_landscape(
            self.downstream_head2, activate=landscape_only
        )

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(
                torch.cat((img1, img2), dim=0),
                torch.cat((true_shape1, true_shape2), dim=0),
            )
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1["img"]
        img2 = view2["img"]
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get(
            "true_shape", torch.tensor(img1.shape[-2:])[None].repeat(B, 1)
        )
        shape2 = view2.get(
            "true_shape", torch.tensor(img2.shape[-2:])[None].repeat(B, 1)
        )
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(
                img1[::2], img2[::2], shape1[::2], shape2[::2]
            )
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(
                img1, img2, shape1, shape2
            )

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2, prompts1=None, prompts2=None):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        # Condition each decoder stream on its view of the correspondences.
        if getattr(self, "prompts_enabled", False):
            if hasattr(self, "correspondence_adapter1") and prompts1 is not None:
                f1 = self.correspondence_adapter1(
                    f1,
                    coords_xy=prompts1.get("coords"),
                    image_ids=prompts1.get("image_ids"),
                    pair_ids=prompts1.get("pair_ids"),
                    image_sizes=prompts1.get("shape"),
                    token_positions=pos1,
                    valid_mask=prompts1.get("valid_mask", None),
                )

            if hasattr(self, "correspondence_adapter2") and prompts2 is not None:
                f2 = self.correspondence_adapter2(
                    f2,
                    coords_xy=prompts2.get("coords"),
                    image_ids=prompts2.get("image_ids"),
                    pair_ids=prompts2.get("pair_ids"),
                    image_sizes=prompts2.get("shape"),
                    token_positions=pos2,
                    valid_mask=prompts2.get("valid_mask", None),
                )

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2, strict=False):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output, strict=False)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        head = getattr(self, f"head{head_num}")
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(
            view1, view2
        )

        # pass per-view prompts (may be None)
        prompts1 = view1.get("sparse_prompts", None)
        prompts2 = view2.get("sparse_prompts", None)

        dec1, dec2 = self._decoder(
            feat1, pos1, feat2, pos2, prompts1=prompts1, prompts2=prompts2
        )

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2["pts3d_in_other_view"] = res2.pop(
            "pts3d"
        )  # predict view2's pts3d in view1's frame
        return res1, res2


# Constructor alias for metadata in existing training checkpoints.
AsymmetricCroCo3DStereo = Click3R


def model_from_spec(spec):
    """Build a model from a trusted checkpoint constructor description."""
    return eval(spec)
