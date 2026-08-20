"""PoundNet source detector and semantic/authenticity feature decomposition.

The checkpoint-compatible prompt layout follows the Apache-2.0 PoundNet code
at commit ``a504acf8c1cf5273128d8ce3278929b85a32bdd1``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from src.data.transforms import build_clip_eval_transform

from .clip_vlm import OPENAI_CLIP_VIT_L14_SHA256, load_openai_clip_model


POUNDNET_COMMIT = "a504acf8c1cf5273128d8ce3278929b85a32bdd1"
POUNDNET_REPOSITORY = "https://github.com/iamwangyabin/PoundNet"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POUNDNET_CLASSES = (
    "airplane",
    "bird",
    "bottle",
    "car",
    "chair",
    "diningtable",
    "horse",
    "person",
    "sheep",
    "train",
    "bicycle",
    "boat",
    "bus",
    "cat",
    "cow",
    "dog",
    "motorbike",
    "pottedplant",
    "sofa",
    "tvmonitor",
)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _normalize_state_dict(checkpoint: Any) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(checkpoint, dict):
        raise TypeError("PoundNet checkpoint must contain a state-dict mapping")
    checkpoint_keys = sorted(str(key) for key in checkpoint)
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(state, dict) or not state:
        raise ValueError("PoundNet checkpoint does not contain model weights")
    normalized = {}
    for name, value in state.items():
        normalized_name = str(name)
        if normalized_name.startswith("module."):
            normalized_name = normalized_name.removeprefix("module.")
        if normalized_name.startswith("model."):
            normalized_name = normalized_name.removeprefix("model.")
        normalized[normalized_name] = value
    return normalized, checkpoint_keys


def pound_prompt_components(paired_text_features: Any) -> tuple[Any, Any, Any]:
    """Return semantic midpoints, unit authenticity directions, and raw differences."""

    import torch.nn.functional as functional

    if paired_text_features.ndim != 3 or int(paired_text_features.shape[1]) != 2:
        raise ValueError("paired PoundNet text features must have shape [classes, 2, dim]")
    real_features = functional.normalize(paired_text_features[:, 0], dim=-1)
    fake_features = functional.normalize(paired_text_features[:, 1], dim=-1)
    raw_midpoints = 0.5 * (real_features + fake_features)
    raw_differences = fake_features - real_features
    semantic_midpoints = functional.normalize(raw_midpoints, dim=-1)
    authenticity_directions = functional.normalize(raw_differences, dim=-1)
    return semantic_midpoints, authenticity_directions, raw_differences


def _attach_ivlp_prompts(
    clip_model: Any,
    *,
    n_ctx: int,
    vision_depth: int,
    text_depth: int,
) -> Any:
    """Add PoundNet's IVLP prompt tokens while preserving checkpoint key names."""

    import torch
    import torch.nn as nn

    class PromptedResidualBlock(nn.Module):
        def __init__(
            self,
            block: Any,
            *,
            layer_index: int,
            depth: int,
            context_tokens: int,
            text_layer: bool,
        ) -> None:
            super().__init__()
            self.attn = block.attn
            self.ln_1 = block.ln_1
            self.mlp = block.mlp
            self.ln_2 = block.ln_2
            self.attn_mask = block.attn_mask
            self.text_layer = text_layer
            self.add_prompt = 0 < layer_index < depth
            if self.add_prompt:
                self.n_ctx = context_tokens
                prompt = torch.empty(
                    context_tokens,
                    int(self.ln_1.weight.numel()),
                    device=self.attn.in_proj_weight.device,
                    dtype=self.attn.in_proj_weight.dtype,
                )
                nn.init.normal_(prompt, std=0.02)
                self.VPT_shallow = nn.Parameter(prompt)

        def _attention(self, values: Any) -> Any:
            mask = self.attn_mask
            if mask is not None:
                mask = mask.to(dtype=values.dtype, device=values.device)
            return self.attn(values, values, values, need_weights=False, attn_mask=mask)[0]

        def forward(self, values: Any) -> Any:
            if self.add_prompt:
                prompt = self.VPT_shallow.to(
                    device=values.device, dtype=values.dtype
                )[:, None].expand(-1, values.shape[1], -1)
                if self.text_layer:
                    values = torch.cat(
                        (values[:1], prompt, values[1 + self.n_ctx :]), dim=0
                    )
                else:
                    values = torch.cat((values[: -self.n_ctx], prompt), dim=0)
            values = values + self._attention(self.ln_1(values))
            return values + self.mlp(self.ln_2(values))

    class PromptedTransformer(nn.Module):
        def __init__(
            self,
            transformer: Any,
            *,
            depth: int,
            context_tokens: int,
            text_layer: bool,
        ) -> None:
            super().__init__()
            blocks = list(transformer.resblocks.children())
            self.width = int(getattr(transformer, "width", blocks[0].ln_1.weight.numel()))
            self.layers = len(blocks)
            self.resblocks = nn.Sequential(
                *[
                    PromptedResidualBlock(
                        block,
                        layer_index=index,
                        depth=depth,
                        context_tokens=context_tokens,
                        text_layer=text_layer,
                    )
                    for index, block in enumerate(blocks)
                ]
            )

        def forward(self, values: Any) -> Any:
            return self.resblocks(values)

    class PromptedVisionTransformer(nn.Module):
        def __init__(self, visual: Any) -> None:
            super().__init__()
            self.input_resolution = int(visual.input_resolution)
            self.output_dim = int(visual.output_dim)
            self.conv1 = visual.conv1
            self.class_embedding = visual.class_embedding
            self.positional_embedding = visual.positional_embedding
            self.ln_pre = visual.ln_pre
            self.transformer = PromptedTransformer(
                visual.transformer,
                depth=vision_depth,
                context_tokens=n_ctx,
                text_layer=False,
            )
            self.ln_post = visual.ln_post
            self.proj = visual.proj
            prompt = torch.empty(
                n_ctx,
                int(self.conv1.out_channels),
                device=self.conv1.weight.device,
                dtype=self.conv1.weight.dtype,
            )
            nn.init.normal_(prompt, std=0.02)
            self.VPT = nn.Parameter(prompt)

        def forward(self, images: Any) -> Any:
            values = self.conv1(images)
            values = values.reshape(values.shape[0], values.shape[1], -1).permute(0, 2, 1)
            class_token = self.class_embedding.to(values.dtype) + torch.zeros(
                values.shape[0],
                1,
                values.shape[-1],
                dtype=values.dtype,
                device=values.device,
            )
            values = torch.cat((class_token, values), dim=1)
            values = values + self.positional_embedding.to(values.dtype)
            prompt = self.VPT.to(device=values.device, dtype=values.dtype)[None].expand(
                values.shape[0], -1, -1
            )
            values = torch.cat((values, prompt), dim=1)
            values = self.ln_pre(values).permute(1, 0, 2)
            values = self.transformer(values).permute(1, 0, 2)
            values = self.ln_post(values[:, 0])
            if self.proj is not None:
                values = values @ self.proj
            return values

    clip_model.visual = PromptedVisionTransformer(clip_model.visual)
    clip_model.transformer = PromptedTransformer(
        clip_model.transformer,
        depth=text_depth,
        context_tokens=n_ctx,
        text_layer=True,
    )
    return clip_model


def _poundnet_classes() -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    from src.official.openai_clip import tokenize

    class TextEncoder(nn.Module):
        def __init__(self, clip_model: Any) -> None:
            super().__init__()
            self.transformer = clip_model.transformer
            self.positional_embedding = clip_model.positional_embedding
            self.ln_final = clip_model.ln_final
            self.text_projection = clip_model.text_projection

        @property
        def dtype(self) -> Any:
            return self.text_projection.dtype

        def forward(self, prompts: Any, tokenized_prompts: Any) -> Any:
            if prompts.ndim == 4:
                prompts = torch.flatten(prompts, start_dim=0, end_dim=1)
            values = prompts.to(
                device=self.positional_embedding.device, dtype=self.dtype
            ) + self.positional_embedding.to(dtype=self.dtype)
            values = values.permute(1, 0, 2)
            values = self.transformer(values)
            values = values.permute(1, 0, 2)
            values = self.ln_final(values).to(dtype=self.dtype)
            indices = tokenized_prompts.argmax(dim=-1)
            return values[torch.arange(values.shape[0], device=values.device), indices] @ (
                self.text_projection
            )

    class ARPromptLearner(nn.Module):
        """Checkpoint-compatible asymmetric real/fake PoundNet prompts."""

        def __init__(
            self,
            class_names: Sequence[str],
            clip_model: Any,
            *,
            n_ctx: int,
            prompt_num: int,
        ) -> None:
            super().__init__()
            if n_ctx < 1 or prompt_num < 1:
                raise ValueError("PoundNet n_ctx and prompt_num must be positive")
            self.n_cls = len(class_names)
            self.n_ctx = n_ctx
            self.prompt_num = prompt_num
            dtype = clip_model.text_projection.dtype
            device = clip_model.text_projection.device
            context_dim = int(clip_model.ln_final.weight.shape[0])
            context = torch.empty(
                prompt_num * 2,
                n_ctx,
                context_dim,
                dtype=dtype,
                device=device,
            )
            nn.init.normal_(context, std=0.02)
            self.ctx_positive = nn.Parameter(context[:prompt_num])
            self.ctx_negative = nn.Parameter(context[prompt_num:])

            prompt_prefix = " ".join(["X"] * n_ctx)
            normalized_names = [str(name).replace("_", " ") for name in class_names]
            positive_tokens = torch.cat(
                [tokenize(f"{prompt_prefix} {name}") for name in normalized_names]
            ).to(device)
            negative_tokens = torch.cat(
                [tokenize(f"{prompt_prefix} {name}") for name in normalized_names]
            ).to(device)
            with torch.no_grad():
                positive_embedding = clip_model.token_embedding(positive_tokens).to(dtype=dtype)
                negative_embedding = clip_model.token_embedding(negative_tokens).to(dtype=dtype)

            positive_embedding = positive_embedding[:, None].repeat(1, prompt_num, 1, 1)
            negative_embedding = negative_embedding[:, None].repeat(1, prompt_num, 1, 1)
            embedding = torch.cat((positive_embedding, negative_embedding), dim=1)
            positive_tokens = positive_tokens[:, None].repeat(1, prompt_num, 1)
            negative_tokens = negative_tokens[:, None].repeat(1, prompt_num, 1)
            ordered_tokens = torch.cat((positive_tokens, negative_tokens), dim=1)
            ordered_tokens = ordered_tokens.permute(1, 0, 2).contiguous().view(
                self.n_cls * prompt_num * 2, -1
            )

            self.register_buffer("token_prefix", embedding[:, :, :1])
            self.register_buffer("token_suffix", embedding[:, :, 1 + n_ctx :])
            self.register_buffer("positive_token_prefix", embedding[:, :prompt_num, :1])
            self.register_buffer(
                "positive_token_suffix", embedding[:, :prompt_num, 1 + n_ctx :]
            )
            self.register_buffer("negative_token_prefix", embedding[:, prompt_num:, :1])
            self.register_buffer(
                "negative_token_suffix", embedding[:, prompt_num:, 1 + n_ctx :]
            )
            self.register_buffer("tokenized_prompts", ordered_tokens, persistent=False)

        def _all_prompts(self) -> Any:
            context = torch.cat((self.ctx_positive, self.ctx_negative), dim=0)
            context = context[None].expand(self.n_cls, -1, -1, -1)
            return torch.cat((self.token_prefix, context, self.token_suffix), dim=2)

        def forward(self) -> Any:
            prompts = self._all_prompts()
            order = [
                item
                for index in range(self.prompt_num)
                for item in (self.prompt_num + index, index)
            ]
            return torch.cat([prompts[:, index] for index in order], dim=0)

        def forward_general_realfake(self) -> Any:
            prompts = self._all_prompts().mean(dim=0)
            fake_prompt = prompts[: self.prompt_num].mean(dim=0)
            real_prompt = prompts[self.prompt_num :].mean(dim=0)
            return torch.stack((real_prompt, fake_prompt), dim=0)

    class PoundNetDetector(nn.Module):
        def __init__(
            self,
            clip_model: Any,
            class_names: Sequence[str],
            input_transform: Any,
            *,
            n_ctx: int,
            prompt_num: int,
        ) -> None:
            super().__init__()
            self.class_names = tuple(class_names)
            self.input_transform = input_transform
            self.prompt_learner = ARPromptLearner(
                self.class_names,
                clip_model,
                n_ctx=n_ctx,
                prompt_num=prompt_num,
            )
            self.image_encoder = clip_model.visual
            self.text_encoder = TextEncoder(clip_model)
            self.logit_scale = clip_model.logit_scale
            self.feature_dim = int(clip_model.visual.output_dim)
            self.register_buffer(
                "_paired_text_features",
                torch.empty(len(self.class_names), 2, self.feature_dim),
                persistent=False,
            )
            self.register_buffer(
                "_binary_text_features", torch.empty(2, self.feature_dim), persistent=False
            )
            self.register_buffer(
                "_semantic_midpoints",
                torch.empty(len(self.class_names), self.feature_dim),
                persistent=False,
            )
            self.register_buffer(
                "_authenticity_directions",
                torch.empty(len(self.class_names), self.feature_dim),
                persistent=False,
            )
            self.register_buffer(
                "_raw_authenticity_differences",
                torch.empty(len(self.class_names), self.feature_dim),
                persistent=False,
            )

        @property
        def dtype(self) -> Any:
            return self.image_encoder.conv1.weight.dtype

        def _encode_text(self, prompts: Any, tokens: Any) -> Any:
            encoded = self.text_encoder(prompts, tokens)
            return functional.normalize(encoded, dim=-1)

        def refresh_text_features(self) -> None:
            with torch.no_grad():
                prompts = self.prompt_learner()
                encoded = self._encode_text(prompts, self.prompt_learner.tokenized_prompts)
                encoded = encoded.view(self.prompt_learner.prompt_num, 2, len(self.class_names), -1)
                paired = functional.normalize(encoded.mean(dim=0), dim=-1).permute(1, 0, 2)

                binary_prompts = self.prompt_learner.forward_general_realfake()
                token_positions = self.prompt_learner.tokenized_prompts.max(dim=0).values
                token_positions = token_positions + torch.arange(
                    token_positions.numel(), device=token_positions.device
                )
                binary_tokens = token_positions[None].repeat(2, 1)
                binary = self._encode_text(binary_prompts, binary_tokens)
                midpoints, directions, raw_differences = pound_prompt_components(paired)
                self._paired_text_features.copy_(paired)
                self._binary_text_features.copy_(binary)
                self._semantic_midpoints.copy_(midpoints)
                self._authenticity_directions.copy_(directions)
                self._raw_authenticity_differences.copy_(raw_differences)

        def forward_features(self, images: Any) -> Any:
            encoded = self.image_encoder(images.to(dtype=self.dtype))
            return functional.normalize(encoded, dim=-1)

        def classifier(self, features: Any) -> Any:
            prototypes = self._binary_text_features.to(dtype=features.dtype)
            return self.logit_scale.exp().to(dtype=features.dtype) * features @ prototypes.t()

        def decompose_features(
            self, features: Any, *, semantic_temperature: float
        ) -> tuple[Any, Any, Any, Any]:
            if semantic_temperature <= 0.0:
                raise ValueError("semantic_temperature must be positive")
            midpoints = self._semantic_midpoints.to(dtype=features.dtype)
            directions = self._authenticity_directions.to(dtype=features.dtype)
            raw_differences = self._raw_authenticity_differences.to(dtype=features.dtype)
            routing = (features @ midpoints.t() / semantic_temperature).softmax(dim=1)
            semantic_keys = functional.normalize(routing @ midpoints, dim=-1)
            coefficients = features @ directions.t()
            residuals = torch.einsum("bc,bc,cd->bd", routing, coefficients, directions)
            conditional_margin = self.logit_scale.exp().to(dtype=features.dtype) * (
                routing * (features @ raw_differences.t())
            ).sum(dim=1)
            return semantic_keys, residuals, conditional_margin, routing

        def forward_pound_features(
            self, images: Any, *, semantic_temperature: float
        ) -> dict[str, Any]:
            features = self.forward_features(images)
            semantic_keys, residuals, conditional_margin, routing = self.decompose_features(
                features, semantic_temperature=semantic_temperature
            )
            return {
                "source_logits": self.classifier(features),
                "features": features,
                "semantic_keys": semantic_keys,
                "residuals": residuals,
                "conditional_margin": conditional_margin,
                "semantic_routing": routing,
            }

        def forward(self, images: Any) -> Any:
            return self.classifier(self.forward_features(images))

    return TextEncoder, ARPromptLearner, PoundNetDetector


def build_poundnet_detector(
    config: dict[str, Any], *, device: str | Any = "cpu"
) -> tuple[Any, dict[str, Any]]:
    """Build the published PoundNet checkpoint on the fixed OpenAI ViT-L/14."""

    import torch

    for key in ("clip_path", "checkpoint"):
        if not config.get(key):
            raise ValueError(f"PoundNet method config requires {key}")
    architecture = str(config.get("architecture", "ViT-L/14"))
    if architecture.lower().replace("-", "").replace("/", "") != "vitl14":
        raise ValueError("PoundNet is fixed to OpenAI CLIP ViT-L/14")
    clip_path = _resolve_path(config["clip_path"])
    checkpoint_path = _resolve_path(config["checkpoint"])
    if not clip_path.is_file():
        raise FileNotFoundError(f"CLIP checkpoint does not exist: {clip_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PoundNet checkpoint does not exist: {checkpoint_path}")

    class_names = tuple(config.get("semantic_class_names", DEFAULT_POUNDNET_CLASSES))
    if not class_names or any(not str(name).strip() for name in class_names):
        raise ValueError("PoundNet semantic_class_names must contain non-empty names")
    n_ctx = int(config.get("n_ctx", 16))
    prompt_num = int(config.get("prompt_num", 1))
    vision_depth = int(config.get("vision_depth", 8))
    text_depth = int(config.get("text_depth", 8))
    if min(n_ctx, prompt_num, vision_depth, text_depth) < 1:
        raise ValueError("PoundNet prompt dimensions and depths must be positive")

    clip_model, clip_metadata = load_openai_clip_model(
        {
            "architecture": "ViT-L/14",
            "checkpoint": str(clip_path),
            "image_size": int(config.get("image_size", 224)),
        },
        device=device,
    )
    clip_model = _attach_ivlp_prompts(
        clip_model,
        n_ctx=n_ctx,
        vision_depth=vision_depth,
        text_depth=text_depth,
    )
    image_size = int(config.get("image_size", 224))
    expected_size = int(getattr(clip_model.visual, "input_resolution", image_size))
    if image_size != expected_size:
        raise ValueError(f"ViT-L/14 expects image_size={expected_size}, got {image_size}")
    _text_encoder, _prompt_learner, detector_class = _poundnet_classes()
    detector = detector_class(
        clip_model,
        class_names,
        build_clip_eval_transform(
            image_size,
            resize_size=int(config.get("resize_size", round(image_size / 0.875))),
        ),
        n_ctx=n_ctx,
        prompt_num=prompt_num,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict, checkpoint_keys = _normalize_state_dict(checkpoint)
    detector.load_state_dict(state_dict, strict=True)
    detector.to(device)
    if str(device).startswith("cpu"):
        detector.float()
    detector.eval()
    detector.requires_grad_(False)
    detector.refresh_text_features()
    metadata = {
        "official_commit": POUNDNET_COMMIT,
        "official_repository": POUNDNET_REPOSITORY,
        "upstream_license": "Apache-2.0",
        "architecture": "ViT-L/14",
        "checkpoint_keys": checkpoint_keys,
        "checkpoint_path": str(checkpoint_path),
        "clip_checkpoint_path": str(clip_path),
        "clip_checkpoint_sha256_expected": OPENAI_CLIP_VIT_L14_SHA256,
        "source_setup": "published_poundnet_prompt_checkpoint_with_openai_clip_vitl14",
        "semantic_class_names": list(class_names),
        "n_ctx": n_ctx,
        "prompt_num": prompt_num,
        "vision_depth": vision_depth,
        "text_depth": text_depth,
        "image_size": image_size,
        "clip_initialization": clip_metadata,
    }
    del checkpoint, state_dict, clip_model
    return detector, metadata


__all__ = [
    "DEFAULT_POUNDNET_CLASSES",
    "POUNDNET_COMMIT",
    "POUNDNET_REPOSITORY",
    "build_poundnet_detector",
    "pound_prompt_components",
]
