"""Pinned DynaPrompt core for its online prompt-selection profile.

Derived from zzzx1224/DynaPrompt commit
acd33cf71f5be817512f99ba3b81ec019595ad59 (`clip/custom_clip_lessctx.py`
and `dynap_classification.py`). The upstream repository did not declare a
repository-wide software license at import time.
"""

from __future__ import annotations

from typing import Any


def entropy(outputs: Any) -> Any:
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    return -(logits * logits.exp()).sum(dim=-1)


def avg_entropy(outputs: Any) -> Any:
    import math
    import torch

    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    averaged = logits.logsumexp(dim=0) - math.log(logits.shape[0])
    minimum = torch.finfo(averaged.dtype).min
    averaged = torch.clamp(averaged, min=minimum)
    return -(averaged * torch.exp(averaged)).sum(dim=-1)


def select_confident_samples(logits: Any, fraction: float) -> tuple[Any, Any]:
    import torch

    batch_entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
    if batch_entropy.dim() > 1:
        batch_entropy = batch_entropy.mean(-1)
    count = int(logits.size(0) * fraction)
    if count < 1:
        raise ValueError("DynaPrompt selection fraction leaves no adaptation view")
    indices = torch.argsort(batch_entropy, descending=False)[:count]
    return logits[indices], indices


def select_dynamic_prompts(
    logits: Any, prompt_learner: Any, num_prompts: int
) -> tuple[Any, Any, Any, Any]:
    """Run DynaPrompt's prompt ranking and online prompt-use bookkeeping."""

    import numpy as np
    import torch

    raw_prediction = logits[0].detach()
    prompt_predictions = logits.view(logits.size(0), num_prompts, -1).detach()
    prompt_entropy = entropy(prompt_predictions).mean(0)
    confidence_gap = prompt_predictions[0].max(-1)[0].unsqueeze(0) - prompt_predictions[
        1:
    ].max(dim=-1)[0]
    confidence_gap = confidence_gap.mean(0)

    entropy_order = prompt_entropy.topk(num_prompts)[1]
    initial_prompt = prompt_learner.ctx_order[0]
    position = torch.where(entropy_order == initial_prompt)[0].item()
    confidence_order = confidence_gap.topk(num_prompts)[1]
    later_entropy = entropy_order[min(position + 1, num_prompts - 1) :]
    earlier_confidence = confidence_order[:position]
    intersection = np.intersect1d(
        later_entropy.cpu().numpy(), earlier_confidence.cpu().numpy()
    )
    if intersection.size:
        selected = torch.cat(
            [
                entropy_order[
                    torch.as_tensor(intersection, device=entropy_order.device, dtype=torch.long)
                ],
                entropy_order[[position]],
            ],
            dim=0,
        )
    else:
        selected = entropy_order[[position]]

    selected_logits = logits.view(logits.size(0), num_prompts, -1)[:, selected]
    for prompt_index in selected.tolist():
        prompt_learner.ctx_order.remove(prompt_index)
        prompt_learner.ctx_use[prompt_index] += 1
        prompt_learner.ctx_order.append(prompt_index)
    raw_prediction = raw_prediction.view(num_prompts, -1)[selected].mean(0)
    return selected, raw_prediction, avg_entropy(raw_prediction.unsqueeze(0)), selected_logits


def build_prompt_model(
    clip_model: Any,
    classnames: list[str],
    *,
    n_ctx: int,
    ctx_init: str,
    num_prompts: int,
) -> Any:
    """Build the end-position, fixed-class DynaPrompt model used by this track."""

    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    from src.official.openai_clip import tokenize

    if not ctx_init:
        raise ValueError("The pinned DynaPrompt profile requires ctx_init")
    if num_prompts < 1:
        raise ValueError("DynaPrompt num_prompts must be positive")

    class TextEncoder(nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.transformer = model.transformer
            self.positional_embedding = model.positional_embedding
            self.ln_final = model.ln_final
            self.text_projection = model.text_projection
            self.dtype = model.dtype

        def forward(self, prompts: Any, tokenized_prompts: Any) -> Any:
            if prompts.dim() > 3:
                tokenized_prompts = tokenized_prompts.unsqueeze(0).repeat(
                    prompts.size(0), 1, 1
                )
                tokenized_prompts = tokenized_prompts.reshape(-1, tokenized_prompts.size(-1))
            encoded = prompts.reshape(-1, prompts.size(-2), prompts.size(-1))
            encoded = encoded + self.positional_embedding.type(self.dtype)
            encoded = encoded.permute(1, 0, 2)
            encoded = self.transformer(encoded)
            encoded = encoded.permute(1, 0, 2)
            encoded = self.ln_final(encoded).type(self.dtype)
            return encoded[
                torch.arange(encoded.shape[0], device=encoded.device),
                tokenized_prompts.argmax(dim=-1),
            ] @ self.text_projection

    class PromptLearner(nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            context = ctx_init.replace("_", " ").strip()
            context_words = len(context.split())
            if n_ctx > context_words:
                raise ValueError("DynaPrompt n_ctx cannot exceed the ctx_init word count")
            dtype = model.dtype
            device = model.visual.conv1.weight.device
            context_tokens = tokenize(context).to(device)
            with torch.no_grad():
                context_embedding = model.token_embedding(context_tokens).type(dtype)
            start = 1 + (context_words - n_ctx)
            stop = 1 + context_words
            context_vectors = torch.zeros_like(context_embedding[0, start:stop, :])
            init_context = context_embedding[0, start:stop, :].clone()
            if num_prompts > 1:
                context_vectors = context_vectors.unsqueeze(0).repeat(num_prompts, 1, 1)

            prompts = [f"{context} {name.replace('_', ' ')}." for name in classnames]
            tokenized_prompts = torch.cat([tokenize(prompt) for prompt in prompts]).to(device)
            with torch.no_grad():
                embeddings = model.token_embedding(tokenized_prompts).type(dtype)
            self.register_buffer(
                "token_prefix", embeddings[:, : 1 + context_words - n_ctx, :]
            )
            self.register_buffer("token_suffix", embeddings[:, 1 + context_words :, :])
            self.register_buffer("init_context", init_context)
            self.register_buffer("ctx_init_state", context_vectors.detach().clone())
            self.register_buffer("tokenized_prompts", tokenized_prompts)
            self.ctx = nn.Parameter(context_vectors)
            self.n_classes = len(classnames)
            self.ctx_order = list(range(num_prompts))
            self.ctx_use = [0] * num_prompts

        def reset(self) -> None:
            with torch.no_grad():
                self.ctx.copy_(self.ctx_init_state)
            self.ctx_order = list(range(num_prompts))
            self.ctx_use = [0] * num_prompts

        def forward(self) -> Any:
            context = self.ctx + self.init_context
            if context.dim() == 2:
                context = context.unsqueeze(0).expand(self.n_classes, -1, -1)
                prefix = self.token_prefix
                suffix = self.token_suffix
            else:
                context = context.unsqueeze(1).expand(-1, self.n_classes, -1, -1)
                prefix = self.token_prefix.unsqueeze(0).repeat(num_prompts, 1, 1, 1)
                suffix = self.token_suffix.unsqueeze(0).repeat(num_prompts, 1, 1, 1)
            return torch.cat([prefix, context, suffix], dim=-2)

    class ClipTestTimeTuning(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.image_encoder = clip_model.visual
            self.text_encoder = TextEncoder(clip_model)
            self.logit_scale = clip_model.logit_scale.data
            self.prompt_learner = PromptLearner(clip_model)

        @property
        def dtype(self) -> Any:
            return self.image_encoder.conv1.weight.dtype

        def reset(self) -> None:
            self.prompt_learner.reset()

        def get_text_features(self, prompt_selection: Any | None = None) -> Any:
            prompts = self.prompt_learner()
            if prompt_selection is not None:
                prompts = prompts[prompt_selection - 1]
            features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return functional.normalize(features, dim=-1)

        def forward(self, images: Any, prompt_selection: Any | None = None) -> Any:
            with torch.no_grad():
                image_features = self.image_encoder(images.type(self.dtype))
            image_features = functional.normalize(image_features, dim=-1)
            text_features = self.get_text_features(prompt_selection)
            return self.logit_scale.exp() * image_features @ text_features.t()

    return ClipTestTimeTuning()
