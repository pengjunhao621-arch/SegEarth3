"""Category-agnostic image-conditioned prompt synthesis for Prompt-SAM3 v1.

The module intentionally contains no source-dataset class table.  Static class
semantics are supplied by SAM3's frozen text encoder; the trainable parts only
learn how to retrieve local image states and inject shared/local/global prompt
residuals.  This is what permits a checkpoint trained on OpenEarthMap to be
reused with a different target vocabulary.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


COMPONENT_MODES = (
    "full",
    "no_shared",
    "no_state",
    "no_global",
    "static_baseline",
)


def _inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1e-5), 1.0 - 1e-5)
    return math.log(value / (1.0 - value))


class PromptSynthesisModule(nn.Module):
    """Compose bounded shared, local-state, and image-global residuals.

    The configurable 4/4/1 defaults are *internal capacity budgets*: four
    shared tokens, four class-conditioned local prototypes, and one pooled
    global token.  Each component is mixed into SAM3's existing 32-position
    text memory and added to the frozen static class-name embedding.
    """

    def __init__(
        self,
        spatial_dim: int = 256,
        language_dim: int = 256,
        text_width: int = 1024,
        global_dim: int = 256,
        context_length: int = 32,
        num_shared_tokens: int = 4,
        num_state_prototypes: int = 4,
        num_global_tokens: int = 1,
        residual_scale: float = 0.05,
        initial_gate: float = 0.10,
        attention_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if min(num_shared_tokens, num_state_prototypes, num_global_tokens) < 1:
            raise ValueError("All prompt component token counts must be positive")
        if context_length < 2:
            raise ValueError("context_length must be at least two")

        self.spatial_dim = int(spatial_dim)
        self.language_dim = int(language_dim)
        self.text_width = int(text_width)
        self.global_dim = int(global_dim)
        self.context_length = int(context_length)
        self.num_shared_tokens = int(num_shared_tokens)
        self.num_state_prototypes = int(num_state_prototypes)
        self.num_global_tokens = int(num_global_tokens)
        self.residual_scale = float(residual_scale)
        self.attention_temperature = float(attention_temperature)

        self.shared_context = nn.Parameter(
            torch.empty(self.num_shared_tokens, self.text_width)
        )
        self.state_slots = nn.Parameter(
            torch.empty(self.num_state_prototypes, self.spatial_dim)
        )
        self.context_position_logits = nn.Parameter(
            torch.zeros(self.num_shared_tokens, self.context_length)
        )
        self.state_position_logits = nn.Parameter(
            torch.zeros(self.num_state_prototypes, self.context_length)
        )
        self.global_position_logits = nn.Parameter(
            torch.zeros(self.num_global_tokens, self.context_length)
        )

        self.class_to_state = nn.Sequential(
            nn.LayerNorm(self.language_dim),
            nn.Linear(
                self.language_dim,
                self.num_state_prototypes * self.spatial_dim,
            ),
        )
        self.spatial_key = nn.Conv2d(
            self.spatial_dim, self.spatial_dim, kernel_size=1, bias=False
        )
        self.spatial_value = nn.Conv2d(
            self.spatial_dim, self.spatial_dim, kernel_size=1, bias=False
        )
        self.state_to_text = nn.Linear(self.spatial_dim, self.text_width)
        self.global_to_text = nn.Sequential(
            nn.LayerNorm(self.global_dim),
            nn.Linear(
                self.global_dim,
                self.num_global_tokens * self.text_width,
            ),
        )

        self.context_norm = nn.LayerNorm(self.text_width)
        self.state_norm = nn.LayerNorm(self.text_width)
        self.global_norm = nn.LayerNorm(self.text_width)
        gate_init = _inverse_sigmoid(initial_gate)
        self.gate_logits = nn.Parameter(
            torch.full((3,), gate_init, dtype=torch.float32)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.shared_context, std=0.02)
        nn.init.normal_(self.state_slots, std=0.02)
        nn.init.normal_(self.context_position_logits, std=0.01)
        nn.init.normal_(self.state_position_logits, std=0.01)
        nn.init.normal_(self.global_position_logits, std=0.01)
        for module in (
            self.class_to_state,
            self.spatial_key,
            self.spatial_value,
            self.state_to_text,
            self.global_to_text,
        ):
            for submodule in module.modules():
                if isinstance(submodule, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(submodule.weight)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)

    @staticmethod
    def _component_switches(mode: str) -> Tuple[float, float, float]:
        mode = str(mode).lower()
        if mode not in COMPONENT_MODES:
            raise ValueError(
                f"Unknown prompt component mode {mode!r}; expected one of "
                f"{COMPONENT_MODES}"
            )
        return (
            0.0 if mode in ("no_shared", "static_baseline") else 1.0,
            0.0 if mode in ("no_state", "static_baseline") else 1.0,
            0.0 if mode in ("no_global", "static_baseline") else 1.0,
        )

    @staticmethod
    def _masked_class_anchor(
        static_language: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        active = (~padding_mask).to(static_language.dtype).unsqueeze(-1)
        return (static_language * active).sum(dim=1) / active.sum(
            dim=1
        ).clamp_min(1.0)

    @staticmethod
    def _masked_position_weights(
        logits: torch.Tensor, active_positions: torch.Tensor
    ) -> torch.Tensor:
        # logits: [M, L], active_positions: [Q, L] -> [Q, M, L]
        expanded = logits.unsqueeze(0).expand(active_positions.shape[0], -1, -1)
        expanded = expanded.masked_fill(
            ~active_positions.unsqueeze(1), torch.finfo(expanded.dtype).min
        )
        return expanded.softmax(dim=-1)

    @staticmethod
    def _prototype_diagnostics(prototypes: torch.Tensor) -> Dict[str, torch.Tensor]:
        # prototypes: [B, Q, K, C]
        normalized = F.normalize(prototypes.float(), dim=-1, eps=1e-6)
        similarity = torch.einsum("bqkc,bqjc->bqkj", normalized, normalized)
        k = prototypes.shape[-2]
        if k > 1:
            off_diagonal = ~torch.eye(
                k, device=prototypes.device, dtype=torch.bool
            )
            pair_cosine = similarity[..., off_diagonal].mean()
        else:
            pair_cosine = similarity.new_zeros(())

        singular_values = torch.linalg.svdvals(prototypes.float())
        spectrum = singular_values / singular_values.sum(dim=-1, keepdim=True).clamp_min(
            1e-6
        )
        effective_rank = torch.exp(
            -(spectrum * spectrum.clamp_min(1e-8).log()).sum(dim=-1)
        ).mean()
        return {
            "prototype_pair_cosine": pair_cosine,
            "prototype_effective_rank": effective_rank,
        }

    def forward(
        self,
        spatial_feature: torch.Tensor,
        static_inputs_embeds: torch.Tensor,
        static_language: torch.Tensor,
        padding_mask: torch.Tensor,
        global_feature: torch.Tensor,
        component_mode: str = "full",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Synthesize continuous SAM3 token embeddings.

        Shapes:
            spatial_feature: ``[B, C, H, W]``
            static_inputs_embeds: ``[Q, L, text_width]``
            static_language: ``[Q, L, language_dim]``
            padding_mask: ``[Q, L]``, True at padding positions
            global_feature: ``[B, global_dim]``
        """
        if spatial_feature.ndim != 4:
            raise ValueError("spatial_feature must have shape [B, C, H, W]")
        batch_size = spatial_feature.shape[0]
        num_queries, sequence_length, _ = static_inputs_embeds.shape
        if sequence_length != self.context_length:
            raise ValueError(
                f"Expected context length {self.context_length}, got "
                f"{sequence_length}"
            )
        if global_feature.shape != (batch_size, self.global_dim):
            raise ValueError(
                f"Expected global feature {(batch_size, self.global_dim)}, "
                f"got {tuple(global_feature.shape)}"
            )

        class_anchor = self._masked_class_anchor(static_language, padding_mask)
        class_queries = self.class_to_state(class_anchor).view(
            num_queries, self.num_state_prototypes, self.spatial_dim
        )
        class_queries = class_queries + self.state_slots.unsqueeze(0)
        class_queries = F.normalize(class_queries, dim=-1, eps=1e-6)

        keys = self.spatial_key(spatial_feature).flatten(2).transpose(1, 2)
        values = self.spatial_value(spatial_feature).flatten(2).transpose(1, 2)
        keys = F.normalize(keys, dim=-1, eps=1e-6)
        attention_logits = torch.einsum(
            "qkc,bnc->bqkn", class_queries, keys
        ) / max(self.attention_temperature, 1e-6)
        attention = attention_logits.softmax(dim=-1)
        prototypes = torch.einsum("bqkn,bnc->bqkc", attention, values)

        active_bool = ~padding_mask
        context_tokens = self.context_norm(self.shared_context)
        context_weights = self._masked_position_weights(
            self.context_position_logits, active_bool
        )
        context_delta = torch.einsum(
            "md,qml->qld", context_tokens, context_weights
        ).unsqueeze(0)

        state_tokens = self.state_norm(self.state_to_text(prototypes))
        state_weights = self._masked_position_weights(
            self.state_position_logits, active_bool
        )
        state_delta = torch.einsum(
            "bqkd,qkl->bqld", state_tokens, state_weights
        )

        global_tokens = self.global_to_text(global_feature).view(
            batch_size, self.num_global_tokens, self.text_width
        )
        global_tokens = self.global_norm(global_tokens)
        global_weights = self._masked_position_weights(
            self.global_position_logits, active_bool
        )
        global_delta = torch.einsum(
            "bmd,qml->bqld", global_tokens, global_weights
        )

        # Padding positions stay exactly unchanged.  The class token embedding
        # itself remains the fixed anchor and receives only a bounded residual.
        active_positions = active_bool.to(static_inputs_embeds.dtype)
        active_positions = active_positions.view(
            1, num_queries, sequence_length, 1
        )
        context_delta = context_delta * active_positions
        state_delta = state_delta * active_positions
        global_delta = global_delta * active_positions

        learned_gates = torch.sigmoid(self.gate_logits)
        switches = self._component_switches(component_mode)
        effective_gates = learned_gates * learned_gates.new_tensor(switches)
        shared_part = self.residual_scale * effective_gates[0] * context_delta
        state_part = self.residual_scale * effective_gates[1] * state_delta
        global_part = self.residual_scale * effective_gates[2] * global_delta

        static = static_inputs_embeds.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )
        dynamic = static + shared_part + state_part + global_part

        attention_entropy = -(
            attention.float()
            * attention.float().clamp_min(1e-8).log()
        ).sum(dim=-1)
        attention_entropy = attention_entropy / math.log(
            max(attention.shape[-1], 2)
        )
        diagnostics: Dict[str, torch.Tensor] = {
            "gate_shared": effective_gates[0],
            "gate_state": effective_gates[1],
            "gate_global": effective_gates[2],
            "shared_residual_norm": shared_part.float().norm(dim=-1).mean(),
            "state_residual_norm": state_part.float().norm(dim=-1).mean(),
            "global_residual_norm": global_part.float().norm(dim=-1).mean(),
            "attention_entropy": attention_entropy.mean(),
            "attention_peak": attention.float().amax(dim=-1).mean(),
            "attention": attention,
            "state_prototypes": prototypes,
            "class_anchor": class_anchor,
        }
        diagnostics.update(self._prototype_diagnostics(prototypes))
        return dynamic, diagnostics
