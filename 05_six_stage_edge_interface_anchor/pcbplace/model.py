from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dropout: float = 0.1


class MaskedPolicy(nn.Module):
    """Autoregressive policy with semantic auxiliary heads.

    Core policy scores a variable-size action space from action features.
    Additional heads predict:
      - a coarse-to-fine region heatmap head over region_grid_shape cells
      - semantic component class (core / power / interface / ...)
    """

    def __init__(
        self,
        obs_dim: int,
        cfg: ModelConfig,
        action_feat_dim: int = 4,
        region_grid_shape: Tuple[int, int] = (6, 6),
        num_region_heatmap_bins: int = 6,
        num_semantic_classes: int = 12,
        num_side_preferences: int = 5,
        num_subzones: int = 6,
        num_pairwise_relations: int = 3,
    ):
        super().__init__()
        self.cfg = cfg
        self.action_feat_dim = int(action_feat_dim)
        self.region_grid_shape = (int(region_grid_shape[0]), int(region_grid_shape[1]))
        self.num_region_heatmap_bins = int(num_region_heatmap_bins)
        self.num_semantic_classes = int(num_semantic_classes)
        self.num_side_preferences = int(num_side_preferences)
        self.num_subzones = int(num_subzones)
        self.num_pairwise_relations = int(num_pairwise_relations)

        self.obs_proj = nn.Linear(obs_dim, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead, dropout=cfg.dropout, batch_first=True
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_mlp = nn.Sequential(
            nn.Linear(self.action_feat_dim, cfg.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        self.region_heatmap_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.d_model, self.num_region_heatmap_bins),
        )
        self.semantic_class_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.d_model, self.num_semantic_classes),
        )
        self.side_preference_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.d_model, self.num_side_preferences),
        )
        self.subzone_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.d_model, self.num_subzones),
        )
        self.pairwise_relation_mlp = nn.Sequential(
            nn.Linear(cfg.d_model * 4, cfg.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.d_model, self.num_pairwise_relations),
        )

    def encode(
        self,
        obs_tokens: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.obs_proj(obs_tokens)
        x = self.tr(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        return x

    def query_last(
        self,
        obs_tokens: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enc = self.encode(obs_tokens, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.q_proj(enc[:, -1, :])

    def action_logits_from_query(self, q: torch.Tensor, action_feat: torch.Tensor) -> torch.Tensor:
        """Return action logits.

        Supports both the original single-action-space form and the batched
        multi-board form:
          q:           [B, D]
          action_feat: [A, F]       -> logits [B, A]
          action_feat: [B, A, F]    -> logits [B, A]

        The 3-D path is the key for true multi-board GPU forward: each board
        can have its own padded action feature matrix while sharing one model
        call. Padding is handled by the caller through the action mask.
        """
        if action_feat.shape[-1] != self.action_feat_dim:
            raise ValueError(
                f"action_feat dim mismatch: got {int(action_feat.shape[-1])}, "
                f"expected {int(self.action_feat_dim)}"
            )
        k = self.k_mlp(action_feat)
        scale = math.sqrt(self.cfg.d_model)
        if action_feat.dim() == 3:
            return torch.einsum("bd,bad->ba", q, k) / scale
        return (q @ k.t()) / scale

    def action_logits_from_query_batched(self, q: torch.Tensor, action_feat: torch.Tensor) -> torch.Tensor:
        """Explicit batched alias for q [B,D], action_feat [B,A,F]."""
        if action_feat.dim() != 3:
            raise ValueError("action_feat must be [B,A,F] for batched action logits")
        return self.action_logits_from_query(q, action_feat)

    def region_heatmap_logits(self, q: torch.Tensor) -> torch.Tensor:
        return self.region_heatmap_head(q)

    def semantic_class_logits(self, q: torch.Tensor) -> torch.Tensor:
        return self.semantic_class_head(q)

    def side_preference_logits(self, q: torch.Tensor) -> torch.Tensor:
        return self.side_preference_head(q)

    def subzone_logits(self, q: torch.Tensor) -> torch.Tensor:
        return self.subzone_head(q)

    def pairwise_relation_logits(self, q_current: torch.Tensor, q_peer: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([q_current, q_peer, torch.abs(q_current - q_peer), q_current * q_peer], dim=-1)
        return self.pairwise_relation_mlp(feat)

    def logits_last(
        self,
        obs_tokens: torch.Tensor,
        action_feat: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.query_last(obs_tokens, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.action_logits_from_query(q, action_feat)

    def forward(
        self,
        obs_tokens: torch.Tensor,
        action_feat: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Run one complete policy forward pass.

        Args:
            obs_tokens:
                Observation/context tokens with shape ``[B, L, obs_dim]``.
            action_feat:
                Action features with shape ``[A, F]`` for a shared action
                space or ``[B, A, F]`` for per-board action spaces.
            attn_mask:
                Optional Transformer attention mask.
            key_padding_mask:
                Optional token padding mask with shape ``[B, L]``. ``True``
                entries are padding tokens.

        Returns:
            A stable dictionary containing every model output:

            ``action_logits``
                Action scores with shape ``[B, A]``.
            ``region_heatmap_logits``
                Region heatmap logits with shape ``[B, R]``.
            ``semantic_class_logits``
                Semantic class logits with shape ``[B, C]``.
            ``side_preference_logits``
                Side preference logits with shape ``[B, S]``.
            ``subzone_logits``
                Subzone logits with shape ``[B, Z]``.
            ``pairwise_relation_logits``
                Pairwise relation logits with shape ``[B, P]``. This is
                ``None`` when the sequence contains only the current token.
            ``encoded_tokens``
                Transformer output with shape ``[B, L, d_model]``.
            ``query``
                Projected query of the final/current token, shape ``[B, d_model]``.

        The final token is treated as the current component, matching the
        existing training and inference paths. Pairwise prediction pools all
        preceding valid tokens as the peer representation.
        """
        encoded_tokens = self.encode(
            obs_tokens,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        query = self.q_proj(encoded_tokens[:, -1, :])
        action_logits = self.action_logits_from_query(query, action_feat)

        region_heatmap_logits = self.region_heatmap_logits(query)
        semantic_class_logits = self.semantic_class_logits(query)
        side_preference_logits = self.side_preference_logits(query)
        subzone_logits = self.subzone_logits(query)

        pairwise_relation_logits: Optional[torch.Tensor] = None
        if encoded_tokens.shape[1] > 1:
            peer_tokens = encoded_tokens[:, :-1, :]
            if key_padding_mask is None:
                peer_query = peer_tokens.mean(dim=1)
            else:
                if key_padding_mask.shape[:2] != obs_tokens.shape[:2]:
                    raise ValueError(
                        "key_padding_mask shape mismatch: "
                        f"got {tuple(key_padding_mask.shape)}, "
                        f"expected {(int(obs_tokens.shape[0]), int(obs_tokens.shape[1]))}"
                    )
                peer_valid = (~key_padding_mask[:, :-1]).to(peer_tokens.dtype).unsqueeze(-1)
                peer_count = peer_valid.sum(dim=1).clamp(min=1.0)
                peer_query = (peer_tokens * peer_valid).sum(dim=1) / peer_count
            pairwise_relation_logits = self.pairwise_relation_logits(query, peer_query)

        return {
            "action_logits": action_logits,
            "region_heatmap_logits": region_heatmap_logits,
            "semantic_class_logits": semantic_class_logits,
            "side_preference_logits": side_preference_logits,
            "subzone_logits": subzone_logits,
            "pairwise_relation_logits": pairwise_relation_logits,
            "encoded_tokens": encoded_tokens,
            "query": query,
        }

    def forward_with_region(
        self,
        obs_tokens: torch.Tensor,
        action_feat: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward-compatible wrapper for the former three-output API."""
        outputs = self.forward(
            obs_tokens,
            action_feat,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        action_logits = outputs["action_logits"]
        region_heatmap_logits = outputs["region_heatmap_logits"]
        semantic_class_logits = outputs["semantic_class_logits"]
        assert action_logits is not None
        assert region_heatmap_logits is not None
        assert semantic_class_logits is not None
        return action_logits, region_heatmap_logits, semantic_class_logits
