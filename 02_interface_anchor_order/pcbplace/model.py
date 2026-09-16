from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

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
      - semantic region type (edge side / core / free)
      - semantic component class (core / power / interface / ...)
    """

    def __init__(
        self,
        obs_dim: int,
        cfg: ModelConfig,
        action_feat_dim: int = 4,
        region_grid_shape: Tuple[int, int] = (6, 6),
        num_region_types: int = 6,
        num_semantic_classes: int = 12,
        num_side_preferences: int = 5,
        num_subzones: int = 6,
        num_pairwise_relations: int = 3,
    ):
        super().__init__()
        self.cfg = cfg
        self.action_feat_dim = int(action_feat_dim)
        self.region_grid_shape = (int(region_grid_shape[0]), int(region_grid_shape[1]))
        self.num_region_types = int(num_region_types)
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

        self.region_type_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.d_model, self.num_region_types),
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

    def encode(self, obs_tokens: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.obs_proj(obs_tokens)
        x = self.tr(x, mask=attn_mask)
        return x

    def query_last(self, obs_tokens: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        enc = self.encode(obs_tokens, attn_mask=attn_mask)
        return self.q_proj(enc[:, -1, :])

    def action_logits_from_query(self, q: torch.Tensor, action_feat: torch.Tensor) -> torch.Tensor:
        k = self.k_mlp(action_feat)
        return (q @ k.t()) / math.sqrt(self.cfg.d_model)

    def region_type_logits(self, q: torch.Tensor) -> torch.Tensor:
        return self.region_type_head(q)

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
        self, obs_tokens: torch.Tensor, action_feat: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        q = self.query_last(obs_tokens, attn_mask=attn_mask)
        return self.action_logits_from_query(q, action_feat)

    def forward_with_region(
        self,
        obs_tokens: torch.Tensor,
        action_feat: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.query_last(obs_tokens, attn_mask=attn_mask)
        logits = self.action_logits_from_query(q, action_feat)
        region_type_logits = self.region_type_logits(q)
        semantic_class_logits = self.semantic_class_logits(q)
        return logits, region_type_logits, semantic_class_logits
