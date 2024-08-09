import math

import torch
import torch.nn as nn


def build_abstractor(config):
    abstractor = AbstractorModel(config)
    return abstractor


class PositionEmbeddingSine(nn.Module):
    def __init__(
        self,
        feature_size,
        num_pos_feats=128,
        temperature=10000,
        normalize=False,
        scale=None,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        not_mask = torch.ones((self.feature_size[0], self.feature_size[1]))  # H x W
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)  # (H X W) X C
        return pos.to(tensor.device)


class AbstractorModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )
        self.abstractor = nn.TransformerDecoder(layer, num_layers=config.num_layers)
        self.query_embeds = torch.nn.Parameter(
            torch.randn(config.num_query, 1, config.hidden_dim)
        )

    def forward(self, vision_embs, feature_size=None):
        batch_size, _, hidden_dim = vision_embs.shape
        # Currently, pos_emb is not ued, since clip has added pos_emb. 
        if self.config.add_pos_emd and feature_size:
            pos_emb = PositionEmbeddingSine(
                feature_size, hidden_dim // 2, normalize=True
            )
            pos_emb = pos_emb(vision_embs)[None, :, :].repeat(batch_size, 1, 1)
            vision_embs = vision_embs + pos_emb
        vision_embs = vision_embs.contiguous().permute(1, 0, 2)
        query_embeds = self.query_embeds.repeat(1, batch_size, 1)
        vision_embs = self.abstractor(query_embeds, vision_embs).permute(1, 0, 2)
        return vision_embs
