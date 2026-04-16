"""
lib/_utils.py  (modified v2)
─────────────────────────────
Key changes vs original:
  1. exist_head uses x_c4 + x_c3 (high-level semantic features, pooled & concat)
  2. class token is broadcast-added (gated) to all language tokens + concat
  3. Both LAVT and LAVTOne share the same improvements
  4. Decoder stays clean — no return_feat needed
"""

from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

        # ── learnable class embedding ────────────────────────────────────
        # 5 classes: 0=normal/neg, 1=benign, 2=prob_benign,
        #            3=prob_suspicious, 4=suspicious
        self.class_embed = nn.Embedding(5, 768)
        self.class_pos_embed = nn.Parameter(torch.zeros(1, 768, 1))
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.normal_(self.class_pos_embed, std=0.02)

        # learnable gate: controls how much class info is injected
        self.class_gate = nn.Sequential(
            nn.Linear(768, 768),
            nn.Sigmoid(),
        )

        # ── existence head — lazy-built from x_c4 + x_c3 ────────────────
        self._exist_head_built = False

    def _build_exist_head(self, c4_channels, c3_channels):
        """Lazy-build so we don't need to know channel counts at __init__."""
        if self._exist_head_built:
            return
        in_ch = c4_channels + c3_channels
        self.exist_head = nn.Sequential(
            nn.Linear(in_ch, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        device = self.class_embed.weight.device
        self.exist_head = self.exist_head.to(device)
        self._exist_head_built = True

    def _inject_class_token(self, l_feats, l_mask, category):
        """
        Inject class info into language features.

        Strategy: concat a class token AND broadcast-add a gated class
        signal to all language tokens.

        l_feats: (B, 768, seq_len)
        l_mask:  (B, seq_len, 1)
        category: (B,) long
        """
        B = l_feats.shape[0]
        cls_emb = self.class_embed(category)                      # (B, 768)

        # 1) broadcast-add gated class signal to ALL tokens
        gate = self.class_gate(cls_emb)                           # (B, 768)
        gated_signal = (cls_emb * gate).unsqueeze(-1)             # (B, 768, 1)
        l_feats = l_feats + gated_signal                          # broadcast

        # 2) also concat as extra token
        cls_token = cls_emb.unsqueeze(-1) + self.class_pos_embed  # (B, 768, 1)
        l_feats = torch.cat([l_feats, cls_token], dim=-1)         # (B, 768, seq+1)
        cls_mask = torch.ones(B, 1, 1, device=l_mask.device)
        l_mask = torch.cat([l_mask, cls_mask], dim=1)             # (B, seq+1, 1)

        return l_feats, l_mask

    def _exist_forward(self, x_c4, x_c3):
        """Pool x_c4 and x_c3, concat, and predict existence."""
        self._build_exist_head(x_c4.shape[1], x_c3.shape[1])
        c4_pool = F.adaptive_avg_pool2d(x_c4, 1).flatten(1)      # (B, c4_ch)
        c3_pool = F.adaptive_avg_pool2d(x_c3, 1).flatten(1)      # (B, c3_ch)
        exist_feat = torch.cat([c4_pool, c3_pool], dim=1)         # (B, c4+c3)
        return self.exist_head(exist_feat).squeeze(-1)             # (B,)

    def forward(self, x, l_feats, l_mask, category=None):
        input_shape = x.shape[-2:]

        if category is not None:
            l_feats, l_mask = self._inject_class_token(l_feats, l_mask, category)

        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features

        # segmentation
        seg = self.classifier(x_c4, x_c3, x_c2, x_c1)
        seg = F.interpolate(seg, size=input_shape, mode='bilinear',
                            align_corners=True)

        # existence from x_c4 + x_c3
        exist_out = self._exist_forward(x_c4, x_c3)

        return seg, exist_out


class LAVT(_LAVTSimpleDecode):
    pass


###############################################
# LAVT One: put BERT inside the overall model #
###############################################
class _LAVTOneSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(_LAVTOneSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

        # ── learnable class embedding ────────────────────────────────────
        self.class_embed = nn.Embedding(5, 768)
        self.class_pos_embed = nn.Parameter(torch.zeros(1, 768, 1))
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.normal_(self.class_pos_embed, std=0.02)

        self.class_gate = nn.Sequential(
            nn.Linear(768, 768),
            nn.Sigmoid(),
        )

        # ── existence head — lazy-built ──────────────────────────────────
        self._exist_head_built = False

    def _build_exist_head(self, c4_channels, c3_channels):
        if self._exist_head_built:
            return
        in_ch = c4_channels + c3_channels
        self.exist_head = nn.Sequential(
            nn.Linear(in_ch, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        device = self.class_embed.weight.device
        self.exist_head = self.exist_head.to(device)
        self._exist_head_built = True

    def _inject_class_token(self, l_feats, l_mask, category):
        B = l_feats.shape[0]
        cls_emb = self.class_embed(category)
        gate = self.class_gate(cls_emb)
        gated_signal = (cls_emb * gate).unsqueeze(-1)
        l_feats = l_feats + gated_signal
        cls_token = cls_emb.unsqueeze(-1) + self.class_pos_embed
        l_feats = torch.cat([l_feats, cls_token], dim=-1)
        cls_mask = torch.ones(B, 1, 1, device=l_mask.device)
        l_mask = torch.cat([l_mask, cls_mask], dim=1)
        return l_feats, l_mask

    def _exist_forward(self, x_c4, x_c3):
        self._build_exist_head(x_c4.shape[1], x_c3.shape[1])
        c4_pool = F.adaptive_avg_pool2d(x_c4, 1).flatten(1)
        c3_pool = F.adaptive_avg_pool2d(x_c3, 1).flatten(1)
        exist_feat = torch.cat([c4_pool, c3_pool], dim=1)
        return self.exist_head(exist_feat).squeeze(-1)

    def forward(self, x, text, l_mask, category=None):
        input_shape = x.shape[-2:]
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]
        l_feats = l_feats.permute(0, 2, 1)
        l_mask = l_mask.unsqueeze(dim=-1)

        if category is not None:
            l_feats, l_mask = self._inject_class_token(l_feats, l_mask, category)

        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features

        seg = self.classifier(x_c4, x_c3, x_c2, x_c1)
        seg = F.interpolate(seg, size=input_shape, mode='bilinear',
                            align_corners=True)

        exist_out = self._exist_forward(x_c4, x_c3)

        return seg, exist_out


class LAVTOne(_LAVTOneSimpleDecode):
    pass