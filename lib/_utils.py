from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        # learnable class embedding (concat as extra token)
        self.class_embed = nn.Embedding(5, 768)
        self.class_pos_embed = nn.Parameter(torch.zeros(1, 768, 1))
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.normal_(self.class_pos_embed, std=0.02)
        # existence head
        feat_ch = classifier.feat_channels
        self.exist_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(feat_ch * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def _inject_class_token(self, l_feats, l_mask, category):
        """Concat learnable class token to language sequence."""
        B = l_feats.shape[0]
        cls_emb = self.class_embed(category).unsqueeze(-1)        # (B, 768, 1)
        cls_emb = cls_emb + self.class_pos_embed                  # add position info
        l_feats = torch.cat([l_feats, cls_emb], dim=-1)           # (B, 768, seq_len+1)
        cls_mask = torch.ones(B, 1, 1, device=l_mask.device)
        l_mask = torch.cat([l_mask, cls_mask], dim=1)             # (B, seq_len+1, 1)
        return l_feats, l_mask

    def forward(self, x, l_feats, l_mask, category=None):
        input_shape = x.shape[-2:]
        if category is not None:
            l_feats, l_mask = self._inject_class_token(l_feats, l_mask, category)
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        seg, feat = self.classifier(x_c4, x_c3, x_c2, x_c1, return_feat=True)
        seg = F.interpolate(seg, size=input_shape, mode='bilinear', align_corners=True)
        exist_out = self.exist_head(feat).squeeze(-1)  # (B,)
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
        # learnable class embedding (concat as extra token)
        self.class_embed = nn.Embedding(5, 768)
        self.class_pos_embed = nn.Parameter(torch.zeros(1, 768, 1))
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.normal_(self.class_pos_embed, std=0.02)
        feat_ch = classifier.feat_channels
        self.exist_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_ch, 1),
        )

    def _inject_class_token(self, l_feats, l_mask, category):
        """Concat learnable class token to language sequence."""
        B = l_feats.shape[0]
        cls_emb = self.class_embed(category).unsqueeze(-1)        # (B, 768, 1)
        cls_emb = cls_emb + self.class_pos_embed
        l_feats = torch.cat([l_feats, cls_emb], dim=-1)           # (B, 768, seq_len+1)
        cls_mask = torch.ones(B, 1, 1, device=l_mask.device)
        l_mask = torch.cat([l_mask, cls_mask], dim=1)             # (B, seq_len+1, 1)
        return l_feats, l_mask

    def forward(self, x, text, l_mask, category=None):
        input_shape = x.shape[-2:]
        ### language inference ###
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # (B, N_l, 768)
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l)
        l_mask = l_mask.unsqueeze(dim=-1)    # (B, N_l, 1)
        ##########################
        if category is not None:
            l_feats, l_mask = self._inject_class_token(l_feats, l_mask, category)
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        seg, feat = self.classifier(x_c4, x_c3, x_c2, x_c1, return_feat=True)
        seg = F.interpolate(seg, size=input_shape, mode='bilinear', align_corners=True)
        exist_out = self.exist_head(feat).squeeze(-1)  # (B,)
        return seg, exist_out


class LAVTOne(_LAVTOneSimpleDecode):
    pass
