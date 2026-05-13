from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel
from bert.tokenization_bert import BertTokenizer


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask):
        input_shape = x.shape[-2:]
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVT(_LAVTSimpleDecode):
    pass


###############################################
# LAVT One: BERT + learnable prompt inside    #
###############################################
# Seed adjectives for the positive soft prompt (nodule cues).
_POS_INIT_ADJECTIVES = [
    "small", "irregular", "suspicious", "abnormal",
    "round", "dense", "bright", "malignant",
]
# Seed adjectives for the negative soft prompt (normal / no-nodule cues).
_NEG_INIT_ADJECTIVES = [
    "normal", "clear", "absent", "unremarkable",
    "negative", "healthy", "nonfocal", "clean",
]
_POS_PREFIX_TEXT = "a slice of chest ct with"
_NEG_PREFIX_TEXT = "a normal chest ct slice without"
_SUFFIX_TEXT = "lung nodule"


def _pad_adjectives(adjs: list, n: int) -> list:
    if n <= len(adjs):
        return adjs[:n]
    return adjs + [adjs[-1]] * (n - len(adjs))


class _LAVTOneSimpleDecode(nn.Module):
    """
    Dual-prompt image-text segmentation head.

    Internal scaffolds:
      positive: "[CLS] a slice of chest ct with [V1]...[Vn] lung nodule [SEP]"
      negative: "[CLS] a normal chest ct slice without [N1]...[Nn] lung nodule [SEP]"

    Training: ``forward(x, has_nodule=...)`` runs both BERT passes and merges
    the per-sample language features by ``has_nodule`` (image backbone runs once).

    Inference: ``forward(x, prompt_type='positive'|'negative')`` runs a single
    BERT pass with the chosen scaffold. Dual-prompt suppression is implemented
    at the test loop by combining the two single-prompt outputs.
    """

    def __init__(self, backbone, classifier, args):
        super(_LAVTOneSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

        n_soft = int(args.n_soft_tokens)
        self.n_soft_tokens = n_soft

        tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
        cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
        suffix_ids = tokenizer.encode(_SUFFIX_TEXT, add_special_tokens=False)

        def _build_scaffold(prefix_text: str, seed_adjectives: list,
                            buf_prefix: str):
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            prefix_full = [cls_id] + prefix_ids
            suffix_full = suffix_ids + [sep_id]

            adjs = _pad_adjectives(seed_adjectives, n_soft)
            adj_token_ids = tokenizer.encode(" ".join(adjs),
                                             add_special_tokens=False)
            if len(adj_token_ids) >= n_soft:
                adj_token_ids = adj_token_ids[:n_soft]
            else:
                pad_id = adj_token_ids[-1] if adj_token_ids else tokenizer.unk_token_id
                adj_token_ids = adj_token_ids + [pad_id] * (n_soft - len(adj_token_ids))

            self.register_buffer(
                f"{buf_prefix}_prefix_input_ids",
                torch.tensor(prefix_full, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"{buf_prefix}_suffix_input_ids",
                torch.tensor(suffix_full, dtype=torch.long),
                persistent=False,
            )

            full_ids = torch.tensor(prefix_full + adj_token_ids + suffix_full,
                                    dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                self.text_encoder.eval()
                attn = torch.ones_like(full_ids)
                hidden = self.text_encoder(full_ids, attention_mask=attn)[0][0]
                soft_start = len(prefix_full)
                soft_init = hidden[soft_start:soft_start + n_soft].clone()
            return len(prefix_full) + n_soft + len(suffix_full), soft_init

        pos_seq_len, pos_soft_init = _build_scaffold(
            _POS_PREFIX_TEXT, _POS_INIT_ADJECTIVES, "pos")
        neg_seq_len, neg_soft_init = _build_scaffold(
            _NEG_PREFIX_TEXT, _NEG_INIT_ADJECTIVES, "neg")
        # Both scaffolds use 6-word prefixes + 2-word suffix with bert-base-uncased,
        # so they tokenize to identical lengths. We rely on this to merge per-sample
        # features without padding.
        assert pos_seq_len == neg_seq_len, (
            f"positive/negative scaffold length mismatch: "
            f"{pos_seq_len} vs {neg_seq_len}")
        self.seq_len = pos_seq_len

        self.text_encoder.train()
        self.pos_soft_tokens = nn.Parameter(pos_soft_init)
        self.neg_soft_tokens = nn.Parameter(neg_soft_init)

    def _encode_prompt(self, batch_size: int, prompt_type: str):
        word_emb = self.text_encoder.embeddings.word_embeddings
        if prompt_type == "positive":
            prefix_ids = self.pos_prefix_input_ids
            suffix_ids = self.pos_suffix_input_ids
            soft = self.pos_soft_tokens
        elif prompt_type == "negative":
            prefix_ids = self.neg_prefix_input_ids
            suffix_ids = self.neg_suffix_input_ids
            soft = self.neg_soft_tokens
        else:
            raise ValueError(f"unknown prompt_type: {prompt_type}")

        prefix_emb = word_emb(prefix_ids)                   # (P, 768)
        suffix_emb = word_emb(suffix_ids)                   # (S, 768)
        seq_emb = torch.cat([prefix_emb, soft, suffix_emb], dim=0)
        seq_emb = seq_emb.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

        attn_mask = torch.ones(batch_size, self.seq_len,
                               dtype=torch.long, device=seq_emb.device)
        out = self.text_encoder(inputs_embeds=seq_emb, attention_mask=attn_mask)
        return out[0], attn_mask                            # (B, L, 768), (B, L)

    def _run_backbone(self, x, l_feats, l_mask):
        input_shape = x.shape[-2:]
        l_feats = l_feats.permute(0, 2, 1)                  # (B, 768, L)
        l_mask = l_mask.unsqueeze(-1)                       # (B, L, 1)
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        out = self.classifier(x_c4, x_c3, x_c2, x_c1)
        out = F.interpolate(out, size=input_shape,
                            mode='bilinear', align_corners=True)
        return out

    def forward(self, x, has_nodule=None, prompt_type=None):
        B = x.shape[0]
        if self.training:
            assert has_nodule is not None, \
                "training requires has_nodule for dual-prompt selection"
            l_pos, m_pos = self._encode_prompt(B, "positive")
            l_neg, _ = self._encode_prompt(B, "negative")
            sel = has_nodule.view(B, 1, 1).to(l_pos.dtype)
            l_feats = sel * l_pos + (1.0 - sel) * l_neg
            l_mask = m_pos  # both scaffolds are full-attended; masks are identical
        else:
            assert prompt_type in ("positive", "negative"), \
                "inference requires prompt_type in {'positive', 'negative'}"
            l_feats, l_mask = self._encode_prompt(B, prompt_type)
        return self._run_backbone(x, l_feats, l_mask)


class LAVTOne(_LAVTOneSimpleDecode):
    pass
