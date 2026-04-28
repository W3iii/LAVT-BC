"""
data/dataset_ln.py
──────────────────
PyTorch Dataset for the LN (Lung Nodule) RIS dataset.

Annotation JSON format (per entry):
  {
    "image":      "CHEST1001_S0136.png",
    "mask":       "CHEST1001_S0136_cls1.png",   (or "" for negatives)
    "sentences":  ["benign lung nodule"],
    "is_pos":     1,                             (0 for negatives)
    "category":   1,                             (0 for normal slices)
    "patient_id": "CHEST1001"
  }

Images are single-channel uint8 PNGs (HU-windowed).
They are replicated to 3-channel RGB on-the-fly.

Masks are binary uint8 PNGs with values in {0, 1}.
When is_pos == 0, mask is "" → dataloader returns an empty (all-zero) mask.
"""

import os
import json
import random

import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

from transformers import BertTokenizer


class LNDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.split            = split
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms
        self.max_tokens       = 20

        # ── load annotation JSON ──────────────────────────────────────────
        ann_path = os.path.join(args.ln_dataset_root, 'annotations', f'{split}.json')
        with open(ann_path, 'r') as f:
            all_annotations = json.load(f)

        # ── negative sampling config ──────────────────────────────────────
        self.neg_ratio = getattr(args, 'neg_ratio', 2.0)
        self._pos = [a for a in all_annotations if a['is_pos'] == 1]
        self._neg = [a for a in all_annotations if a['is_pos'] == 0]

        if split == 'train' and self.neg_ratio > 0:
            self.resample_negatives(epoch=0)
        else:
            self.annotations = all_annotations

        self.image_dir = os.path.join(args.ln_dataset_root, 'images', split)
        self.mask_dir  = os.path.join(args.ln_dataset_root, 'masks',  split)

        # ── tokeniser (cache by unique sentence) ─────────────────────────
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
        self._token_cache = {}  # sentence_str → (input_ids_tensor, attn_mask_tensor)
        self._build_token_cache(all_annotations)

    # ── helpers ───────────────────────────────────────────────────────────

    def _build_token_cache(self, all_annotations):
        """Tokenise each unique sentence once and cache the result."""
        for ann in all_annotations:
            for s in ann['sentences']:
                if s in self._token_cache:
                    continue
                padded = [0] * self.max_tokens
                attn   = [0] * self.max_tokens
                ids = self.tokenizer.encode(text=s, add_special_tokens=True)
                ids = ids[:self.max_tokens]
                padded[:len(ids)] = ids
                attn[:len(ids)]   = [1] * len(ids)
                self._token_cache[s] = (
                    torch.tensor(padded).unsqueeze(0),
                    torch.tensor(attn).unsqueeze(0),
                )
        print(f'  Token cache: {len(self._token_cache)} unique sentences')

    def resample_negatives(self, epoch=0):
        """Image-grouped negative resampling.

        For every positive image, keep its 'paired' negatives — i.e. negatives
        that come from the SAME image (these are the "wrong prompt" entries
        that drive the contrastive signal).  Then add a controlled number of
        pure-normal-slice negatives (images that have NO positive entry at all).

        This preserves the (correct prompt vs wrong prompt) pairing on the same
        image, which is what teaches the model to differentiate prompts.
        """
        from collections import defaultdict

        rng = random.Random(42 + epoch)

        # group negatives by image
        neg_by_img = defaultdict(list)
        for ann in self._neg:
            neg_by_img[ann['image']].append(ann)

        # group positives by image
        pos_by_img = defaultdict(list)
        for ann in self._pos:
            pos_by_img[ann['image']].append(ann)

        sampled = []
        n_paired_neg = 0

        # Stage 1: for every positive image, keep all (or up to ratio*pos)
        # of its same-image negatives.  These are the "wrong prompt" pairs.
        for img, pos_anns in pos_by_img.items():
            sampled.extend(pos_anns)
            same_img_negs = neg_by_img.get(img, [])
            if not same_img_negs:
                continue
            # cap at neg_ratio * (pos count for this image), but at least keep
            # 1 if any exist, since the contrastive signal is image-local
            cap = max(1, int(round(len(pos_anns) * self.neg_ratio)))
            if len(same_img_negs) > cap:
                picked = rng.sample(same_img_negs, cap)
            else:
                picked = same_img_negs
            sampled.extend(picked)
            n_paired_neg += len(picked)

        # Stage 2: pure-normal-slice images (no positive entry at all).
        # These teach the model "even with cat 1~4 prompt, normal slice → no".
        # Control overall budget: ~30% of positive count by default.
        pure_normal_imgs = set(neg_by_img.keys()) - set(pos_by_img.keys())
        pure_normal_negs = [a for img in pure_normal_imgs
                              for a in neg_by_img[img]]

        pure_budget = int(len(self._pos) * 0.3)
        if len(pure_normal_negs) > pure_budget:
            pure_picked = rng.sample(pure_normal_negs, pure_budget)
        else:
            pure_picked = pure_normal_negs
        sampled.extend(pure_picked)

        rng.shuffle(sampled)
        self.annotations = sampled
        print(f'  [train] resample epoch {epoch}: '
              f'pos={len(self._pos)}, '
              f'paired-neg={n_paired_neg}, '
              f'pure-normal-neg={len(pure_picked)}/{len(pure_normal_negs)}, '
              f'total={len(sampled)}')

    def get_classes(self):
        return []

    def __len__(self):
        return len(self.annotations)

    # ── __getitem__ ───────────────────────────────────────────────────────

    def __getitem__(self, index):
        ann = self.annotations[index]

        # ── image: grayscale PNG → 3-channel RGB ─────────────────────────
        img_path = os.path.join(self.image_dir, ann['image'])
        img_gray = Image.open(img_path).convert('L')
        img      = Image.merge('RGB', [img_gray, img_gray, img_gray])

        # ── mask ─────────────────────────────────────────────────────────
        if ann['is_pos'] == 1 and ann['mask']:
            mask_path = os.path.join(self.mask_dir, ann['mask'])
            mask = Image.open(mask_path).convert('P')
        else:
            # negative sample → empty mask (same size as image)
            w, h = img_gray.size
            mask = Image.fromarray(np.zeros((h, w), dtype=np.uint8), mode='P')

        # ── joint spatial transforms ──────────────────────────────────────
        if self.image_transforms is not None:
            img, mask = self.image_transforms(img, mask)

        # ── sentence embedding (from cache) ──────────────────────────────
        sentences = ann['sentences']
        if self.split == 'train':
            choice_sent = np.random.choice(len(sentences))
        else:
            choice_sent = 0
        tensor_embeddings, attention_mask = self._token_cache[sentences[choice_sent]]

        meta = {
            'is_pos':   ann['is_pos'],
            'category': ann.get('category', -1),
        }

        return img, mask, tensor_embeddings, attention_mask, meta