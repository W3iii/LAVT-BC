"""
test_language_disc.py
─────────────────────
Test whether language prompts actually affect segmentation output.

For each positive sample, run inference with all 4 category prompts and compare:
  1. Are the 4 masks identical? (language has zero effect)
  2. How different are they? (IoU between mask pairs)
  3. Does the correct prompt produce the best IoU with GT?

Output:
  - Console summary
  - language_disc_results.json with per-sample and aggregate stats
  - Optional: side-by-side visualization (4 masks + GT)

Usage:
    python test_language_disc.py \
        --resume ./checkpoints/ln/model_best_lavt_ln.pth \
        --swin_type base --window12 --img_size 384 \
        --bert_tokenizer dmis-lab/biobert-base-cased-v1.2 \
        --ck_bert dmis-lab/biobert-base-cased-v1.2 \
        --max_samples 200 \
        --save_vis
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.utils.data
from torch.nn import functional as F
from transformers import BertModel, BertTokenizer

from lib import segmentation
import transforms as T
import utils


# ── 4 category prompts ───────────────────────────────────────────────────────

CAT_PROMPTS = {
    1: "benign lung nodule",
    2: "probably lung nodule",
    3: "Probably Suspicious lung nodule",
    4: "suspicious lung nodule",
}

CAT_NAMES = {
    1: "benign",
    2: "prob_benign",
    3: "prob_suspicious",
    4: "suspicious",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def tokenize_sentence(tokenizer, sentence, max_tokens=20):
    padded = [0] * max_tokens
    attn   = [0] * max_tokens
    ids = tokenizer.encode(text=sentence, add_special_tokens=True)
    ids = ids[:max_tokens]
    padded[:len(ids)] = ids
    attn[:len(ids)]   = [1] * len(ids)
    return (torch.tensor(padded).unsqueeze(0),
            torch.tensor(attn).unsqueeze(0))


def mask_iou(m1, m2):
    """IoU between two binary masks (numpy)."""
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return inter / union if union > 0 else 1.0  # both empty = identical


def run_inference(model, bert_model, image, ids_t, mask_t, category, device):
    """Run model forward, return binary mask (H, W) numpy."""
    ids_t  = ids_t.to(device)
    mask_t = mask_t.to(device)
    cat_t  = torch.tensor([category], device=device)

    if bert_model is not None:
        last_hidden = bert_model(ids_t, attention_mask=mask_t)[0]
        embedding   = last_hidden.permute(0, 2, 1)
        seg_out, exist_out = model(
            image, embedding, l_mask=mask_t.unsqueeze(-1), category=cat_t)
    else:
        seg_out, exist_out = model(
            image, ids_t, l_mask=mask_t, category=cat_t)

    pred_mask = seg_out.cpu().argmax(1).numpy()[0]  # (H, W)
    exist_prob = torch.sigmoid(exist_out).item()
    return pred_mask, exist_prob


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Language discriminability test')
    parser.add_argument('--ln_dataset_root', default='../dataset')
    parser.add_argument('--split', default='test')
    parser.add_argument('--model', default='lavt')
    parser.add_argument('--swin_type', default='base')
    parser.add_argument('--window12', action='store_true')
    parser.add_argument('--mha', default='')
    parser.add_argument('--fusion_drop', default=0.0, type=float)
    parser.add_argument('--img_size', default=384, type=int)
    parser.add_argument('--resume', required=True)
    parser.add_argument('--bert_tokenizer', default='bert-base-uncased')
    parser.add_argument('--ck_bert', default='bert-base-uncased')
    parser.add_argument('--neg_ratio', default=2.0, type=float)
    parser.add_argument('--pretrained_swin_weights', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('--max_samples', default=0, type=int,
                        help='Max positive samples to test (0=all)')
    parser.add_argument('--save_vis', action='store_true',
                        help='Save side-by-side visualization')
    parser.add_argument('--output_dir', default='./language_disc_results')
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── dataset (only positive samples) ──────────────────────────────────
    from data.dataset_ln import LNDataset
    transform = T.Compose([
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = LNDataset(args, split=args.split, image_transforms=transform,
                        target_transforms=None, eval_mode=True)

    pos_indices = [i for i, ann in enumerate(dataset.annotations)
                   if ann['is_pos'] == 1]
    if args.max_samples > 0:
        pos_indices = pos_indices[:args.max_samples]
    print(f'Testing {len(pos_indices)} positive samples with all 4 prompts')

    # ── tokenize all 4 prompts ───────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
    prompt_tokens = {}
    for cat, sent in CAT_PROMPTS.items():
        prompt_tokens[cat] = tokenize_sentence(tokenizer, sent)

    # ── model ────────────────────────────────────────────────────────────
    single_model = segmentation.__dict__[args.model](pretrained='', args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'])
    model = single_model.to(device)
    model.eval()

    if args.model != 'lavt_one':
        single_bert = BertModel.from_pretrained(args.ck_bert)
        single_bert.pooler = None
        if 'bert_model' in checkpoint:
            single_bert.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert.to(device)
        bert_model.eval()
    else:
        bert_model = None

    # ── output dir ───────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_vis:
        vis_dir = os.path.join(args.output_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)

    # ── run ──────────────────────────────────────────────────────────────
    all_identical = 0
    correct_prompt_best = 0
    pairwise_ious = []  # IoU between masks from different prompts
    gt_ious_by_prompt = {c: [] for c in [1, 2, 3, 4]}  # IoU with GT per prompt
    per_sample = []

    with torch.no_grad():
        for count, idx in enumerate(pos_indices):
            if (count + 1) % 50 == 0:
                print(f'  {count+1}/{len(pos_indices)}...')

            image, target, _, _, meta = dataset[idx]
            image = image.unsqueeze(0).to(device)
            target_np = target.numpy()
            gt_cat = meta['category'].item() if isinstance(meta['category'], torch.Tensor) else meta['category']
            ann = dataset.annotations[idx]

            # Run inference with all 4 prompts
            masks = {}
            exist_probs = {}
            gt_ious = {}
            for cat in [1, 2, 3, 4]:
                ids_t, mask_t = prompt_tokens[cat]
                pred, ep = run_inference(
                    model, bert_model, image, ids_t, mask_t, cat, device)
                masks[cat] = pred
                exist_probs[cat] = ep

                # IoU with GT
                inter = np.logical_and(pred > 0, target_np > 0).sum()
                union = np.logical_or(pred > 0, target_np > 0).sum()
                gt_ious[cat] = inter / union if union > 0 else 0.0
                gt_ious_by_prompt[cat].append(gt_ious[cat])

            # Check if all 4 masks are identical
            is_identical = all(
                np.array_equal(masks[1], masks[c]) for c in [2, 3, 4]
            )
            if is_identical:
                all_identical += 1

            # Pairwise IoU between different prompt masks
            cats = [1, 2, 3, 4]
            pair_ious = []
            for i in range(len(cats)):
                for j in range(i + 1, len(cats)):
                    piou = mask_iou(masks[cats[i]] > 0, masks[cats[j]] > 0)
                    pair_ious.append(piou)
            avg_pair_iou = np.mean(pair_ious)
            pairwise_ious.append(avg_pair_iou)

            # Does correct prompt give best IoU?
            best_cat = max(gt_ious, key=gt_ious.get)
            if best_cat == gt_cat:
                correct_prompt_best += 1

            sample_info = {
                'image': ann['image'],
                'gt_category': gt_cat,
                'identical': is_identical,
                'avg_pairwise_iou': round(avg_pair_iou, 4),
                'best_prompt_cat': int(best_cat),
                'gt_ious': {str(c): round(v, 4) for c, v in gt_ious.items()},
                'exist_probs': {str(c): round(v, 4) for c, v in exist_probs.items()},
            }
            per_sample.append(sample_info)

            # ── save visualization ───────────────────────────────────────
            if args.save_vis and count < 100:
                from PIL import Image as PILImage
                orig_img = PILImage.open(
                    os.path.join(dataset.image_dir, ann['image'])
                ).convert('L')
                H, W = masks[1].shape
                orig_img = orig_img.resize((W, H), PILImage.BILINEAR)
                gray = np.array(orig_img)

                panels = []
                # GT panel
                gt_rgb = np.stack([gray, gray, gray], axis=-1).copy()
                gt_bin = target_np > 0
                gt_rgb[gt_bin, 1] = np.clip(
                    gray[gt_bin] * 0.55 + 255 * 0.45, 0, 255).astype(np.uint8)
                panels.append(gt_rgb)

                # 4 prompt panels
                colors = {1: (0, 200, 0), 2: (200, 200, 0),
                          3: (255, 128, 0), 4: (255, 0, 0)}
                for cat in [1, 2, 3, 4]:
                    rgb = np.stack([gray, gray, gray], axis=-1).copy()
                    m = masks[cat] > 0
                    if m.any():
                        for ch in range(3):
                            rgb[m, ch] = np.clip(
                                gray[m] * 0.55 + colors[cat][ch] * 0.45,
                                0, 255).astype(np.uint8)
                    panels.append(rgb)

                canvas = np.concatenate(panels, axis=1)
                # Add labels
                img_stem = os.path.splitext(ann['image'])[0]
                tag = 'SAME' if is_identical else 'DIFF'
                save_name = f'{img_stem}_gt{gt_cat}_{tag}.png'
                PILImage.fromarray(canvas).save(os.path.join(vis_dir, save_name))

    # ── aggregate ────────────────────────────────────────────────────────
    n = len(pos_indices)
    identical_rate = all_identical / n if n > 0 else 0
    correct_best_rate = correct_prompt_best / n if n > 0 else 0
    mean_pair_iou = float(np.mean(pairwise_ious)) if pairwise_ious else 0

    print('\n' + '=' * 60)
    print('Language Discriminability Results')
    print('=' * 60)
    print(f'Samples tested: {n}')
    print(f'\n  All 4 masks identical: {all_identical}/{n} ({identical_rate*100:.1f}%)')
    print(f'  Mean pairwise IoU between prompts: {mean_pair_iou*100:.1f}%')
    print(f'    (100% = masks always identical, lower = more discriminative)')
    print(f'\n  Correct prompt gives best GT IoU: {correct_prompt_best}/{n} ({correct_best_rate*100:.1f}%)')
    print(f'    (random chance = 25%)')

    print(f'\n  Mean GT IoU per prompt:')
    for cat in [1, 2, 3, 4]:
        m = np.mean(gt_ious_by_prompt[cat]) if gt_ious_by_prompt[cat] else 0
        print(f'    cls{cat} ({CAT_NAMES[cat]}): {m*100:.2f}%')

    # ── per GT category breakdown ────────────────────────────────────────
    print(f'\n  Per GT-category breakdown (does correct prompt win?):')
    for gt_c in [1, 2, 3, 4]:
        samples_c = [s for s in per_sample if s['gt_category'] == gt_c]
        if not samples_c:
            continue
        n_c = len(samples_c)
        n_correct = sum(1 for s in samples_c if s['best_prompt_cat'] == gt_c)
        n_identical = sum(1 for s in samples_c if s['identical'])
        avg_pair = np.mean([s['avg_pairwise_iou'] for s in samples_c])
        print(f'    cls{gt_c} ({CAT_NAMES[gt_c]}, n={n_c}):')
        print(f'      correct prompt best: {n_correct}/{n_c} ({n_correct/n_c*100:.1f}%)')
        print(f'      identical masks: {n_identical}/{n_c} ({n_identical/n_c*100:.1f}%)')
        print(f'      mean pairwise IoU: {avg_pair*100:.1f}%')

    # ── save JSON ────────────────────────────────────────────────────────
    results = {
        'summary': {
            'total_samples': n,
            'all_identical_rate': round(identical_rate, 4),
            'mean_pairwise_iou': round(mean_pair_iou, 4),
            'correct_prompt_best_rate': round(correct_best_rate, 4),
            'gt_iou_per_prompt': {
                str(c): round(float(np.mean(gt_ious_by_prompt[c])), 4)
                if gt_ious_by_prompt[c] else 0
                for c in [1, 2, 3, 4]
            },
        },
        'per_sample': per_sample,
    }
    json_path = os.path.join(args.output_dir, 'language_disc_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to: {json_path}')


if __name__ == '__main__':
    main()
