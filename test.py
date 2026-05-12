import torch
import torch.utils.data

import numpy as np

from lib import segmentation
from data.dataset_lung_nodule import LungNoduleDataset
import transforms as T
import utils


def get_transform(args):
    return T.Compose([
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    seg_correct = np.zeros(len(iou_thresholds), dtype=np.int64)

    iou_chunks = []
    cum_inter = torch.zeros((), dtype=torch.float64, device=device)
    cum_union = torch.zeros((), dtype=torch.float64, device=device)
    n_pos, n_neg, n_tn = 0, 0, 0

    for image, target in metric_logger.log_every(data_loader, 100, header):
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits = model(image)                          # (B, 2, H, W)
        pred = logits.argmax(dim=1)                    # (B, H, W) {0, 1}

        target_flat = target.flatten(1)
        pred_flat = pred.flatten(1)
        is_pos = target_flat.any(dim=1)

        inter = (pred_flat * target_flat).sum(dim=1).double()
        union = (pred_flat.sum(dim=1) + target_flat.sum(dim=1)).double() - inter
        iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))

        if is_pos.any():
            iou_pos = iou[is_pos]
            iou_chunks.append(iou_pos.cpu())
            cum_inter += inter[is_pos].sum()
            cum_union += union[is_pos].sum()
            n_pos += int(is_pos.sum().item())
            iou_np = iou_pos.cpu().numpy()
            for k, thr in enumerate(iou_thresholds):
                seg_correct[k] += int((iou_np >= thr).sum())

        neg = ~is_pos
        if neg.any():
            pred_neg_sum = pred_flat[neg].sum(dim=1)
            n_tn += int((pred_neg_sum == 0).sum().item())
            n_neg += int(neg.sum().item())

    mean_iou = (torch.cat(iou_chunks).mean().item() if iou_chunks else 0.0) * 100.0
    overall_iou = ((cum_inter / cum_union).item() * 100.0) if cum_union.item() > 0 else 0.0
    tn_rate = (n_tn / n_neg) * 100.0 if n_neg > 0 else 0.0

    print('Final results:')
    print(f'  Mean IoU:    {mean_iou:.2f}  ({n_pos} positive samples)')
    print(f'  Overall IoU: {overall_iou:.2f}')
    print(f'  TN rate:     {tn_rate:.2f}  ({n_tn}/{n_neg} negatives all-zero)')
    for thr, ok in zip(iou_thresholds, seg_correct):
        prec = (100.0 * ok / n_pos) if n_pos > 0 else 0.0
        print(f'  precision@{thr:.1f}: {prec:.2f}')


def main(args):
    device = torch.device(args.device)

    dataset_test = LungNoduleDataset(
        data_root=args.data_root,
        split=args.split,
        transforms=get_transform(args),
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.pin_mem,
    )

    n_neg_in_set = len(dataset_test.samples) - len(dataset_test.positives)
    print(f'Model: {args.model}')
    print(f'Test split "{args.split}": {len(dataset_test)} samples '
          f'({len(dataset_test.positives)} pos + {n_neg_in_set} neg)')

    model = segmentation.__dict__[args.model](pretrained='', args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    evaluate(model, data_loader, device=device)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print(f'Image size: {args.img_size}')
    main(args)
