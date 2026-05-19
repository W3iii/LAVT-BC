import numpy as np
from PIL import Image, ImageFilter, ImageOps
import random

import torch
import torch.nn.functional as Fnn
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(object):
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, image, target):
        image = F.resize(image, (self.h, self.w))
        # If size is a sequence like (h, w), the output size will be matched to this.
        # If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio
        target = F.resize(target, (self.h, self.w), interpolation=Image.NEAREST)
        return image, target


class PadOrCropToSize(object):
    """
    Deterministically fit a lobe-cropped slice into a fixed HxW canvas.

    If the slice exceeds the target size, crop around the label center for
    positive slices and around the image center for normal slices. Then pad
    any short side with zeros.
    """

    def __init__(self, h, w, image_fill=0, target_fill=0):
        self.h = h
        self.w = w
        self.image_fill = image_fill
        self.target_fill = target_fill

    @staticmethod
    def _label_center(target):
        arr = np.asarray(target)
        ys, xs = np.nonzero(arr > 0)
        if xs.size == 0:
            return None
        return float(xs.mean()), float(ys.mean())

    @staticmethod
    def _crop_start(center, length, target_length):
        if length <= target_length:
            return 0
        start = int(round(center - target_length / 2.0))
        return max(0, min(start, length - target_length))

    def __call__(self, image, target):
        w, h = image.size
        label_center = self._label_center(target)
        if label_center is None:
            cx, cy = w / 2.0, h / 2.0
        else:
            cx, cy = label_center

        left = self._crop_start(cx, w, self.w)
        top = self._crop_start(cy, h, self.h)
        right = left + min(w, self.w)
        bottom = top + min(h, self.h)
        if left != 0 or top != 0 or right != w or bottom != h:
            image = image.crop((left, top, right, bottom))
            target = target.crop((left, top, right, bottom))

        w, h = image.size
        pad_left = max((self.w - w) // 2, 0)
        pad_top = max((self.h - h) // 2, 0)
        pad_right = max(self.w - w - pad_left, 0)
        pad_bottom = max(self.h - h - pad_top, 0)
        if pad_left or pad_top or pad_right or pad_bottom:
            border = (pad_left, pad_top, pad_right, pad_bottom)
            image = ImageOps.expand(image, border=border, fill=self.image_fill)
            target = ImageOps.expand(target, border=border, fill=self.target_fill)

        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)  # Return a random integer N such that a <= N <= b. Alias for randrange(a, b+1)
        image = F.resize(image, size)
        # If size is a sequence like (h, w), the output size will be matched to this.
        # If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.asarray(target).copy(), dtype=torch.int64)
        return image, target


def _affine_with_fill(img, angle, translate, scale, shear,
                      interpolation, fill):
    interpolation_mode = interpolation
    mode_cls = getattr(T, "InterpolationMode", None)
    if mode_cls is not None:
        if interpolation == Image.BILINEAR:
            interpolation_mode = mode_cls.BILINEAR
        elif interpolation == Image.NEAREST:
            interpolation_mode = mode_cls.NEAREST
    try:
        return F.affine(
            img,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=interpolation_mode,
            fill=fill,
        )
    except (TypeError, ValueError):
        return F.affine(
            img,
            angle,
            translate,
            scale,
            shear,
            resample=interpolation,
            fillcolor=fill,
        )


class RandomMildAffine(object):
    def __init__(self, prob=0.5, degrees=7.0,
                 translate_px=10, scale_range=(0.95, 1.05)):
        self.prob = prob
        self.degrees = degrees
        self.translate_px = translate_px
        self.scale_range = scale_range

    def __call__(self, image, target):
        if random.random() >= self.prob:
            return image, target

        angle = random.uniform(-self.degrees, self.degrees)
        translate = [
            random.randint(-self.translate_px, self.translate_px),
            random.randint(-self.translate_px, self.translate_px),
        ]
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        shear = [0.0, 0.0]

        image = _affine_with_fill(
            image, angle, translate, scale, shear,
            Image.BILINEAR, 0)
        target = _affine_with_fill(
            target, angle, translate, scale, shear,
            Image.NEAREST, 0)
        return image, target


class RandomAffine(object):
    def __init__(self, angle, translate, scale, shear, resample=0, fillcolor=None):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.resample = resample
        self.fillcolor = fillcolor

    def __call__(self, image, target):
        affine_params = T.RandomAffine.get_params(self.angle, self.translate, self.scale, self.shear, image.size)
        image = F.affine(image, *affine_params)
        target = F.affine(target, *affine_params)
        return image, target


class RandomGaussianBlur(object):
    def __init__(self, prob=0.1, sigma_max=0.5):
        self.prob = prob
        self.sigma_max = sigma_max

    def __call__(self, image, target):
        if random.random() < self.prob:
            sigma = random.uniform(0.1, self.sigma_max)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        return image, target


class Clip01(object):
    def __call__(self, image, target):
        return image.clamp(0.0, 1.0), target


class RandomIntensityShiftScale(object):
    def __init__(self, shift=0.05, scale=0.1,
                 shift_prob=0.5, scale_prob=0.5):
        self.shift = shift
        self.scale = scale
        self.shift_prob = shift_prob
        self.scale_prob = scale_prob

    def __call__(self, image, target):
        if random.random() < self.scale_prob:
            factor = random.uniform(1.0 - self.scale, 1.0 + self.scale)
            image = image * factor
        if random.random() < self.shift_prob:
            offset = random.uniform(-self.shift, self.shift)
            image = image + offset
        return image.clamp(0.0, 1.0), target


class RandomGaussianNoise(object):
    def __init__(self, prob=0.3, std=0.015):
        self.prob = prob
        self.std = std

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image + torch.randn_like(image) * self.std
        return image.clamp(0.0, 1.0), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


# ── nnUNet v2-style augmentation (2D) ──────────────────────────────────────
# Pipeline order mirrors get_training_transforms() in nnUNetv2:
#   SpatialTransform → GaussianNoise → GaussianBlur →
#   MultiplicativeBrightness → Contrast → SimulateLowResolution →
#   GammaInverted → Gamma → Mirror
# Probabilities and ranges match nnUNetv2 defaults except Mirror (H-only)
# and rotation (±15° instead of ±180°) per chest CT anatomy.


class SpatialTransform2D(object):
    """Rotation + scaling on the model-input canvas. fill=0 in image and mask.

    Rotation and scaling roll independently (matches nnUNet v2's
    p_rotation / p_scaling). Pure pixel translation is NOT applied
    (nnUNet uses patch_center_dist_from_border=0 with random_crop=False,
    which leaves the center fixed).
    """

    def __init__(self, rotation_deg=15.0, scaling_range=(0.7, 1.4),
                 p_rotation=0.2, p_scaling=0.2):
        self.rotation_deg = rotation_deg
        self.scaling_range = scaling_range
        self.p_rotation = p_rotation
        self.p_scaling = p_scaling

    def __call__(self, image, target):
        angle = 0.0
        scale = 1.0
        if random.random() < self.p_rotation:
            angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        if random.random() < self.p_scaling:
            scale = random.uniform(*self.scaling_range)

        if angle == 0.0 and scale == 1.0:
            return image, target

        image = _affine_with_fill(image, angle, [0, 0], scale, [0.0, 0.0],
                                  Image.BILINEAR, 0)
        target = _affine_with_fill(target, angle, [0, 0], scale, [0.0, 0.0],
                                   Image.NEAREST, 0)
        return image, target


class RandomGaussianNoiseV2(object):
    """nnUNet-style additive Gaussian noise. variance ~ U(0, max_variance)."""

    def __init__(self, prob=0.1, max_variance=0.1):
        self.prob = prob
        self.max_variance = max_variance

    def __call__(self, image, target):
        if random.random() < self.prob:
            variance = random.uniform(0.0, self.max_variance)
            std = variance ** 0.5
            image = image + torch.randn_like(image) * std
            image = image.clamp(0.0, 1.0)
        return image, target


class RandomGaussianBlurTensor(object):
    """Per-call sigma; nnUNet uses sigma ~ U(0.5, 1.0) with p=0.2."""

    def __init__(self, prob=0.2, sigma_range=(0.5, 1.0), kernel_size=5):
        self.prob = prob
        self.sigma_range = sigma_range
        self.kernel_size = kernel_size

    def __call__(self, image, target):
        if random.random() < self.prob:
            sigma = random.uniform(*self.sigma_range)
            image = F.gaussian_blur(image, kernel_size=self.kernel_size,
                                    sigma=sigma)
        return image, target


class RandomMultiplicativeBrightness(object):
    def __init__(self, prob=0.15, multiplier_range=(0.75, 1.25)):
        self.prob = prob
        self.multiplier_range = multiplier_range

    def __call__(self, image, target):
        if random.random() < self.prob:
            mult = random.uniform(*self.multiplier_range)
            image = (image * mult).clamp(0.0, 1.0)
        return image, target


class RandomContrast(object):
    """(img - mean) * factor + mean, clipped to original range."""

    def __init__(self, prob=0.15, contrast_range=(0.75, 1.25),
                 preserve_range=True):
        self.prob = prob
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range

    def __call__(self, image, target):
        if random.random() < self.prob:
            factor = random.uniform(*self.contrast_range)
            mean = image.mean()
            if self.preserve_range:
                lo, hi = image.min().item(), image.max().item()
            image = (image - mean) * factor + mean
            if self.preserve_range:
                image = image.clamp(lo, hi)
        return image, target


class RandomSimulateLowResolution(object):
    """Downsample with nearest then upsample with bicubic — mimics
    scanner low-res artifacts. Mask is left untouched."""

    def __init__(self, prob=0.25, scale_range=(0.5, 1.0)):
        self.prob = prob
        self.scale_range = scale_range

    def __call__(self, image, target):
        if random.random() < self.prob:
            scale = random.uniform(*self.scale_range)
            if scale >= 0.999:
                return image, target
            _, h, w = image.shape
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            img = image.unsqueeze(0)
            img = Fnn.interpolate(img, size=(new_h, new_w), mode='nearest')
            img = Fnn.interpolate(img, size=(h, w), mode='bicubic',
                                  align_corners=False)
            image = img.squeeze(0).clamp(0.0, 1.0)
        return image, target


class RandomGammaTransform(object):
    """Gamma correction. If ``invert``, applies on inverted image then
    inverts back (nnUNet's GammaInverted variant).

    Retain-stats matches nnUNet: re-normalize output to the input
    image's pre-gamma mean/std.
    """

    def __init__(self, prob=0.3, gamma_range=(0.7, 1.5),
                 invert=False, retain_stats=True):
        self.prob = prob
        self.gamma_range = gamma_range
        self.invert = invert
        self.retain_stats = retain_stats

    def __call__(self, image, target):
        if random.random() >= self.prob:
            return image, target

        gamma = random.uniform(*self.gamma_range)
        if self.retain_stats:
            mean_in = image.mean()
            std_in = image.std()

        if self.invert:
            image = 1.0 - image

        image = image.clamp(min=1e-7).pow(gamma)

        if self.invert:
            image = 1.0 - image

        if self.retain_stats:
            mean_out = image.mean()
            std_out = image.std().clamp(min=1e-7)
            image = (image - mean_out) / std_out * std_in + mean_in
            image = image.clamp(0.0, 1.0)
        return image, target

